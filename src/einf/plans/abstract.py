from dataclasses import dataclass, field

from einf.axis import AxisSide
from einf.backend import BACKEND_RESOLVER, BackendProfile
from einf.diagnostics import ErrorCode, ValidationError
from einf.ir import IRProgram
from einf.ir.routing.runtime import resolve_route_output_indices, route_outputs
from einf.ir.routing.static import precompute_route_output_indices
from einf.plans.context import PlanSelectionContext
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram
from einf.tensor_types import TensorLike

from ..lowering import LoweringProgram
from .cache import (
    BackendProfileCache,
    RouteOutputIndexCache,
    RunnerCache,
    RunnerCacheKey,
    SelectionCache,
    SelectionCacheKey,
)
from .fusion import (
    RuntimeStepFusions,
    SingleOutputRunner,
    TupleRunner,
    discover_tuple_step_fusions,
)
from .symbolic import SymbolicPlan


@dataclass(frozen=True, slots=True)
class AbstractPlan:
    """Ingress-normalized operation model independent from runtime context."""

    op_name: str
    lhs: AxisSide
    rhs: AxisSide
    explicit_sizes_items: tuple[tuple[str, int], ...]
    lowering: LoweringProgram
    ir_program: IRProgram = field(init=False)
    symbolic_candidates: tuple[SymbolicPlan, ...] = field(init=False)
    _candidate_indices_by_input_arity: dict[int, tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _selection_cache: SelectionCache = field(
        init=False,
        repr=False,
        compare=False,
    )
    _backend_profile_cache: BackendProfileCache = field(
        init=False,
        repr=False,
        compare=False,
    )
    _route_output_index_cache: RouteOutputIndexCache = field(
        init=False,
        repr=False,
        compare=False,
    )
    _requires_input_shapes_by_input_arity: dict[int, bool] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _explicit_sizes: dict[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _single_output_runner_cache: RunnerCache[SingleOutputRunner] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _tuple_runner_cache: RunnerCache[TupleRunner] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Lower one abstract operation once into deterministic symbolic candidates."""
        ir_program = self.lowering.ir_program(
            op_name=self.op_name,
            lhs=self.lhs,
            rhs=self.rhs,
            explicit_sizes_items=self.explicit_sizes_items,
        )
        object.__setattr__(
            self,
            "ir_program",
            ir_program,
        )
        symbolic_candidates = self.lowering.symbolic_candidates(
            ir_program=ir_program,
            explicit_sizes_items=self.explicit_sizes_items,
        )
        object.__setattr__(
            self,
            "symbolic_candidates",
            symbolic_candidates,
        )
        candidate_indices_by_input_arity: dict[int, list[int]] = {}
        for candidate_index, candidate in enumerate(symbolic_candidates):
            indices = candidate_indices_by_input_arity.get(candidate.input_arity)
            if indices is None:
                candidate_indices_by_input_arity[candidate.input_arity] = [
                    candidate_index
                ]
                continue
            indices.append(candidate_index)
        object.__setattr__(
            self,
            "_candidate_indices_by_input_arity",
            {
                input_arity: tuple(indices)
                for input_arity, indices in candidate_indices_by_input_arity.items()
            },
        )
        object.__setattr__(self, "_explicit_sizes", dict(self.explicit_sizes_items))
        object.__setattr__(self, "_selection_cache", SelectionCache())
        object.__setattr__(self, "_backend_profile_cache", BackendProfileCache())
        object.__setattr__(self, "_single_output_runner_cache", RunnerCache())
        object.__setattr__(self, "_tuple_runner_cache", RunnerCache())
        static_output_indices = precompute_route_output_indices(self.lhs, self.rhs)
        object.__setattr__(
            self,
            "_route_output_index_cache",
            RouteOutputIndexCache(static_output_indices=static_output_indices),
        )
        object.__setattr__(
            self,
            "_requires_input_shapes_by_input_arity",
            self._build_requires_input_shapes_map(
                candidate_indices_by_input_arity={
                    input_arity: tuple(indices)
                    for input_arity, indices in candidate_indices_by_input_arity.items()
                },
                static_route_output_indices=static_output_indices,
            ),
        )

    def _build_requires_input_shapes_map(
        self,
        *,
        candidate_indices_by_input_arity: dict[int, tuple[int, ...]],
        static_route_output_indices: tuple[int, ...] | None,
    ) -> dict[int, bool]:
        """Build one input-shape requirement map by input arity."""
        requires_by_arity: dict[int, bool] = {}
        for input_arity, indices in candidate_indices_by_input_arity.items():
            if len(indices) != 1:
                requires_by_arity[input_arity] = True
                continue
            symbolic_plan = self.symbolic_candidates[indices[0]]
            if symbolic_plan.kind == "route" and not symbolic_plan.steps:
                requires_by_arity[input_arity] = static_route_output_indices is None
                continue
            requires_by_arity[input_arity] = any(
                step.specialization_depends_on_input_shapes()
                for step in symbolic_plan.steps
            )
        return requires_by_arity

    def requires_input_shapes(self, input_arity: int, /) -> bool:
        """Return whether call-time input shapes are required for execution."""
        return self._requires_input_shapes_by_input_arity.get(input_arity, True)

    def select_symbolic_plan(self, context: PlanSelectionContext, /) -> SymbolicPlan:
        """Select one symbolic candidate deterministically for given runtime context."""
        if not self.symbolic_candidates:
            raise ValueError("no symbolic plan candidates are available")

        input_arity = len(context.input_shapes)
        matching_candidate_indices = self._candidate_indices_by_input_arity.get(
            input_arity,
            (),
        )

        if not matching_candidate_indices:
            raise ValueError(
                f"no symbolic plan candidate matches input arity {input_arity}"
            )

        if len(matching_candidate_indices) == 1:
            return self.symbolic_candidates[matching_candidate_indices[0]]

        cache_key = SelectionCacheKey(
            input_shapes=context.input_shapes,
            explicit_sizes=tuple(sorted(context.explicit_sizes.items())),
        )
        cached_index = self._selection_cache.get_index(cache_key)
        if cached_index is not None:
            cached_candidate = self.symbolic_candidates[cached_index]
            if cached_candidate.input_arity == input_arity:
                return cached_candidate

        best_index = min(
            matching_candidate_indices,
            key=lambda candidate_index: (
                self.symbolic_candidates[candidate_index].score(context),
                candidate_index,
            ),
        )
        self._selection_cache.set_index(cache_key, best_index)

        return self.symbolic_candidates[best_index]

    def _resolve_runtime_context(
        self,
        *,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
    ) -> RuntimeSpecializationContext:
        """Resolve backend profile when context was created without one."""
        backend_profile = context.backend_profile
        if backend_profile is not None:
            return context
        return RuntimeSpecializationContext(
            input_shapes=context.input_shapes,
            backend_profile=self.resolve_backend_profile(
                op_name=self.op_name,
                tensors=tensors,
            ),
        )

    def _select_runtime_symbolic_plan(
        self,
        *,
        context: RuntimeSpecializationContext,
    ) -> SymbolicPlan:
        """Select one symbolic plan for already-normalized runtime context."""
        input_arity = len(context.input_shapes)
        matching_candidate_indices = self._candidate_indices_by_input_arity.get(
            input_arity,
            (),
        )
        if not matching_candidate_indices:
            raise ValueError(
                f"no symbolic plan candidate matches input arity {input_arity}"
            )
        if len(matching_candidate_indices) == 1:
            return self.symbolic_candidates[matching_candidate_indices[0]]
        selection_context = PlanSelectionContext(
            input_shapes=context.input_shapes,
            explicit_sizes=self._explicit_sizes,
        )
        return self.select_symbolic_plan(selection_context)

    def _build_runner_cache_key(
        self,
        *,
        tensors: tuple[TensorLike, ...],
        requires_shapes: bool,
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> RunnerCacheKey:
        """Build one deterministic compiled-runner cache key."""
        input_arity = len(tensors)
        if input_arity == 1:
            tensor_types = (type(tensors[0]),)
        elif input_arity == 2:
            tensor_types = (type(tensors[0]), type(tensors[1]))
        else:
            tensor_types = tuple(type(tensor) for tensor in tensors)
        shape_key = input_shapes if requires_shapes else None
        return (tensor_types, shape_key)

    def _run_runtime_step(
        self,
        *,
        runtime_step: RuntimeStep[StepProgram],
        current: tuple[TensorLike, ...],
    ) -> tuple[TensorLike, ...]:
        """Run one runtime step with unary/binary fast dispatch."""
        if (
            len(current) == 1
            and runtime_step.input_arity == 1
            and runtime_step.output_arity == 1
        ):
            return (runtime_step.run_unary(current[0]),)
        if (
            len(current) == 2
            and runtime_step.input_arity == 2
            and runtime_step.output_arity == 1
        ):
            return (runtime_step.run_binary(current[0], current[1]),)
        return runtime_step.run(current)

    def _compile_step_chain_runner(
        self,
        *,
        runtime_steps: tuple[RuntimeStep[StepProgram], ...],
        fusions: RuntimeStepFusions,
    ) -> TupleRunner:
        """Compile one tuple runner with optional fused runtime-step segments."""
        if not runtime_steps:

            def run_identity(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> tuple[TensorLike, ...]:
                return runtime_tensors

            return run_identity
        if (
            len(fusions) == 1
            and fusions[0].start == 0
            and fusions[0].stop == len(runtime_steps)
        ):
            return fusions[0].runner
        if len(runtime_steps) == 1:
            runtime_step = runtime_steps[0]
            if runtime_step.input_arity == 1 and runtime_step.output_arity == 1:
                run_unary_method = runtime_step.run_unary

                def run_unary(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> tuple[TensorLike, ...]:
                    return (run_unary_method(runtime_tensors[0]),)

                return run_unary
            if runtime_step.input_arity == 2 and runtime_step.output_arity == 1:
                run_binary_method = runtime_step.run_binary

                def run_binary(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> tuple[TensorLike, ...]:
                    return (run_binary_method(runtime_tensors[0], runtime_tensors[1]),)

                return run_binary

            def run_single(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> tuple[TensorLike, ...]:
                return runtime_step.run(runtime_tensors)

            return run_single

        fusion_by_start = {fusion.start: fusion for fusion in fusions}

        def run_chain(
            runtime_tensors: tuple[TensorLike, ...], /
        ) -> tuple[TensorLike, ...]:
            current = runtime_tensors
            step_index = 0
            while step_index < len(runtime_steps):
                fusion = fusion_by_start.get(step_index)
                if fusion is not None:
                    if len(current) != fusion.input_arity:
                        raise ValueError(
                            "runtime step fusion input arity mismatch: "
                            f"expected {fusion.input_arity}, got {len(current)}"
                        )
                    current = fusion.runner(current)
                    step_index = fusion.stop
                    continue
                current = self._run_runtime_step(
                    runtime_step=runtime_steps[step_index],
                    current=current,
                )
                step_index += 1
            return current

        return run_chain

    def _compile_tuple_runner(
        self,
        *,
        symbolic_plan: SymbolicPlan,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
    ) -> TupleRunner:
        """Compile one tuple-output runtime runner from one symbolic plan."""
        if symbolic_plan.kind == "route" and not symbolic_plan.steps:
            output_indices = self._resolve_route_output_indices(
                context=context,
                tensors=tensors,
            )

            def run_route(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> tuple[TensorLike, ...]:
                return route_outputs(
                    tensors=runtime_tensors,
                    output_indices=output_indices,
                )

            return run_route

        runtime_steps = symbolic_plan.specialize(context)
        fusions = discover_tuple_step_fusions(runtime_steps)
        return self._compile_step_chain_runner(
            runtime_steps=runtime_steps,
            fusions=fusions,
        )

    def _compile_single_output_runner(
        self,
        *,
        symbolic_plan: SymbolicPlan,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
    ) -> SingleOutputRunner:
        """Compile one single-output runtime runner from one symbolic plan."""
        if symbolic_plan.kind == "route" and not symbolic_plan.steps:
            output_indices = self._resolve_route_output_indices(
                context=context,
                tensors=tensors,
            )
            if len(output_indices) != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    f"expected 1, got {len(output_indices)}"
                )
            output_index = output_indices[0]

            def run_route(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
                return runtime_tensors[output_index]

            return run_route

        runtime_steps = symbolic_plan.specialize(context)
        if not runtime_steps:
            if symbolic_plan.input_arity != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    "cannot emit one output without runtime steps"
                )

            def run_identity(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
                return runtime_tensors[0]

            return run_identity

        fusions = discover_tuple_step_fusions(runtime_steps)

        tuple_runner = self._compile_step_chain_runner(
            runtime_steps=runtime_steps,
            fusions=fusions,
        )

        def run_chain(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
            outputs = tuple_runner(runtime_tensors)
            if len(outputs) != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    f"expected 1, got {len(outputs)}"
                )
            return outputs[0]

        return run_chain

    def execute(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        """Execute this abstract plan through symbolic and runtime stages."""
        runner = self.resolve_tuple_runner(context, tensors)
        return runner(tensors)

    def resolve_tuple_runner(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> TupleRunner:
        """Resolve or compile one cached tuple-output runtime runner."""
        input_arity = len(tensors)
        requires_shapes = self._requires_input_shapes_by_input_arity.get(
            input_arity,
            True,
        )
        runner_cache_key = self._build_runner_cache_key(
            tensors=tensors,
            requires_shapes=requires_shapes,
            input_shapes=context.input_shapes,
        )
        cached_runner = self._tuple_runner_cache.get(runner_cache_key)
        if cached_runner is not None:
            return cached_runner

        runtime_context = self._resolve_runtime_context(
            context=context, tensors=tensors
        )
        if self.op_name == "view":
            self._validate_view_backend_profile(runtime_context)
        symbolic_plan = self._select_runtime_symbolic_plan(context=runtime_context)
        runner = self._compile_tuple_runner(
            symbolic_plan=symbolic_plan,
            context=runtime_context,
            tensors=tensors,
        )
        self._tuple_runner_cache.set(runner_cache_key, runner)
        return runner

    def execute_single_output(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> TensorLike:
        """Execute this abstract plan and return exactly one output tensor."""
        runner = self.resolve_single_output_runner(context, tensors)
        return runner(tensors)

    def resolve_single_output_runner(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> SingleOutputRunner:
        """Resolve or compile one cached single-output runtime runner."""
        input_arity = len(tensors)
        requires_shapes = self._requires_input_shapes_by_input_arity.get(
            input_arity,
            True,
        )
        runner_cache_key = self._build_runner_cache_key(
            tensors=tensors,
            requires_shapes=requires_shapes,
            input_shapes=context.input_shapes,
        )
        cached_runner = self._single_output_runner_cache.get(runner_cache_key)
        if cached_runner is not None:
            return cached_runner

        runtime_context = self._resolve_runtime_context(
            context=context, tensors=tensors
        )
        if self.op_name == "view":
            self._validate_view_backend_profile(runtime_context)
        symbolic_plan = self._select_runtime_symbolic_plan(context=runtime_context)
        runner = self._compile_single_output_runner(
            symbolic_plan=symbolic_plan,
            context=runtime_context,
            tensors=tensors,
        )
        self._single_output_runner_cache.set(runner_cache_key, runner)
        return runner

    def _validate_view_backend_profile(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> None:
        """Validate strict view backend capability before symbolic execution."""
        backend_profile = context.backend_profile
        if backend_profile is None or backend_profile.supports_strict_view:
            return
        raise ValidationError(
            code=ErrorCode.NOT_A_VIEW,
            message=(
                "not a view: view requires affine zero-copy mapping on the "
                "selected backend"
            ),
            help=(
                "use a strict view-capable backend (numpy or torch) and an "
                "affine mapping expressible without materialization"
            ),
            related=("view affine mapping", "backend capability"),
            data={"operation": "view", "backend": backend_profile.namespace_id},
        )

    def _resolve_route_output_indices(
        self,
        *,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
    ) -> tuple[int, ...]:
        """Resolve cached output routing indices for one route-only symbolic plan."""
        input_shapes = context.input_shapes
        if not input_shapes:
            input_shapes = tuple(tensor.shape for tensor in tensors)
        cached_output_indices = self._route_output_index_cache.get(input_shapes)
        if cached_output_indices is not None:
            return cached_output_indices

        output_indices = resolve_route_output_indices(
            lhs=self.lhs,
            rhs=self.rhs,
            explicit_sizes_items=self.explicit_sizes_items,
            tensors=tensors,
            input_shapes=input_shapes,
        )
        self._route_output_index_cache.set(
            input_shapes=input_shapes,
            output_indices=output_indices,
        )
        return output_indices

    def resolve_backend_profile(
        self,
        *,
        op_name: str,
        tensors: tuple[TensorLike, ...],
    ) -> BackendProfile:
        """Resolve and cache backend profile by input runtime tensor types."""
        cached_profile = self._backend_profile_cache.get(tensors)
        if cached_profile is not None:
            return cached_profile
        backend_profile = BACKEND_RESOLVER.resolve(*tensors, op_name=op_name)
        self._backend_profile_cache.set(
            tensors=tensors,
            profile=backend_profile,
        )
        return backend_profile


__all__ = [
    "AbstractPlan",
]
