from dataclasses import dataclass, field

from einf.axis import AxisSide
from einf.backend import (
    BACKEND_POLICY,
    BACKEND_RESOLVER,
    BackendExecutionIdentity,
    BackendProfile,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.ir import IRProgram, LoweringSignature
from einf.ir.routing.static import precompute_route_output_indices
from einf.solver import validate_dimensions
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram
from einf.steps.context import PlanSelectionContext
from einf.tensor_types import TensorLike

from .cache import (
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
    discover_step_fusions,
)
from .lowering_protocol import LoweringProgram
from .routing import resolve_route_output_indices
from .runners import RouteRunnerKernel, RunnerKernel, StepChainRunnerKernel
from .symbolic import SymbolicPlan


@dataclass(slots=True)
class AbstractPlanRuntimeCaches:
    """Mutable runtime memoization owned by one abstract plan."""

    last_validated_input_shapes: tuple[tuple[int, ...], ...] | None = None
    selection: SelectionCache = field(default_factory=SelectionCache)
    route_output_indices: RouteOutputIndexCache | None = None
    single_output_runners: RunnerCache[RunnerCacheKey, SingleOutputRunner] = field(
        default_factory=lambda: RunnerCache()
    )
    tuple_runners: RunnerCache[RunnerCacheKey, TupleRunner] = field(
        default_factory=lambda: RunnerCache()
    )


@dataclass(frozen=True, slots=True)
class AbstractPlan:
    """Represent one source-bound abstract operation and its lowering.

    Parameters
    ----------
    source : LoweringSignature
        Canonical structural operation consumed by lowering.
    lowering : LoweringProgram
        Lowering policy used to produce the IR and symbolic candidates.

    Raises
    ------
    ValueError
        If the lowering output does not carry the same source.
    """

    source: LoweringSignature
    lowering: LoweringProgram
    ir_program: IRProgram = field(init=False)
    symbolic_candidates: tuple[SymbolicPlan, ...] = field(init=False)
    _specialization_depends_on_input_shapes: bool = field(
        init=False,
        repr=False,
        compare=False,
    )
    _explicit_sizes: dict[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime: AbstractPlanRuntimeCaches = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Lower one abstract operation once into deterministic symbolic candidates."""
        ir_program = self.lowering.ir_program(self.source)
        if ir_program.source != self.source:
            raise ValueError(
                "lowering IR source does not match the abstract plan source: "
                f"expected {self.source!r}, got {ir_program.source!r}"
            )
        object.__setattr__(
            self,
            "ir_program",
            ir_program,
        )
        symbolic_candidates = self.lowering.symbolic_candidates(
            ir_program=ir_program,
        )
        for candidate_index, candidate in enumerate(symbolic_candidates):
            if candidate.source != self.source:
                raise ValueError(
                    "symbolic plan source does not match the abstract plan source "
                    f"for candidate {candidate_index}: "
                    f"expected {self.source!r}, got {candidate.source!r}"
                )
        object.__setattr__(
            self,
            "symbolic_candidates",
            symbolic_candidates,
        )
        object.__setattr__(self, "_explicit_sizes", dict(self.explicit_sizes_items))
        static_output_indices = precompute_route_output_indices(self.lhs, self.rhs)
        object.__setattr__(
            self,
            "_runtime",
            AbstractPlanRuntimeCaches(
                route_output_indices=RouteOutputIndexCache(
                    static_output_indices=static_output_indices
                )
            ),
        )
        object.__setattr__(
            self,
            "_specialization_depends_on_input_shapes",
            self._compute_specialization_shape_dependency(
                static_route_output_indices=static_output_indices,
            ),
        )

    @property
    def op_name(self) -> str:
        """Return the operation name used by lowering.

        Returns
        -------
        str
            Operation name from the canonical source.
        """
        return self.source.op_name

    @property
    def lhs(self) -> AxisSide:
        """Return normalized input axis terms.

        Returns
        -------
        AxisSide
            Input side of the canonical signature.
        """
        return self.source.signature.inputs

    @property
    def rhs(self) -> AxisSide:
        """Return normalized output axis terms.

        Returns
        -------
        AxisSide
            Output side of the canonical signature.
        """
        return self.source.signature.outputs

    @property
    def explicit_sizes_items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical explicit axis-size bindings.

        Returns
        -------
        tuple[tuple[str, int], ...]
            Explicit size bindings in canonical order.
        """
        return self.source.explicit_sizes_items

    def _compute_specialization_shape_dependency(
        self,
        *,
        static_route_output_indices: tuple[int, ...] | None,
    ) -> bool:
        """Return whether candidate specialization depends on input shapes."""
        if len(self.symbolic_candidates) != 1:
            return True
        symbolic_plan = self.symbolic_candidates[0]
        if symbolic_plan.kind == "route" and not symbolic_plan.steps:
            return static_route_output_indices is None
        return any(
            step.specialization_depends_on_input_shapes()
            for step in symbolic_plan.steps
        )

    def specialization_depends_on_input_shapes(self, input_arity: int, /) -> bool:
        """Return whether runner specialization depends on concrete input shapes."""
        return (
            input_arity != self.source.input_arity
            or self._specialization_depends_on_input_shapes
        )

    def validate_input_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        /,
    ) -> None:
        """Validate concrete input shapes against the canonical operation contract."""
        if input_shapes == self._runtime.last_validated_input_shapes:
            return
        try:
            validate_dimensions(
                self.source.signature,
                input_shapes,
                explicit_sizes=self._explicit_sizes,
            )
        except ValidationError:
            raise
        except (TypeError, ValueError) as error:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: {error}",
                help="provide input shapes consistent with the operation signature",
                related=("TensorOp input shape contract",),
                data={"operation": self.op_name},
            ) from error
        self._runtime.last_validated_input_shapes = input_shapes

    def select_symbolic_plan(self, context: PlanSelectionContext, /) -> SymbolicPlan:
        """Select one symbolic candidate deterministically for given runtime context."""
        if not self.symbolic_candidates:
            raise ValueError("no symbolic plan candidates are available")

        input_arity = len(context.input_shapes)
        if input_arity != self.source.input_arity:
            raise ValueError(
                f"no symbolic plan candidate matches input arity {input_arity}"
            )

        if len(self.symbolic_candidates) == 1:
            return self.symbolic_candidates[0]

        cache_key = SelectionCacheKey(
            input_shapes=context.input_shapes,
            explicit_sizes=tuple(sorted(context.explicit_sizes.items())),
        )
        cached_index = self._runtime.selection.get_index(cache_key)
        if cached_index is not None:
            return self.symbolic_candidates[cached_index]

        best_index = min(
            range(len(self.symbolic_candidates)),
            key=lambda candidate_index: (
                self.symbolic_candidates[candidate_index].score(context),
                candidate_index,
            ),
        )
        self._runtime.selection.set_index(cache_key, best_index)

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
            backend_profile=self.resolve_backend_profile(tensors),
        )

    def _select_runtime_symbolic_plan(
        self,
        *,
        context: RuntimeSpecializationContext,
    ) -> SymbolicPlan:
        """Select one symbolic plan for already-normalized runtime context."""
        input_arity = len(context.input_shapes)
        if input_arity != self.source.input_arity:
            raise ValueError(
                f"no symbolic plan candidate matches input arity {input_arity}"
            )
        if len(self.symbolic_candidates) == 1:
            return self.symbolic_candidates[0]
        selection_context = PlanSelectionContext(
            input_shapes=context.input_shapes,
            explicit_sizes=self._explicit_sizes,
        )
        return self.select_symbolic_plan(selection_context)

    def _build_runner_cache_key(
        self,
        *,
        tensors: tuple[TensorLike, ...],
        backend_identity: BackendExecutionIdentity,
        specialization_depends_on_shapes: bool,
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
        shape_key = input_shapes if specialization_depends_on_shapes else None
        return (tensor_types, backend_identity, shape_key)

    def _build_step_chain_runner_kernel(
        self,
        *,
        input_arity: int,
        output_arity: int,
        runtime_steps: tuple[RuntimeStep[StepProgram], ...],
        fusions: RuntimeStepFusions,
    ) -> StepChainRunnerKernel:
        """Build one runner kernel for specialized runtime-step chains."""
        return StepChainRunnerKernel(
            _input_arity=input_arity,
            _output_arity=output_arity,
            runtime_steps=runtime_steps,
            fusions=fusions,
        )

    def _build_runner_kernel(
        self,
        *,
        symbolic_plan: SymbolicPlan,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
    ) -> RunnerKernel:
        """Build one runner kernel from one symbolic plan."""
        self._validate_symbolic_plan_backend_profile(
            symbolic_plan=symbolic_plan,
            context=context,
        )
        if symbolic_plan.kind == "route" and not symbolic_plan.steps:
            output_indices = self._resolve_route_output_indices(
                context=context,
                tensors=tensors,
            )
            return RouteRunnerKernel(
                _input_arity=symbolic_plan.input_arity,
                output_indices=output_indices,
            )

        runtime_steps = symbolic_plan.specialize(context)
        fusions = discover_step_fusions(runtime_steps)
        return self._build_step_chain_runner_kernel(
            input_arity=symbolic_plan.input_arity,
            output_arity=symbolic_plan.output_arity,
            runtime_steps=runtime_steps,
            fusions=fusions,
        )

    def _validate_symbolic_plan_backend_profile(
        self,
        *,
        symbolic_plan: SymbolicPlan,
        context: RuntimeSpecializationContext,
    ) -> None:
        """Validate selected-plan requirements against the resolved backend."""
        if not symbolic_plan.requires_einsum_backend:
            return
        backend_profile = context.backend_profile
        if backend_profile is None:
            raise RuntimeError("symbolic plan validation requires a backend profile")
        BACKEND_POLICY.validate_einsum_capability(
            profile=backend_profile,
            op_name=self.op_name,
        )

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
        specialization_depends_on_shapes = self.specialization_depends_on_input_shapes(
            input_arity
        )
        runtime_context = self._resolve_runtime_context(
            context=context, tensors=tensors
        )
        if self.op_name == "view":
            self._validate_view_backend_profile(runtime_context)
        backend_profile = runtime_context.backend_profile
        if backend_profile is None:
            raise RuntimeError("runtime runner resolution requires a backend profile")
        runner_cache_key = self._build_runner_cache_key(
            tensors=tensors,
            backend_identity=backend_profile.execution_identity,
            specialization_depends_on_shapes=specialization_depends_on_shapes,
            input_shapes=runtime_context.input_shapes,
        )
        cached_runner = self._runtime.tuple_runners.get(runner_cache_key)
        if cached_runner is not None:
            return cached_runner

        symbolic_plan = self._select_runtime_symbolic_plan(context=runtime_context)
        runner_kernel = self._build_runner_kernel(
            symbolic_plan=symbolic_plan,
            context=runtime_context,
            tensors=tensors,
        )
        runner = runner_kernel.build_tuple_runner()
        self._runtime.tuple_runners.set(runner_cache_key, runner)
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
        specialization_depends_on_shapes = self.specialization_depends_on_input_shapes(
            input_arity
        )
        runtime_context = self._resolve_runtime_context(
            context=context, tensors=tensors
        )
        if self.op_name == "view":
            self._validate_view_backend_profile(runtime_context)
        backend_profile = runtime_context.backend_profile
        if backend_profile is None:
            raise RuntimeError("runtime runner resolution requires a backend profile")
        runner_cache_key = self._build_runner_cache_key(
            tensors=tensors,
            backend_identity=backend_profile.execution_identity,
            specialization_depends_on_shapes=specialization_depends_on_shapes,
            input_shapes=runtime_context.input_shapes,
        )
        cached_runner = self._runtime.single_output_runners.get(runner_cache_key)
        if cached_runner is not None:
            return cached_runner

        symbolic_plan = self._select_runtime_symbolic_plan(context=runtime_context)
        runner_kernel = self._build_runner_kernel(
            symbolic_plan=symbolic_plan,
            context=runtime_context,
            tensors=tensors,
        )
        runner = runner_kernel.build_single_output_runner()
        self._runtime.single_output_runners.set(runner_cache_key, runner)
        return runner

    def _validate_view_backend_profile(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> None:
        """Validate strict view backend capability before symbolic execution."""
        backend_profile = context.backend_profile
        if backend_profile is None or BACKEND_POLICY.supports_strict_view(
            backend_family=backend_profile.backend_family
        ):
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
        route_output_indices = self._runtime.route_output_indices
        if route_output_indices is None:
            raise RuntimeError("abstract plan route cache is not initialized")
        cached_output_indices = route_output_indices.get(input_shapes)
        if cached_output_indices is not None:
            return cached_output_indices

        output_indices = resolve_route_output_indices(
            lhs=self.lhs,
            rhs=self.rhs,
            explicit_sizes_items=self.explicit_sizes_items,
            tensors=tensors,
            input_shapes=input_shapes,
        )
        route_output_indices.set(
            input_shapes=input_shapes,
            output_indices=output_indices,
        )
        return output_indices

    def resolve_backend_profile(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> BackendProfile:
        """Resolve this plan's backend profile from current runtime tensors."""
        return BACKEND_RESOLVER.resolve(*tensors, op_name=self.op_name)


__all__ = [
    "AbstractPlan",
]
