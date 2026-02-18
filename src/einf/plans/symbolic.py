from dataclasses import dataclass, field

from einf.plans.context import PlanSelectionContext
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    StepProgram,
    SymbolicStep,
)
from einf.tensor_types import TensorLike

from .cache import RuntimeStepSpecializationCache
from .scoring import SymbolicPlanScore


@dataclass(frozen=True, slots=True)
class SymbolicPlan:
    """Symbolic program composed of ordered symbolic steps."""

    kind: str
    input_arity: int
    output_arity: int
    steps: tuple[SymbolicStep[StepProgram], ...]
    _runtime_step_cache: RuntimeStepSpecializationCache = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate symbolic step chain arities."""
        current_arity = self.input_arity
        for step in self.steps:
            if step.program is None:
                raise ValueError(
                    "symbolic step "
                    f"{type(step).__name__} must embed a non-null `program` field"
                )
            if step.input_arity != current_arity:
                raise ValueError(
                    f"symbolic step {step.name!r} expects {step.input_arity} inputs "
                    f"but current arity is {current_arity}"
                )
            current_arity = step.output_arity
        if current_arity != self.output_arity:
            raise ValueError(
                f"symbolic plan output arity mismatch: expected {self.output_arity}, got {current_arity}"
            )
        object.__setattr__(
            self,
            "_runtime_step_cache",
            RuntimeStepSpecializationCache(
                depends_on_input_shapes=any(
                    step.specialization_depends_on_input_shapes() for step in self.steps
                )
            ),
        )

    def _specialize_runtime_steps(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> tuple[RuntimeStep[StepProgram], ...]:
        """Specialize symbolic steps once per runtime cache key."""
        cached_steps = self._runtime_step_cache.get(context)
        if cached_steps is not None:
            return cached_steps

        runtime_steps = tuple(step.specialize(context) for step in self.steps)
        for symbolic_step, runtime_step in zip(
            self.steps,
            runtime_steps,
            strict=True,
        ):
            if runtime_step.program is None:
                raise ValueError(
                    "runtime step "
                    f"{type(runtime_step).__name__} must embed a non-null `program` field"
                )
            if runtime_step.input_arity != symbolic_step.input_arity:
                raise ValueError(
                    "runtime step input arity drift: "
                    f"{runtime_step.input_arity} != {symbolic_step.input_arity} "
                    f"for step {symbolic_step.name!r}"
                )
            if runtime_step.output_arity != symbolic_step.output_arity:
                raise ValueError(
                    "runtime step output arity drift: "
                    f"{runtime_step.output_arity} != {symbolic_step.output_arity} "
                    f"for step {symbolic_step.name!r}"
                )
        self._runtime_step_cache.set(context=context, steps=runtime_steps)
        return runtime_steps

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> tuple[RuntimeStep[StepProgram], ...]:
        """Specialize symbolic steps once for one runtime context."""
        return self._specialize_runtime_steps(context)

    def _run_runtime_steps(
        self,
        *,
        runtime_steps: tuple[RuntimeStep[StepProgram], ...],
        current: tuple[TensorLike, ...],
    ) -> tuple[TensorLike, ...]:
        """Run specialized runtime-step chain with unary/binary dispatch shortcuts."""
        for runtime_step in runtime_steps:
            if (
                len(current) == 1
                and runtime_step.input_arity == 1
                and runtime_step.output_arity == 1
            ):
                current = (runtime_step.run_unary(current[0]),)
                continue
            if (
                len(current) == 2
                and runtime_step.input_arity == 2
                and runtime_step.output_arity == 1
            ):
                current = (runtime_step.run_binary(current[0], current[1]),)
                continue
            current = runtime_step.run(current)
        return current

    def execute(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        """Run specialized runtime steps with strict arity checks."""
        if len(tensors) != self.input_arity:
            raise ValueError(
                "runtime step input arity mismatch: "
                f"expected {self.input_arity}, got {len(tensors)}"
            )

        runtime_steps = self._specialize_runtime_steps(context)
        if not runtime_steps:
            current = tensors
        else:
            current = self._run_runtime_steps(
                runtime_steps=runtime_steps, current=tensors
            )
        if len(current) != self.output_arity:
            raise ValueError(
                "runtime step chain output arity mismatch: "
                f"expected {self.output_arity}, got {len(current)}"
            )
        return current

    def execute_single_output(
        self,
        context: RuntimeSpecializationContext,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> TensorLike:
        """Execute one symbolic plan that must produce exactly one output tensor."""
        if len(tensors) != self.input_arity:
            raise ValueError(
                "runtime step input arity mismatch: "
                f"expected {self.input_arity}, got {len(tensors)}"
            )
        if self.output_arity != 1:
            raise ValueError(
                "runtime step output arity mismatch: "
                f"expected 1, got {self.output_arity}"
            )

        runtime_steps = self._specialize_runtime_steps(context)
        if not runtime_steps:
            if self.input_arity != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    "cannot emit one output without runtime steps"
                )
            return tensors[0]

        current = self._run_runtime_steps(runtime_steps=runtime_steps, current=tensors)
        if len(current) != 1:
            raise ValueError(
                "runtime step chain output arity mismatch: "
                f"expected 1, got {len(current)}"
            )
        return current[0]

    def score(self, context: PlanSelectionContext, /) -> SymbolicPlanScore:
        """Estimate deterministic symbolic-plan score for candidate selection."""
        peak_einsum_numel = 0
        pre_einsum_materialize_numel = 0
        post_einsum_materialize_numel = 0
        allocation_count = 0
        kernel_count = 0
        seen_einsum = False

        for step in self.steps:
            step_score = step.score(context)
            peak_einsum_numel = max(
                peak_einsum_numel,
                step_score.peak_einsum_numel,
            )
            allocation_count += step_score.allocation_count
            kernel_count += step_score.kernel_count

            if step_score.peak_einsum_numel > 0:
                seen_einsum = True
            elif seen_einsum:
                post_einsum_materialize_numel += step_score.materialize_numel
            else:
                pre_einsum_materialize_numel += step_score.materialize_numel

        return SymbolicPlanScore(
            peak_einsum_numel=peak_einsum_numel,
            pre_einsum_materialize_numel=pre_einsum_materialize_numel,
            post_einsum_materialize_numel=post_einsum_materialize_numel,
            allocation_count=allocation_count,
            kernel_count=kernel_count,
            step_count=len(self.steps),
        )


__all__ = ["SymbolicPlan"]
