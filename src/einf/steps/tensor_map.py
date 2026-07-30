from dataclasses import dataclass, field

from einf.steps.base import (
    RuntimeProgram,
    RuntimeSpecializationContext,
    RuntimeStep,
    StepProgram,
    SymbolicProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.steps.context import PlanSelectionContext
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class TensorMapSymbolicProgram(SymbolicProgram):
    """Independent unary symbolic-step chains indexed by tensor position."""

    chains: tuple[tuple[SymbolicStep[StepProgram], ...], ...]

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("tensor map requires at least one unary chain")
        for chain in self.chains:
            for step in chain:
                if step.input_arity != 1 or step.output_arity != 1:
                    raise ValueError("tensor map child steps must be unary")
                if step.specialization_depends_on_input_shapes():
                    raise ValueError(
                        "tensor map child steps must specialize independently of shapes"
                    )


@dataclass(frozen=True, slots=True)
class TensorMapRuntimeProgram(RuntimeProgram):
    """Independent unary runtime-step chains indexed by tensor position."""

    chains: tuple[tuple[RuntimeStep[StepProgram], ...], ...]

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("tensor map requires at least one unary chain")
        for chain in self.chains:
            for step in chain:
                if step.input_arity != 1 or step.output_arity != 1:
                    raise ValueError("tensor map child steps must be unary")

    def __call__(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != len(self.chains):
            raise ValueError(
                "tensor map input arity mismatch: "
                f"expected {len(self.chains)}, got {len(tensors)}"
            )

        outputs: list[TensorLike] = []
        for tensor, chain in zip(tensors, self.chains, strict=True):
            output = tensor
            for step in chain:
                output = step.run_unary(output)
            outputs.append(output)
        return tuple(outputs)


@dataclass(frozen=True, slots=True)
class TensorMapRuntimeStep(RuntimeStep[TensorMapRuntimeProgram]):
    """Runtime step applying one unary chain to each input tensor."""

    program: TensorMapRuntimeProgram
    name: str = "tensor_map"
    input_arity: int = field(init=False)
    output_arity: int = field(init=False)

    def __post_init__(self) -> None:
        arity = len(self.program.chains)
        object.__setattr__(self, "input_arity", arity)
        object.__setattr__(self, "output_arity", arity)

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        return self.program(tensors)


@dataclass(frozen=True, slots=True)
class TensorMapSymbolicStep(SymbolicStep[TensorMapSymbolicProgram]):
    """Symbolic step applying one shape-independent unary chain per tensor."""

    program: TensorMapSymbolicProgram
    name: str = "tensor_map"
    input_arity: int = field(init=False)
    output_arity: int = field(init=False)

    def __post_init__(self) -> None:
        arity = len(self.program.chains)
        object.__setattr__(self, "input_arity", arity)
        object.__setattr__(self, "output_arity", arity)

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        child_context = RuntimeSpecializationContext(
            input_shapes=((),),
            backend_profile=context.backend_profile,
        )
        runtime_chains = tuple(
            tuple(step.specialize(child_context) for step in chain)
            for chain in self.program.chains
        )
        return TensorMapRuntimeStep(
            program=TensorMapRuntimeProgram(chains=runtime_chains)
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        child_context = PlanSelectionContext(
            input_shapes=((),),
            explicit_sizes=context.explicit_sizes,
        )
        scores = tuple(
            step.score(child_context) for chain in self.program.chains for step in chain
        )
        return SymbolicStepScore(
            peak_einsum_numel=max(
                (score.peak_einsum_numel for score in scores),
                default=0,
            ),
            materialize_numel=sum(score.materialize_numel for score in scores),
            allocation_count=sum(score.allocation_count for score in scores),
            kernel_count=sum(score.kernel_count for score in scores),
        )

    def requires_einsum_backend(self) -> bool:
        """Return whether any nested unary chain requires einsum."""
        return any(
            step.requires_einsum_backend()
            for chain in self.program.chains
            for step in chain
        )


__all__ = [
    "TensorMapRuntimeProgram",
    "TensorMapRuntimeStep",
    "TensorMapSymbolicProgram",
    "TensorMapSymbolicStep",
]
