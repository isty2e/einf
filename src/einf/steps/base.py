from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isabstract
from typing import Generic, TypeVar

from einf.axis import AxisSide
from einf.backend import BackendProfile
from einf.plans.context import PlanSelectionContext
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class SymbolicStepScore:
    """Deterministic local symbolic-step score."""

    peak_einsum_numel: int
    materialize_numel: int
    allocation_count: int
    kernel_count: int


@dataclass(frozen=True, slots=True)
class RuntimeSpecializationContext:
    """Call-time specialization context for symbolic steps."""

    input_shapes: tuple[tuple[int, ...], ...]
    backend_profile: BackendProfile | None


class StepProgram(ABC):
    """Base program contract shared by symbolic/runtime step programs."""


class RuntimeProgram(StepProgram):
    """Runtime callable mapping tensors to tensors."""

    def __call__(self, tensors: tuple[TensorLike, ...], /) -> tuple[TensorLike, ...]:
        """Run this runtime program."""
        raise NotImplementedError


class SymbolicProgram(StepProgram):
    """Symbolic callable mapping runtime context to runtime program."""

    def __call__(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeProgram:
        """Specialize this symbolic program for one runtime context."""
        raise NotImplementedError


StepProgramT = TypeVar("StepProgramT", bound=StepProgram, covariant=True)


class RuntimeStep(Generic[StepProgramT], ABC):
    """Executable runtime instruction."""

    name: str
    input_arity: int
    output_arity: int
    program: StepProgramT

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if isabstract(cls):
            return
        if "run" not in cls.__dict__:
            return
        program_annotation = cls.__dict__.get("__annotations__", {}).get("program")
        if not isinstance(program_annotation, type) or not issubclass(
            program_annotation,
            StepProgram,
        ):
            raise TypeError(
                f"{cls.__name__} must annotate `program` as a StepProgram subtype"
            )

    @abstractmethod
    def run(self, tensors: tuple[TensorLike, ...], /) -> tuple[TensorLike, ...]:
        """Execute one runtime step."""
        raise NotImplementedError

    def run_single_output(self, tensors: tuple[TensorLike, ...], /) -> TensorLike:
        """Execute one runtime step that must produce exactly one output tensor."""
        outputs = self.run(tensors)
        if len(outputs) != 1:
            raise ValueError(
                f"runtime step output arity mismatch: expected 1, got {len(outputs)}"
            )
        return outputs[0]

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one unary runtime step that must produce exactly one output."""
        if self.input_arity != 1:
            raise ValueError(
                f"runtime step input arity mismatch: expected 1, got {self.input_arity}"
            )
        return self.run_single_output((tensor,))

    def run_binary(self, lhs: TensorLike, rhs: TensorLike, /) -> TensorLike:
        """Execute one binary runtime step that must produce exactly one output."""
        if self.input_arity != 2:
            raise ValueError(
                f"runtime step input arity mismatch: expected 2, got {self.input_arity}"
            )
        return self.run_single_output((lhs, rhs))


class SymbolicStep(Generic[StepProgramT], ABC):
    """Symbolic instruction that can specialize to a runtime step."""

    name: str
    input_arity: int
    output_arity: int
    program: StepProgramT

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if isabstract(cls):
            return
        if "specialize" not in cls.__dict__:
            return
        program_annotation = cls.__dict__.get("__annotations__", {}).get("program")
        if not isinstance(program_annotation, type) or not issubclass(
            program_annotation,
            StepProgram,
        ):
            raise TypeError(
                f"{cls.__name__} must annotate `program` as a StepProgram subtype"
            )

    @abstractmethod
    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        """Estimate deterministic symbolic-step score under current context."""
        raise NotImplementedError

    def specialization_depends_on_input_shapes(self) -> bool:
        """Return whether step specialization changes with runtime input shapes."""
        return False

    @abstractmethod
    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep[StepProgram]:
        """Specialize one symbolic step using call-time context."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AxisSideSymbolicStep(SymbolicStep[StepProgramT]):
    """Base symbolic step for operations defined by `(lhs, rhs)` sides."""

    lhs: AxisSide
    rhs: AxisSide
    name: str
    explicit_sizes_items: tuple[tuple[str, int], ...] = ()
    input_arity: int = field(init=False)
    output_arity: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_arity", len(self.lhs))
        object.__setattr__(self, "output_arity", len(self.rhs))

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=0,
            allocation_count=0,
            kernel_count=1,
        )


__all__ = [
    "AxisSideSymbolicStep",
    "RuntimeProgram",
    "RuntimeSpecializationContext",
    "RuntimeStep",
    "StepProgram",
    "SymbolicProgram",
    "SymbolicStepScore",
    "SymbolicStep",
]
