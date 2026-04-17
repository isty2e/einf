from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from einf.steps.base import RuntimeStep, StepProgram
from einf.tensor_types import TensorLike

TupleRunner: TypeAlias = Callable[[tuple[TensorLike, ...]], tuple[TensorLike, ...]]
SingleOutputRunner: TypeAlias = Callable[[tuple[TensorLike, ...]], TensorLike]
RuntimeSteps: TypeAlias = tuple[RuntimeStep[StepProgram], ...]

RunnerT = TypeVar("RunnerT")


@dataclass(frozen=True, slots=True)
class RuntimeStepFusion(Generic[RunnerT]):
    """One fused consecutive runtime-step segment."""

    name: str
    start: int
    stop: int
    input_arity: int
    output_arity: int
    tuple_runner: TupleRunner


RuntimeStepFusions: TypeAlias = tuple[RuntimeStepFusion[TupleRunner], ...]


@dataclass(frozen=True, slots=True)
class RuntimeStepFusionRule:
    """One runtime-step fusion rule for a fixed consecutive window."""

    name: str
    window_size: int
    build_tuple_runner: Callable[[RuntimeSteps], TupleRunner | None]


__all__ = [
    "RuntimeStepFusion",
    "RuntimeStepFusionRule",
    "RuntimeStepFusions",
    "RuntimeSteps",
    "SingleOutputRunner",
    "TupleRunner",
]
