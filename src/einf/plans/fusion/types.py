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
    runner: RunnerT


RuntimeStepFusions: TypeAlias = tuple[RuntimeStepFusion[TupleRunner], ...]


@dataclass(frozen=True, slots=True)
class TupleFusionRule:
    """One tuple-runner fusion rule for a fixed window size."""

    name: str
    window_size: int
    build_runner: Callable[[RuntimeSteps], TupleRunner | None]


__all__ = [
    "RuntimeStepFusion",
    "RuntimeStepFusions",
    "RuntimeSteps",
    "SingleOutputRunner",
    "TupleFusionRule",
    "TupleRunner",
]
