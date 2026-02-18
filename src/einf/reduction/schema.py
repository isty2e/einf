from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..axis import AxisTermBase, AxisTerms
from ..tensor_types import TensorLike

ReducerResult: TypeAlias = TensorLike | bool | int | float | complex
StringReducer: TypeAlias = Literal["sum", "prod", "mean", "max", "min", "all", "any"]
ReducerCallable: TypeAlias = Callable[..., ReducerResult]
Reducer: TypeAlias = str | ReducerCallable


@dataclass(frozen=True, slots=True)
class ReducerPhase:
    """One normalized reduction phase."""

    axes: AxisTerms | tuple[AxisTermBase | int, ...]
    reducer: Reducer

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", AxisTerms.from_spec(self.axes))


ReducerPlan: TypeAlias = tuple[ReducerPhase, ...]
STRING_REDUCERS = frozenset(("sum", "prod", "mean", "max", "min", "all", "any"))
