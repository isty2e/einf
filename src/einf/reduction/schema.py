from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from ..axis import AxisTermBase, AxisTerms
from ..diagnostics import ErrorCode, ValidationError
from ..tensor_types import TensorLike

ReducerResult: TypeAlias = TensorLike | bool | int | float | complex
ReducerCallable: TypeAlias = Callable[..., ReducerResult]
Reducer: TypeAlias = str | ReducerCallable


class ReducerName(str, Enum):
    """Closed vocabulary of backend-native reducer names."""

    SUM = "sum"
    PROD = "prod"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    ALL = "all"
    ANY = "any"


CanonicalReducer: TypeAlias = ReducerName | ReducerCallable


@dataclass(frozen=True, slots=True, init=False)
class ReducerPhase:
    """One normalized reduction phase."""

    axes: AxisTerms
    reducer: CanonicalReducer

    def __init__(
        self,
        axes: AxisTerms | tuple[AxisTermBase | int, ...],
        reducer: Reducer,
    ) -> None:
        normalized_reducer: CanonicalReducer
        if isinstance(reducer, str):
            try:
                normalized_reducer = ReducerName(reducer)
            except ValueError as error:
                supported_names = ", ".join(name.value for name in ReducerName)
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=f"inconsistent dims: unsupported reducer {reducer!r}",
                    help=f"use one of: {supported_names}",
                    related=("reduce reducer",),
                    data={"reducer": reducer},
                ) from error
        elif callable(reducer):
            normalized_reducer = reducer
        else:
            raise TypeError("reducer must be a string or callable")

        object.__setattr__(self, "axes", AxisTerms.from_spec(axes))
        object.__setattr__(self, "reducer", normalized_reducer)


ReducerPlan: TypeAlias = tuple[ReducerPhase, ...]
