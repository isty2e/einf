from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TypeAlias

from ..axis import AxisTermBase, AxisTerms
from ..diagnostics import ErrorCode, ValidationError
from .callable import CallableReducerBinding, ReducerCallable

Reducer: TypeAlias = str | ReducerCallable


class ReducerName(StrEnum):
    """Closed vocabulary of backend-native reducer names."""

    __str__ = Enum.__str__

    def __format__(self, format_spec: str) -> str:
        """Format the qualified member name.

        Parameters
        ----------
        format_spec : str
            String format specification applied to the member name.

        Returns
        -------
        str
            Formatted qualified member name.
        """
        return Enum.__format__(self, format_spec)

    SUM = "sum"
    PROD = "prod"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    ALL = "all"
    ANY = "any"


CanonicalReducer: TypeAlias = ReducerName | CallableReducerBinding


@dataclass(frozen=True, slots=True, init=False)
class ReducerPhase:
    """One normalized reduction phase."""

    axes: AxisTerms
    reducer: CanonicalReducer

    def __init__(
        self,
        axes: AxisTerms | tuple[AxisTermBase | int, ...],
        reducer: Reducer | CallableReducerBinding,
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
        elif isinstance(reducer, CallableReducerBinding):
            normalized_reducer = reducer
        elif callable(reducer):
            normalized_reducer = CallableReducerBinding(reducer)
        else:
            raise TypeError("reducer must be a string or callable")

        object.__setattr__(self, "axes", AxisTerms.from_spec(axes))
        object.__setattr__(self, "reducer", normalized_reducer)


ReducerPlan: TypeAlias = tuple[ReducerPhase, ...]
