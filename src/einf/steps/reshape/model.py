from collections.abc import Callable
from dataclasses import dataclass

from einf.axis import ScalarAxisTermBase
from einf.signature import Signature
from einf.steps.base import SymbolicProgram

from .constants import ZERO_COPY_ALLOWED_RESHAPE_MODE, ZeroCopyReshapeMode


@dataclass(frozen=True, slots=True)
class ReshapeCompiledProgram:
    """Prevalidated unary reshape symbolic plan used by reshape runtime."""

    lhs_terms: tuple[ScalarAxisTermBase, ...]
    lhs_axis_names: tuple[str | None, ...]
    lhs_literal_dims: tuple[int | None, ...]
    lhs_axis_equal_checks: tuple[tuple[int, int], ...]
    lhs_literal_checks: tuple[tuple[int, int], ...]
    rhs_terms: tuple[ScalarAxisTermBase, ...]
    rhs_shape_eval_fns: tuple[
        Callable[[tuple[int, ...], dict[str, int]], int | None],
        ...,
    ]
    rhs_fast_shape_eval_fns: (
        tuple[
            Callable[[tuple[int, ...], dict[str, int]], int],
            ...,
        ]
        | None
    )
    rhs_required_explicit_names: tuple[str, ...]
    axis_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReshapeSymbolicProgram(SymbolicProgram):
    """Step-level reshape program that always exists for symbolic/runtime flow."""

    signature: Signature
    compiled: ReshapeCompiledProgram | None
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE
    reject_not_a_view: bool = False


__all__ = [
    "ReshapeCompiledProgram",
    "ReshapeSymbolicProgram",
]
