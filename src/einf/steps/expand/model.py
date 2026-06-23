from collections.abc import Callable
from dataclasses import dataclass

from einf.axis import AxisTerms
from einf.steps.base import SymbolicProgram


@dataclass(frozen=True, slots=True)
class ExpandCompiledProgram:
    """Compiled unary-expand program independent of concrete extents."""

    lhs_axis_names: tuple[str | None, ...]
    lhs_literal_dims: tuple[int | None, ...]
    lhs_axis_equal_checks: tuple[tuple[int, int], ...]
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
    output_to_input: tuple[int | None, ...]
    insert_axes: tuple[int, ...]
    permutation: tuple[int, ...]
    has_non_identity_permutation: bool
    axis_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class ExpandSymbolicProgram(SymbolicProgram):
    """Lowered unary-expand program consumed by expand symbolic/runtime steps."""

    lhs_terms: AxisTerms
    rhs_terms: AxisTerms
    compiled: ExpandCompiledProgram | None


__all__ = [
    "ExpandCompiledProgram",
    "ExpandSymbolicProgram",
]
