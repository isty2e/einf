from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from einf.axis import (
    Axis,
    AxisInt,
    AxisTerms,
    ScalarAxisTerms,
)
from einf.shape import (
    compile_fast_shape_eval_fn,
    compile_shape_eval_fn,
    compile_shape_node,
)
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


@lru_cache(maxsize=2_048)
def build_expand_symbolic_program(
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
) -> ExpandSymbolicProgram:
    """Build one expand program from canonical unary axis terms."""
    compiled = _compile_expand_program(
        lhs_terms=lhs_terms,
        rhs_terms=rhs_terms,
    )
    return ExpandSymbolicProgram(
        lhs_terms=lhs_terms,
        rhs_terms=rhs_terms,
        compiled=compiled,
    )


def _compile_expand_program(
    *,
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
) -> ExpandCompiledProgram | None:
    try:
        scalar_lhs_terms = ScalarAxisTerms.from_spec(lhs_terms)
        scalar_rhs_terms = ScalarAxisTerms.from_spec(rhs_terms)
    except TypeError:
        return None

    if not scalar_lhs_terms.is_shape_bindable():
        return None

    output_to_input_value = _build_output_to_input(
        lhs_terms=scalar_lhs_terms,
        rhs_terms=scalar_rhs_terms,
    )
    if output_to_input_value is None:
        return None

    permutation = tuple(
        input_index for input_index in output_to_input_value if input_index is not None
    )
    identity_permutation = tuple(range(len(permutation)))
    insert_axes = tuple(
        output_index
        for output_index, mapped_input in enumerate(output_to_input_value)
        if mapped_input is None
    )

    lhs_axis_names: list[str | None] = []
    lhs_literal_dims: list[int | None] = []
    axis_index_by_name: dict[str, int] = {}
    lhs_axis_equal_checks: list[tuple[int, int]] = []
    for lhs_index, lhs_term in enumerate(scalar_lhs_terms):
        if isinstance(lhs_term, Axis):
            lhs_axis_names.append(lhs_term.name)
            lhs_literal_dims.append(None)
            known_lhs_index = axis_index_by_name.get(lhs_term.name)
            if known_lhs_index is None:
                axis_index_by_name[lhs_term.name] = lhs_index
            else:
                lhs_axis_equal_checks.append((known_lhs_index, lhs_index))
            continue
        if isinstance(lhs_term, AxisInt):
            lhs_axis_names.append(None)
            lhs_literal_dims.append(lhs_term.value)
            continue
        return None

    rhs_shape_eval_fns: list[
        Callable[[tuple[int, ...], dict[str, int]], int | None]
    ] = []
    rhs_fast_shape_eval_fns: list[Callable[[tuple[int, ...], dict[str, int]], int]] = []
    has_fast_shape_eval = True
    for rhs_term in scalar_rhs_terms:
        shape_node = compile_shape_node(
            term=rhs_term,
            axis_index_by_name=axis_index_by_name,
        )
        if shape_node is None:
            return None
        rhs_shape_eval_fns.append(compile_shape_eval_fn(shape_node))
        fast_shape_eval_fn = compile_fast_shape_eval_fn(shape_node)
        if fast_shape_eval_fn is None:
            has_fast_shape_eval = False
        else:
            rhs_fast_shape_eval_fns.append(fast_shape_eval_fn)

    axis_names: set[str] = set()
    for lhs_term in scalar_lhs_terms:
        axis_names.update(lhs_term.axis_names())
    for rhs_term in scalar_rhs_terms:
        axis_names.update(rhs_term.axis_names())

    return ExpandCompiledProgram(
        lhs_axis_names=tuple(lhs_axis_names),
        lhs_literal_dims=tuple(lhs_literal_dims),
        lhs_axis_equal_checks=tuple(lhs_axis_equal_checks),
        rhs_shape_eval_fns=tuple(rhs_shape_eval_fns),
        rhs_fast_shape_eval_fns=(
            tuple(rhs_fast_shape_eval_fns) if has_fast_shape_eval else None
        ),
        output_to_input=output_to_input_value,
        insert_axes=insert_axes,
        permutation=permutation,
        has_non_identity_permutation=permutation != identity_permutation,
        axis_names=frozenset(axis_names),
    )


def _build_output_to_input(
    *,
    lhs_terms: ScalarAxisTerms,
    rhs_terms: ScalarAxisTerms,
) -> tuple[int | None, ...] | None:
    """Build rhs-position to lhs-axis mapping for unary expand lowering."""
    lhs_tokens = tuple(lhs_term.stable_token() for lhs_term in lhs_terms)
    remaining_lhs = list(range(len(lhs_tokens)))
    output_to_input: list[int | None] = []

    for rhs_term in rhs_terms:
        mapped_input: int | None = None
        rhs_token = rhs_term.stable_token()
        for remaining_index, lhs_index in enumerate(remaining_lhs):
            if lhs_tokens[lhs_index] != rhs_token:
                continue
            mapped_input = lhs_index
            del remaining_lhs[remaining_index]
            break
        output_to_input.append(mapped_input)

    if remaining_lhs:
        return None
    return tuple(output_to_input)


__all__ = [
    "ExpandCompiledProgram",
    "ExpandSymbolicProgram",
    "build_expand_symbolic_program",
]
