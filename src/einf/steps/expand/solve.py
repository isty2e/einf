from functools import lru_cache

from einf.axis import (
    AxisInt,
    AxisTerms,
    ScalarAxisTermBase,
    ScalarAxisTerms,
    term_size,
)
from einf.lowering.expand import (
    ExpandSymbolicProgram,
    build_expand_symbolic_program,
)
from einf.signature import Signature
from einf.solver import solve_dimensions


@lru_cache(maxsize=1_024)
def _repeat_solver_signature(lhs_terms: AxisTerms, rhs_terms: AxisTerms) -> Signature:
    """Build cached unary signature for expand shape solving."""
    return Signature(inputs=(lhs_terms,), outputs=(rhs_terms,))


def _bind_axis_terms_with_axis_sizes(
    *,
    axis_terms: AxisTerms,
    axis_sizes: dict[str, int],
) -> AxisTerms | None:
    """Resolve expression terms to literals under solved axis sizes."""
    try:
        scalar_terms = ScalarAxisTerms.from_spec(axis_terms)
    except TypeError:
        return None

    bound_terms: list[ScalarAxisTermBase] = []
    for term in scalar_terms:
        evaluated = term.evaluate(axis_sizes)
        if evaluated is not None:
            if evaluated < 0:
                return None
            bound_terms.append(AxisInt(evaluated))
            continue
        bound_terms.append(term)

    return AxisTerms(tuple(bound_terms))


def solve_expand_program_from_input_shape(
    *,
    input_shape: tuple[int, ...],
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
    explicit_sizes: dict[str, int],
) -> tuple[ExpandSymbolicProgram, tuple[int, ...]] | None:
    """Solve unary expand from one input shape and build executable program."""
    signature = _repeat_solver_signature(lhs_terms, rhs_terms)
    solved = solve_dimensions(
        signature,
        input_shapes=(input_shape,),
        explicit_sizes=explicit_sizes,
    )

    bound_lhs_terms = _bind_axis_terms_with_axis_sizes(
        axis_terms=lhs_terms,
        axis_sizes=solved.axis_sizes,
    )
    if bound_lhs_terms is None:
        return None

    bound_rhs_terms = _bind_axis_terms_with_axis_sizes(
        axis_terms=rhs_terms,
        axis_sizes=solved.axis_sizes,
    )
    if bound_rhs_terms is None:
        return None

    program = build_expand_symbolic_program(bound_lhs_terms, bound_rhs_terms)
    if program.compiled is None:
        return None

    scalar_rhs_terms = ScalarAxisTerms.from_spec(bound_rhs_terms)
    try:
        target_shape = tuple(
            term_size(term, solved.axis_sizes) for term in scalar_rhs_terms
        )
    except KeyError:
        return None
    if any(dim < 0 for dim in target_shape):
        return None

    return program, target_shape


__all__ = [
    "solve_expand_program_from_input_shape",
]
