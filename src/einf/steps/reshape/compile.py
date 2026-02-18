from collections.abc import Callable
from functools import lru_cache

from einf.axis import (
    Axis,
    AxisExpr,
    AxisInt,
    AxisTerms,
    ScalarAxisTermBase,
    ScalarAxisTerms,
)
from einf.shape import (
    compile_fast_shape_eval_fn,
    compile_fixed_rank_shape_evaluator,
    compile_shape_eval_fn,
    compile_shape_node,
    shape_node_axis_names,
)
from einf.signature import Signature

from .constants import (
    ZERO_COPY_ALLOWED_RESHAPE_MODE,
    ZERO_COPY_REQUIRED_RESHAPE_MODE,
    ZeroCopyReshapeMode,
)
from .model import (
    ReshapeCompiledProgram,
    ReshapeSymbolicProgram,
)


@lru_cache(maxsize=2_048)
def build_reshape_compiled_program(
    lhs_terms: AxisTerms, rhs_terms: AxisTerms
) -> ReshapeCompiledProgram | None:
    """Compile one symbolic unary reshape plan independent of concrete extents."""
    return _compile_reshape_compiled_program(
        lhs_terms=lhs_terms,
        rhs_terms=rhs_terms,
    )


def _compile_reshape_compiled_program(
    *,
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
) -> ReshapeCompiledProgram | None:
    """Compile one reshape program independent of concrete extents."""
    try:
        scalar_lhs_terms = ScalarAxisTerms.from_spec(lhs_terms)
        scalar_rhs_terms = ScalarAxisTerms.from_spec(rhs_terms)
    except TypeError:
        return None

    axis_names: set[str] = set()
    for term in scalar_lhs_terms:
        axis_names.update(term.axis_names())
    for term in scalar_rhs_terms:
        axis_names.update(term.axis_names())

    lhs_axis_names: list[str | None] = []
    lhs_literal_dims: list[int | None] = []
    for lhs_term in scalar_lhs_terms:
        if isinstance(lhs_term, Axis):
            lhs_axis_names.append(lhs_term.name)
            lhs_literal_dims.append(None)
            continue
        if isinstance(lhs_term, AxisInt):
            lhs_axis_names.append(None)
            lhs_literal_dims.append(lhs_term.value)
            continue
        if isinstance(lhs_term, AxisExpr):
            lhs_axis_names.append(None)
            lhs_literal_dims.append(None)
            continue
        return None

    normalized_rhs_terms: list[ScalarAxisTermBase] = []
    for rhs_term in scalar_rhs_terms:
        if isinstance(rhs_term, (Axis, AxisExpr, AxisInt)):
            normalized_rhs_terms.append(rhs_term)
            continue
        return None

    axis_index_by_name: dict[str, int] = {}
    lhs_axis_equal_checks: list[tuple[int, int]] = []
    lhs_literal_checks: list[tuple[int, int]] = []
    for lhs_index, axis_name in enumerate(lhs_axis_names):
        literal_dim = lhs_literal_dims[lhs_index]
        if literal_dim is not None:
            lhs_literal_checks.append((lhs_index, literal_dim))
            continue
        if axis_name is None:
            continue
        previous_index = axis_index_by_name.get(axis_name)
        if previous_index is None:
            axis_index_by_name[axis_name] = lhs_index
            continue
        lhs_axis_equal_checks.append((previous_index, lhs_index))

    rhs_shape_eval_fns: list[
        Callable[[tuple[int, ...], dict[str, int]], int | None]
    ] = []
    rhs_fast_shape_eval_fns: list[Callable[[tuple[int, ...], dict[str, int]], int]] = []
    rhs_required_explicit_names: set[str] = set()
    has_fast_shape_eval = True
    for rhs_term in normalized_rhs_terms:
        compiled_node = compile_shape_node(
            term=rhs_term,
            axis_index_by_name=axis_index_by_name,
        )
        if compiled_node is None:
            has_fast_shape_eval = False
            rhs_shape_eval_fns.append(_eval_unresolved_shape_node)
            continue
        rhs_required_explicit_names.update(shape_node_axis_names(compiled_node))
        rhs_shape_eval_fns.append(compile_shape_eval_fn(compiled_node))
        fast_shape_eval_fn = compile_fast_shape_eval_fn(compiled_node)
        if fast_shape_eval_fn is None:
            has_fast_shape_eval = False
        else:
            rhs_fast_shape_eval_fns.append(fast_shape_eval_fn)

    return ReshapeCompiledProgram(
        lhs_terms=tuple(
            ScalarAxisTermBase.coerce(term) for term in scalar_lhs_terms
        ),
        lhs_axis_names=tuple(lhs_axis_names),
        lhs_literal_dims=tuple(lhs_literal_dims),
        lhs_axis_equal_checks=tuple(lhs_axis_equal_checks),
        lhs_literal_checks=tuple(lhs_literal_checks),
        rhs_terms=tuple(normalized_rhs_terms),
        rhs_shape_eval_fns=tuple(rhs_shape_eval_fns),
        rhs_fast_shape_eval_fns=(
            tuple(rhs_fast_shape_eval_fns) if has_fast_shape_eval else None
        ),
        rhs_required_explicit_names=tuple(sorted(rhs_required_explicit_names)),
        axis_names=frozenset(axis_names),
    )


def build_reshape_symbolic_program(
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
    *,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
    reject_not_a_view: bool = False,
) -> ReshapeSymbolicProgram:
    """Build one step-level reshape program from canonical unary sides."""
    if zero_copy_mode not in {
        ZERO_COPY_ALLOWED_RESHAPE_MODE,
        ZERO_COPY_REQUIRED_RESHAPE_MODE,
    }:
        raise ValueError("reshape symbolic program requires a valid zero_copy_mode")
    return ReshapeSymbolicProgram(
        signature=Signature(
            inputs=(lhs_terms,),
            outputs=(rhs_terms,),
        ),
        compiled=build_reshape_compiled_program(lhs_terms, rhs_terms),
        zero_copy_mode=zero_copy_mode,
        reject_not_a_view=reject_not_a_view,
    )


def _eval_unresolved_shape_node(
    input_shape: tuple[int, ...],
    explicit_sizes: dict[str, int],
) -> int | None:
    """Return None for shape terms that require symbolic fallback evaluation."""
    _ = input_shape
    _ = explicit_sizes
    return None


def compile_reshape_target_shape_evaluator(
    *,
    plan: ReshapeCompiledProgram,
    explicit_sizes: dict[str, int],
) -> Callable[[tuple[int, ...]], tuple[int, ...] | None]:
    """Compile one runtime target-shape evaluator with fixed explicit sizes."""
    if any(name not in plan.axis_names for name in explicit_sizes):
        return lambda input_shape: None

    lhs_rank = len(plan.lhs_axis_names)
    lhs_literal_checks = plan.lhs_literal_checks
    lhs_axis_equal_checks = plan.lhs_axis_equal_checks
    fast_shape_eval_fns = plan.rhs_fast_shape_eval_fns
    shape_eval_fns = plan.rhs_shape_eval_fns
    required_explicit_names = plan.rhs_required_explicit_names
    has_required_explicit_names = all(
        name in explicit_sizes for name in required_explicit_names
    )

    if (
        has_required_explicit_names
        and not lhs_literal_checks
        and not lhs_axis_equal_checks
        and fast_shape_eval_fns is not None
    ):
        fixed_rank_evaluator = compile_fixed_rank_shape_evaluator(
            fast_shape_eval_fns=fast_shape_eval_fns,
            lhs_rank=lhs_rank,
            explicit_sizes=explicit_sizes,
        )
        if fixed_rank_evaluator is not None:
            return fixed_rank_evaluator

    def evaluate_target_shape(input_shape: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(input_shape) != lhs_rank:
            return None

        for index, literal_dim in lhs_literal_checks:
            if index >= len(input_shape):
                return None
            if input_shape[index] != literal_dim:
                return None
        for left_index, right_index in lhs_axis_equal_checks:
            if left_index >= len(input_shape) or right_index >= len(input_shape):
                return None
            if input_shape[left_index] != input_shape[right_index]:
                return None

        if fast_shape_eval_fns is not None:
            try:
                return tuple(
                    shape_eval(input_shape, explicit_sizes)
                    for shape_eval in fast_shape_eval_fns
                )
            except KeyError:
                return None

        target_shape: list[int] = []
        for shape_eval in shape_eval_fns:
            value = shape_eval(input_shape, explicit_sizes)
            if value is None:
                return None
            target_shape.append(value)
        return tuple(target_shape)

    return evaluate_target_shape


__all__ = [
    "build_reshape_compiled_program",
    "build_reshape_symbolic_program",
    "compile_reshape_target_shape_evaluator",
]
