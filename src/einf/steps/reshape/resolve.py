from einf.axis import AxisExpr, ScalarAxisTermBase, term_size
from einf.tensor_types import TensorLike

from .model import ReshapeCompiledProgram


def _bind_symbolic_axis_sizes(
    *,
    tensor: TensorLike,
    lhs_terms: tuple[ScalarAxisTermBase, ...],
    lhs_axis_names: tuple[str | None, ...],
    lhs_literal_dims: tuple[int | None, ...],
    axis_names: frozenset[str],
    explicit_sizes: dict[str, int],
) -> dict[str, int] | None:
    """Bind scalar-axis sizes from one unary lhs and one concrete tensor shape."""
    if len(tensor.shape) != len(lhs_axis_names):
        return None

    for explicit_name in explicit_sizes:
        if explicit_name not in axis_names:
            return None

    axis_sizes = dict(explicit_sizes)
    unresolved_expr_terms: list[tuple[ScalarAxisTermBase, int]] = []
    for axis_index, axis_name in enumerate(lhs_axis_names):
        dim = tensor.shape[axis_index]
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None

        lhs_term = lhs_terms[axis_index]
        if isinstance(lhs_term, AxisExpr):
            unresolved_expr_terms.append((lhs_term, dim))
            continue

        literal_dim = lhs_literal_dims[axis_index]
        if literal_dim is not None:
            if literal_dim != dim:
                return None
            continue

        if axis_name is None:
            return None

        bound = axis_sizes.get(axis_name)
        if bound is None:
            axis_sizes[axis_name] = dim
            continue
        if bound != dim:
            return None

    if not unresolved_expr_terms:
        return axis_sizes

    pending = unresolved_expr_terms
    while pending:
        next_pending: list[tuple[ScalarAxisTermBase, int]] = []
        resolved = False
        for term, dim in pending:
            expected_dim = _evaluate_symbolic_term(term=term, axis_sizes=axis_sizes)
            if expected_dim is None:
                next_pending.append((term, dim))
                continue
            if expected_dim != dim:
                return None
            resolved = True
        if not next_pending:
            return axis_sizes
        if not resolved:
            return None
        pending = next_pending

    return axis_sizes


def _evaluate_symbolic_term(
    *, term: ScalarAxisTermBase, axis_sizes: dict[str, int]
) -> int | None:
    """Evaluate one scalar symbolic axis term with bound axis sizes."""
    try:
        return term_size(term, axis_sizes)
    except Exception:
        return None


def _evaluate_target_shape_from_input_shape(
    *,
    program: ReshapeCompiledProgram,
    tensor: TensorLike,
    explicit_sizes: dict[str, int],
) -> tuple[int, ...] | None:
    """Evaluate rhs shape directly from input shape and compiled rhs nodes."""
    input_shape = tensor.shape
    if len(input_shape) != len(program.lhs_axis_names):
        return None

    for index, literal_dim in program.lhs_literal_checks:
        if index >= len(input_shape):
            return None
        if input_shape[index] != literal_dim:
            return None
    for left_index, right_index in program.lhs_axis_equal_checks:
        if left_index >= len(input_shape) or right_index >= len(input_shape):
            return None
        if input_shape[left_index] != input_shape[right_index]:
            return None

    if any(name not in program.axis_names for name in explicit_sizes):
        return None

    fast_shape_eval_fns = program.rhs_fast_shape_eval_fns
    if fast_shape_eval_fns is not None:
        try:
            return tuple(
                shape_eval(input_shape, explicit_sizes)
                for shape_eval in fast_shape_eval_fns
            )
        except KeyError:
            return None

    target_shape: list[int] = []
    for shape_eval in program.rhs_shape_eval_fns:
        value = shape_eval(input_shape, explicit_sizes)
        if value is None:
            return None
        target_shape.append(value)
    return tuple(target_shape)


def _evaluate_symbolic_target_shape(
    *, rhs_terms: tuple[ScalarAxisTermBase, ...], axis_sizes: dict[str, int]
) -> tuple[int, ...] | None:
    """Evaluate rhs target shape from symbolic rhs terms and bound sizes."""
    target_shape: list[int] = []
    for rhs_term in rhs_terms:
        size = _evaluate_symbolic_term(term=rhs_term, axis_sizes=axis_sizes)
        if size is None:
            return None
        target_shape.append(size)
    return tuple(target_shape)


def resolve_reshape_target_shape(
    *,
    tensor: TensorLike,
    explicit_sizes: dict[str, int],
    program: ReshapeCompiledProgram,
) -> tuple[int, ...] | None:
    """Resolve target shape from input shape and explicit sizes."""
    target_shape = _evaluate_target_shape_from_input_shape(
        program=program,
        tensor=tensor,
        explicit_sizes=explicit_sizes,
    )
    if target_shape is not None:
        return target_shape

    axis_sizes = _bind_symbolic_axis_sizes(
        tensor=tensor,
        lhs_terms=program.lhs_terms,
        lhs_axis_names=program.lhs_axis_names,
        lhs_literal_dims=program.lhs_literal_dims,
        axis_names=program.axis_names,
        explicit_sizes=explicit_sizes,
    )
    if axis_sizes is None:
        return None
    return _evaluate_symbolic_target_shape(
        rhs_terms=program.rhs_terms,
        axis_sizes=axis_sizes,
    )


__all__ = ["resolve_reshape_target_shape"]
