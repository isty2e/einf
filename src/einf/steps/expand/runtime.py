from collections.abc import Callable

from einf.backend import ArrayNamespace, BackendArrayOps
from einf.lowering.expand import ExpandSymbolicProgram
from einf.shape import compile_fixed_rank_shape_evaluator
from einf.tensor_types import TensorLike


def compile_expand_target_shape_evaluator(
    *,
    plan: ExpandSymbolicProgram,
    explicit_sizes: dict[str, int],
) -> Callable[[tuple[int, ...]], tuple[int, ...] | None]:
    """Compile one runtime target-shape evaluator with fixed explicit sizes."""
    compiled = plan.compiled
    if compiled is None:
        return lambda input_shape: None

    lhs_axis_names = compiled.lhs_axis_names
    lhs_literal_dims = compiled.lhs_literal_dims
    lhs_axis_equal_checks = compiled.lhs_axis_equal_checks
    rhs_shape_eval_fns = compiled.rhs_shape_eval_fns
    rhs_fast_shape_eval_fns = compiled.rhs_fast_shape_eval_fns
    axis_names = compiled.axis_names
    lhs_bound_axis_names = frozenset(
        axis_name for axis_name in lhs_axis_names if axis_name is not None
    )
    rhs_required_explicit_names = tuple(sorted(axis_names - lhs_bound_axis_names))

    if any(name not in axis_names for name in explicit_sizes):
        return lambda input_shape: None

    lhs_rank = len(lhs_axis_names)
    lhs_has_literal_checks = any(literal is not None for literal in lhs_literal_dims)
    lhs_has_explicit_constraints = any(
        axis_name is not None and axis_name in explicit_sizes
        for axis_name in lhs_axis_names
    )
    has_required_explicit_names = all(
        axis_name in explicit_sizes for axis_name in rhs_required_explicit_names
    )

    if (
        has_required_explicit_names
        and not lhs_has_literal_checks
        and not lhs_axis_equal_checks
        and not lhs_has_explicit_constraints
        and rhs_fast_shape_eval_fns is not None
    ):
        fixed_rank_evaluator = compile_fixed_rank_shape_evaluator(
            fast_shape_eval_fns=rhs_fast_shape_eval_fns,
            lhs_rank=lhs_rank,
            explicit_sizes=explicit_sizes,
        )
        if fixed_rank_evaluator is not None:
            return fixed_rank_evaluator

    def evaluate_target_shape(input_shape: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(input_shape) != lhs_rank:
            return None

        for lhs_index, axis_name in enumerate(lhs_axis_names):
            dim = input_shape[lhs_index]
            if isinstance(dim, bool) or not isinstance(dim, int):
                return None

            literal_dim = lhs_literal_dims[lhs_index]
            if literal_dim is not None:
                if literal_dim != dim:
                    return None
                continue

            if axis_name is None:
                return None

            explicit_bound = explicit_sizes.get(axis_name)
            if explicit_bound is None:
                continue
            if explicit_bound != dim:
                return None

        for first_index, second_index in lhs_axis_equal_checks:
            if input_shape[first_index] != input_shape[second_index]:
                return None

        if rhs_fast_shape_eval_fns is not None:
            try:
                return tuple(
                    shape_eval_fn(input_shape, explicit_sizes)
                    for shape_eval_fn in rhs_fast_shape_eval_fns
                )
            except KeyError:
                return None

        target_shape: list[int] = []
        for shape_eval_fn in rhs_shape_eval_fns:
            dim = shape_eval_fn(input_shape, explicit_sizes)
            if dim is None:
                return None
            if dim < 0:
                return None
            target_shape.append(dim)
        return tuple(target_shape)

    return evaluate_target_shape


def run_expand_program(
    *,
    plan: ExpandSymbolicProgram,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
    backend_ops: BackendArrayOps | None,
    xp: ArrayNamespace | None,
) -> TensorLike:
    """Run one symbolic unary-expand program via backend-native primitives."""
    compiled = plan.compiled
    if compiled is None:
        raise ValueError("expand program must be compiled before runtime execution")
    if backend_ops is not None:
        transformed = tensor
        if compiled.has_non_identity_permutation:
            transformed = backend_ops.permute(transformed, compiled.permutation)
        for output_index in compiled.insert_axes:
            transformed = backend_ops.expand_dims(transformed, output_index)
        return backend_ops.broadcast_to(transformed, target_shape)

    if xp is None:
        raise ValueError("array namespace is required for expand runtime")
    transformed = tensor
    if compiled.has_non_identity_permutation:
        transformed = xp.permute_dims(transformed, compiled.permutation)
    for output_index in compiled.insert_axes:
        transformed = xp.expand_dims(transformed, axis=output_index)
    return xp.broadcast_to(transformed, target_shape)


__all__ = [
    "compile_expand_target_shape_evaluator",
    "run_expand_program",
]
