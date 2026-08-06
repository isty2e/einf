from collections.abc import Callable

from einf.backend import ArrayNamespace, BackendArrayOps
from einf.backend.runtime import is_trusted_backend_array_ops
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError
from einf.shape import compile_fixed_rank_shape_evaluator
from einf.steps.permute import permute_execution_error
from einf.steps.runtime import validate_runtime_output_shape
from einf.tensor_types import TensorLike

from .model import ExpandSymbolicProgram


def expand_execution_error(error: Exception, /) -> ExecutionError:
    """Build the canonical expand backend failure.

    Parameters
    ----------
    error
        Backend exception raised while executing an expand route.

    Returns
    -------
    ExecutionError
        Structured failure for the public operation boundary.
    """
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: expand runtime failed: {error}",
        help="ensure expand mapping is valid for the given tensor shape and backend",
        related=("expand runtime",),
        data={"operation": "expand"},
    )


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
    transformed = tensor
    active_stage = "expand"
    try:
        if backend_ops is not None:
            trusted_route = is_trusted_backend_array_ops(
                backend_ops=backend_ops,
                tensor=tensor,
            )
            expected_shape = tensor.shape
            if compiled.has_non_identity_permutation:
                active_stage = "permute"
                transformed = backend_ops.permute(
                    transformed,
                    compiled.permutation,
                )
                if not trusted_route:
                    expected_shape = tuple(
                        expected_shape[input_index]
                        for input_index in compiled.permutation
                    )
                    transformed = validate_runtime_output_shape(
                        transformed,
                        expected_shape,
                        operation="permute",
                    )
            active_stage = "expand"
            for output_index in compiled.insert_axes:
                transformed = backend_ops.expand_dims(transformed, output_index)
                if not trusted_route:
                    expected_shape = (
                        expected_shape[:output_index]
                        + (1,)
                        + expected_shape[output_index:]
                    )
                    transformed = validate_runtime_output_shape(
                        transformed,
                        expected_shape,
                        operation="expand",
                    )
            output = backend_ops.broadcast_to(transformed, target_shape)
            return validate_runtime_output_shape(
                output,
                target_shape,
                operation="expand",
            )

        if xp is None:
            raise ValueError("array namespace is required for expand runtime")
        expected_shape = tensor.shape
        if compiled.has_non_identity_permutation:
            active_stage = "permute"
            transformed = xp.permute_dims(transformed, compiled.permutation)
            expected_shape = tuple(
                expected_shape[input_index] for input_index in compiled.permutation
            )
            transformed = validate_runtime_output_shape(
                transformed,
                expected_shape,
                operation="permute",
            )
        active_stage = "expand"
        for output_index in compiled.insert_axes:
            transformed = xp.expand_dims(transformed, axis=output_index)
            expected_shape = (
                expected_shape[:output_index] + (1,) + expected_shape[output_index:]
            )
            transformed = validate_runtime_output_shape(
                transformed,
                expected_shape,
                operation="expand",
            )
        output = xp.broadcast_to(transformed, target_shape)
        return validate_runtime_output_shape(
            output,
            target_shape,
            operation="expand",
        )
    except TensorOpError:
        raise
    except Exception as error:
        if active_stage == "permute":
            raise permute_execution_error(error) from error
        raise expand_execution_error(error) from error


__all__ = [
    "compile_expand_target_shape_evaluator",
    "expand_execution_error",
    "run_expand_program",
]
