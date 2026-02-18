from functools import lru_cache

from einf.diagnostics import ErrorCode, ValidationError
from einf.operations.policy import OpPolicy
from einf.output_normalization import normalize_runtime_outputs
from einf.steps.base import RuntimeSpecializationContext
from einf.tensor_types import TensorLike

from .abstract import AbstractPlan


def execute_tensor_op_call(
    op_name: str,
    expected_input_arity: int,
    expected_output_arity: int,
    op_policy: OpPolicy,
    abstract_plan: AbstractPlan,
    tensors: tuple[TensorLike, ...],
) -> TensorLike | tuple[TensorLike, ...]:
    """Execute one TensorOp call through canonical validation and plan dispatch."""
    input_arity = len(tensors)
    op_policy.validate_call(
        op_name=op_name,
        expected_input_arity=expected_input_arity,
        input_arity=input_arity,
    )
    if abstract_plan.requires_input_shapes(input_arity):
        input_shapes = extract_input_shapes(op_name=op_name, tensors=tensors)
    else:
        input_shapes = tuple(() for _ in range(input_arity))
    context = RuntimeSpecializationContext(
        input_shapes=input_shapes,
        backend_profile=None,
    )
    if expected_output_arity == 1:
        try:
            return abstract_plan.execute_single_output(context, tensors)
        except ValidationError:
            raise
        except TypeError as error:
            if not _is_shape_contract_type_error(error):
                raise
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: {error}",
                help="ensure lowering resolves to one feasible symbolic plan",
                related=("TensorOp lowering",),
                data={"operation": op_name},
            ) from error
        except ValueError as error:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: {error}",
                help="ensure lowering resolves to one feasible symbolic plan",
                related=("TensorOp lowering",),
                data={"operation": op_name},
            ) from error

    raw_outputs = _execute_abstract_plan(
        op_name,
        abstract_plan,
        context,
        tensors,
    )
    if len(raw_outputs) != expected_output_arity:
        return normalize_runtime_outputs(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            raw_outputs=raw_outputs,
        )
    return raw_outputs


def extract_input_shapes(
    *,
    op_name: str,
    tensors: tuple[TensorLike, ...],
) -> tuple[tuple[int, ...], ...]:
    """Extract exact-arity input shapes with TensorLike contract checks."""
    input_shapes: list[tuple[int, ...]] = []
    for input_index, tensor in enumerate(tensors):
        input_shapes.append(
            _extract_input_shape(
                op_name=op_name,
                tensor=tensor,
                index=input_index,
            )
        )
    return tuple(input_shapes)


def _extract_input_shape(
    op_name: str,
    tensor: TensorLike,
    index: int,
) -> tuple[int, ...]:
    """Extract one input shape with TensorLike contract checks."""
    try:
        shape = tensor.shape
    except Exception as error:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                f"inconsistent dims: {op_name} input[{index}] "
                "does not expose a readable shape attribute"
            ),
            help="provide TensorLike inputs with tuple[int, ...] shapes",
            related=("TensorOp input shape contract",),
            data={"operation": op_name, "index": index},
        ) from error

    if not isinstance(shape, tuple):
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                f"inconsistent dims: {op_name} input[{index}] "
                "shape must be tuple[int, ...]"
            ),
            help="provide TensorLike inputs with tuple[int, ...] shapes",
            related=("TensorOp input shape contract",),
            data={"operation": op_name, "index": index},
        )

    if _is_trusted_tensor_type(type(tensor)):
        return shape

    for dim in shape:
        if type(dim) is not int:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    f"inconsistent dims: {op_name} input[{index}] "
                    "shape entries must be ints"
                ),
                help="provide TensorLike inputs with integer shape entries",
                related=("TensorOp input shape contract",),
                data={"operation": op_name, "index": index},
            )
    return shape


@lru_cache(maxsize=64)
def _is_trusted_tensor_type(tensor_type: type[object]) -> bool:
    """Return whether tensor type can skip per-dimension int entry checks."""
    tensor_module = tensor_type.__module__
    return tensor_module.startswith("torch") or tensor_module.startswith("numpy")


def _execute_abstract_plan(
    op_name: str,
    abstract_plan: AbstractPlan,
    context: RuntimeSpecializationContext,
    tensors: tuple[TensorLike, ...],
) -> tuple[TensorLike, ...]:
    """Execute one abstract plan and normalize non-diagnostic failures."""
    try:
        return abstract_plan.execute(context, tensors)
    except ValidationError:
        raise
    except TypeError as error:
        if not _is_shape_contract_type_error(error):
            raise
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=f"inconsistent dims: {error}",
            help="ensure lowering resolves to one feasible symbolic plan",
            related=("TensorOp lowering",),
            data={"operation": op_name},
        ) from error


def _is_shape_contract_type_error(error: TypeError, /) -> bool:
    """Return whether TypeError came from input-shape contract validation."""
    message = str(error)
    if message == "shape entries must be integers":
        return True
    if message == "each input shape must be a tuple[int, ...]":
        return True
    return False


__all__ = [
    "execute_tensor_op_call",
    "extract_input_shapes",
]
