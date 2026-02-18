from .diagnostics import ErrorCode, ExecutionError
from .tensor_types import TensorLike

RuntimeOutputs = TensorLike | tuple[TensorLike, ...] | list[TensorLike]


def normalize_outputs(
    *,
    op_name: str,
    expected_output_arity: int,
    raw_outputs: RuntimeOutputs,
) -> TensorLike | tuple[TensorLike, ...]:
    """Normalize backend outputs to TensorOp output protocol."""
    if expected_output_arity == 1:
        if isinstance(raw_outputs, tuple):
            if len(raw_outputs) != 1:
                raise _build_output_protocol_error(
                    op_name=op_name,
                    expected_output_arity=1,
                    observed_output_arity=len(raw_outputs),
                )
            _validate_output_tensor(
                op_name=op_name, output=raw_outputs[0], output_index=0
            )
            return raw_outputs[0]

        if isinstance(raw_outputs, list):
            if len(raw_outputs) != 1:
                raise _build_output_protocol_error(
                    op_name=op_name,
                    expected_output_arity=1,
                    observed_output_arity=len(raw_outputs),
                )
            _validate_output_tensor(
                op_name=op_name, output=raw_outputs[0], output_index=0
            )
            return raw_outputs[0]

        _validate_output_tensor(op_name=op_name, output=raw_outputs, output_index=0)
        return raw_outputs

    if isinstance(raw_outputs, tuple):
        normalized = raw_outputs
    elif isinstance(raw_outputs, list):
        normalized = tuple(raw_outputs)
    else:
        raise _build_output_protocol_error(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            observed_output_arity=1,
        )

    if len(normalized) != expected_output_arity:
        raise _build_output_protocol_error(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            observed_output_arity=len(normalized),
        )

    for output_index, output in enumerate(normalized):
        _validate_output_tensor(
            op_name=op_name, output=output, output_index=output_index
        )

    return normalized


def normalize_runtime_outputs(
    *,
    op_name: str,
    expected_output_arity: int,
    raw_outputs: tuple[TensorLike, ...],
) -> TensorLike | tuple[TensorLike, ...]:
    """Normalize runtime tuple outputs to TensorOp output protocol."""
    if expected_output_arity == 1:
        if len(raw_outputs) != 1:
            raise _build_output_protocol_error(
                op_name=op_name,
                expected_output_arity=1,
                observed_output_arity=len(raw_outputs),
            )
        return raw_outputs[0]

    if len(raw_outputs) != expected_output_arity:
        raise _build_output_protocol_error(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            observed_output_arity=len(raw_outputs),
        )
    return raw_outputs


def _validate_output_tensor(
    *,
    op_name: str,
    output: TensorLike,
    output_index: int,
) -> None:
    """Validate one backend output for TensorLike shape contract."""
    try:
        shape = output.shape
    except Exception as exc:
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                f"{op_name} output protocol violation: output[{output_index}] "
                "must expose a shape attribute"
            ),
            help="return TensorLike outputs with tuple[int, ...] shape",
            related=("TensorOp output protocol",),
            data={"operation": op_name, "index": output_index},
        ) from exc

    if not isinstance(shape, tuple):
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                f"{op_name} output protocol violation: output[{output_index}] "
                "shape must be tuple[int, ...]"
            ),
            help="return TensorLike outputs with tuple[int, ...] shape",
            related=("TensorOp output protocol",),
            data={"operation": op_name, "index": output_index},
        )

    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ExecutionError(
                code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
                message=(
                    f"{op_name} output protocol violation: output[{output_index}] "
                    "shape entries must be ints"
                ),
                help="return TensorLike outputs with integer shape entries",
                related=("TensorOp output protocol",),
                data={"operation": op_name, "index": output_index},
            )


def _build_output_protocol_error(
    *,
    op_name: str,
    expected_output_arity: int,
    observed_output_arity: int,
) -> ExecutionError:
    """Build output-arity protocol error for exact-arity execution."""
    return ExecutionError(
        code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
        message=(
            f"{op_name} output protocol violation: expected {expected_output_arity} "
            f"outputs, got {observed_output_arity}"
        ),
        help="return outputs matching signature output arity in declared order",
        related=("TensorOp output protocol",),
        data={
            "operation": op_name,
            "expected": expected_output_arity,
            "got": observed_output_arity,
        },
    )


__all__ = ["normalize_outputs", "normalize_runtime_outputs"]
