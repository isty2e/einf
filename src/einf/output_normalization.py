from dataclasses import dataclass, field
from threading import RLock

from .diagnostics import ErrorCode, ExecutionError, TensorOpError
from .tensor_types import TensorLike, is_trusted_tensor_type

RuntimeOutputs = TensorLike | tuple[TensorLike, ...] | list[TensorLike]
_OutputTypeSignature = tuple[type[object], ...]
_TRUSTED_OUTPUT_SIGNATURE_CACHE_SIZE = 16


@dataclass(frozen=True, slots=True, eq=False)
class RuntimeOutputContract:
    """Normalize and validate runtime outputs at one public operation boundary."""

    op_name: str
    expected_output_arity: int
    _trusted_type_signatures: list[_OutputTypeSignature] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _last_trusted_type_signature: _OutputTypeSignature | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def normalize(
        self,
        raw_outputs: RuntimeOutputs,
        /,
    ) -> TensorLike | tuple[TensorLike, ...]:
        """Return canonical outputs after enforcing the runtime output contract."""
        if self.expected_output_arity == 1 and not isinstance(
            raw_outputs, (tuple, list)
        ):
            last_trusted = self._last_trusted_type_signature
            if (
                last_trusted is not None
                and len(last_trusted) == 1
                and type(raw_outputs) is last_trusted[0]
            ):
                return raw_outputs

        outputs = _normalize_output_tuple(
            op_name=self.op_name,
            expected_output_arity=self.expected_output_arity,
            raw_outputs=raw_outputs,
        )
        type_signature = tuple(type(output) for output in outputs)
        if not _same_type_signature(
            type_signature,
            self._last_trusted_type_signature,
        ):
            trusted_signature = all(
                is_trusted_tensor_type(output_type) for output_type in type_signature
            )
            if not trusted_signature or not self._has_trusted(type_signature):
                _validate_output_tensors(op_name=self.op_name, outputs=outputs)
                if trusted_signature:
                    self._remember_trusted(type_signature)
        return _project_outputs(
            expected_output_arity=self.expected_output_arity,
            outputs=outputs,
        )

    def _has_trusted(self, type_signature: _OutputTypeSignature, /) -> bool:
        """Return whether one trusted output signature was already validated."""
        with self._lock:
            if not any(
                _same_type_signature(type_signature, cached_signature)
                for cached_signature in self._trusted_type_signatures
            ):
                return False
            object.__setattr__(
                self,
                "_last_trusted_type_signature",
                type_signature,
            )
            return True

    def _remember_trusted(self, type_signature: _OutputTypeSignature, /) -> None:
        """Record one successfully validated trusted output signature."""
        with self._lock:
            if (
                len(self._trusted_type_signatures)
                >= _TRUSTED_OUTPUT_SIGNATURE_CACHE_SIZE
            ):
                self._trusted_type_signatures.clear()
            self._trusted_type_signatures.append(type_signature)
            object.__setattr__(
                self,
                "_last_trusted_type_signature",
                type_signature,
            )


def _same_type_signature(
    first: _OutputTypeSignature,
    second: _OutputTypeSignature | None,
    /,
) -> bool:
    """Compare output type signatures without invoking metaclass equality."""
    if second is None or len(first) != len(second):
        return False
    return all(
        first_type is second_type
        for first_type, second_type in zip(first, second, strict=True)
    )


def normalize_outputs(
    *,
    op_name: str,
    expected_output_arity: int,
    raw_outputs: RuntimeOutputs,
) -> TensorLike | tuple[TensorLike, ...]:
    """Normalize backend outputs to TensorOp output protocol."""
    outputs = _normalize_output_tuple(
        op_name=op_name,
        expected_output_arity=expected_output_arity,
        raw_outputs=raw_outputs,
    )
    _validate_output_tensors(op_name=op_name, outputs=outputs)
    return _project_outputs(
        expected_output_arity=expected_output_arity,
        outputs=outputs,
    )


def _normalize_output_tuple(
    *,
    op_name: str,
    expected_output_arity: int,
    raw_outputs: RuntimeOutputs,
) -> tuple[TensorLike, ...]:
    """Normalize one raw output value to an exact-arity tensor tuple."""
    if isinstance(raw_outputs, tuple):
        outputs = raw_outputs
    elif isinstance(raw_outputs, list):
        outputs = tuple(raw_outputs)
    elif expected_output_arity == 1:
        outputs = (raw_outputs,)
    else:
        raise _build_output_protocol_error(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            observed_output_arity=1,
        )

    if len(outputs) != expected_output_arity:
        raise _build_output_protocol_error(
            op_name=op_name,
            expected_output_arity=expected_output_arity,
            observed_output_arity=len(outputs),
        )
    return outputs


def _project_outputs(
    *,
    expected_output_arity: int,
    outputs: tuple[TensorLike, ...],
) -> TensorLike | tuple[TensorLike, ...]:
    """Project one canonical output tuple to the public return convention."""
    if expected_output_arity == 1:
        return outputs[0]
    return outputs


def normalize_runtime_outputs(
    *,
    op_name: str,
    expected_output_arity: int,
    raw_outputs: tuple[TensorLike, ...],
) -> TensorLike | tuple[TensorLike, ...]:
    """Normalize runtime tuple outputs to TensorOp output protocol."""
    return normalize_outputs(
        op_name=op_name,
        expected_output_arity=expected_output_arity,
        raw_outputs=raw_outputs,
    )


def _validate_output_tensors(
    *,
    op_name: str,
    outputs: tuple[TensorLike, ...],
) -> None:
    """Validate each tensor in one canonical runtime output tuple."""
    for output_index, output in enumerate(outputs):
        validated_output_shape(
            op_name=op_name,
            output=output,
            output_index=output_index,
        )


def validated_output_shape(
    *,
    op_name: str,
    output: TensorLike,
    output_index: int,
) -> tuple[int, ...]:
    """Return one backend output's canonical TensorLike shape.

    Parameters
    ----------
    op_name
        Operation that produced the output.
    output
        Backend value whose TensorLike shape contract is validated.
    output_index
        Position of the output in the operation result.

    Returns
    -------
    tuple of int
        Validated output shape.

    Raises
    ------
    ExecutionError
        The output does not satisfy the TensorLike shape contract.
    """
    try:
        shape = output.shape
    except TensorOpError:
        raise
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

    try:
        canonical_shape = shape if type(shape) is tuple else tuple(shape)
    except TensorOpError:
        raise
    except Exception as exc:
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                f"{op_name} output protocol violation: output[{output_index}] "
                "shape must be readable as tuple[int, ...]"
            ),
            help="return TensorLike outputs with readable tuple[int, ...] shape",
            related=("TensorOp output protocol",),
            data={"operation": op_name, "index": output_index},
        ) from exc

    for dim in canonical_shape:
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

    try:
        getitem = getattr(output, "__getitem__", None)
    except TensorOpError:
        raise
    except Exception as exc:
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                f"{op_name} output protocol violation: output[{output_index}] "
                "must expose callable __getitem__"
            ),
            help="return TensorLike outputs that support tensor indexing",
            related=("TensorOp output protocol",),
            data={"operation": op_name, "index": output_index},
        ) from exc
    if not callable(getitem):
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                f"{op_name} output protocol violation: output[{output_index}] "
                "must expose callable __getitem__"
            ),
            help="return TensorLike outputs that support tensor indexing",
            related=("TensorOp output protocol",),
            data={"operation": op_name, "index": output_index},
        )
    return canonical_shape


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


__all__ = [
    "RuntimeOutputContract",
    "normalize_outputs",
    "normalize_runtime_outputs",
]
