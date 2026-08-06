from dataclasses import dataclass

from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    get_backend_array_ops,
)
from einf.backend.runtime import ArrayNamespaceBinding
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError
from einf.output_normalization import validated_output_shape
from einf.steps.base import RuntimeSpecializationContext
from einf.tensor_types import TensorLike, is_trusted_tensor_type

FALLBACK_ELIGIBLE_BACKEND_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)
"""Backend exceptions that permit a local semantically equivalent fallback.

This tuple is fallback policy, not proof that a backend route is unavailable.
Final routes must project failures through the structured execution channel.
"""


@dataclass(frozen=True, slots=True)
class RuntimeBackendBinding:
    """Internal backend binding resolved during step specialization."""

    backend_ops: BackendArrayOps | None
    xp: ArrayNamespace | None


def coerce_step_outputs(
    raw_outputs: TensorLike | tuple[TensorLike, ...] | list[TensorLike],
    /,
) -> tuple[TensorLike, ...]:
    """Normalize runtime outputs to runtime-step tuple protocol."""
    if isinstance(raw_outputs, tuple):
        return raw_outputs
    if isinstance(raw_outputs, list):
        return tuple(raw_outputs)
    return (raw_outputs,)


def _runtime_output_shape(
    output: TensorLike,
    /,
    *,
    operation: str,
    output_index: int,
) -> tuple[int, ...]:
    """Read a trusted shape directly and validate custom tensor protocols."""
    if is_trusted_tensor_type(type(output)):
        return output.shape
    return validated_output_shape(
        op_name=operation,
        output=output,
        output_index=output_index,
    )


def runtime_output_has_shape(
    output: TensorLike,
    expected_shape: tuple[int, ...],
    /,
    *,
    operation: str,
    output_index: int = 0,
) -> bool:
    """Return whether a runtime output has the required shape.

    Parameters
    ----------
    output
        Backend-produced tensor whose shape is inspected.
    expected_shape
        Canonical shape required by the runtime step.
    operation
        Operation name included in structured diagnostics.
    output_index
        Position of the output in a multi-output result.

    Returns
    -------
    bool
        Whether the output shape equals ``expected_shape``.

    Raises
    ------
    ExecutionError
        The output does not satisfy the TensorLike shape protocol.
    """
    return (
        _runtime_output_shape(
            output,
            operation=operation,
            output_index=output_index,
        )
        == expected_shape
    )


def validate_runtime_output_shape(
    output: TensorLike,
    expected_shape: tuple[int, ...],
    /,
    *,
    operation: str,
    output_index: int = 0,
) -> TensorLike:
    """Validate and return one shape-deterministic runtime output.

    Parameters
    ----------
    output
        Backend-produced tensor to validate.
    expected_shape
        Canonical shape required by the runtime step.
    operation
        Operation name included in structured diagnostics.
    output_index
        Position of the output in a multi-output result.

    Returns
    -------
    TensorLike
        ``output`` after its shape satisfies the runtime contract.

    Raises
    ------
    ExecutionError
        The output violates the TensorLike protocol or has a shape different
        from ``expected_shape``.
    """
    actual_shape = _runtime_output_shape(
        output,
        operation=operation,
        output_index=output_index,
    )
    if actual_shape == expected_shape:
        return output

    raise ExecutionError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            f"inconsistent dims: {operation} output shape does not match "
            f"runtime contract at output[{output_index}]"
        ),
        help="return a tensor whose shape matches the requested output shape",
        related=(f"{operation} runtime output",),
        data={
            "operation": operation,
            "output_index": output_index,
            "expected_shape": repr(expected_shape),
            "actual_shape": repr(actual_shape),
        },
    )


def backend_specialization_error(
    *,
    operation: str,
    error: Exception,
) -> ExecutionError:
    """Build one structured backend specialization failure.

    Parameters
    ----------
    operation
        Operation whose runtime capabilities were being specialized.
    error
        Backend exception raised during specialization.

    Returns
    -------
    ExecutionError
        Structured specialization failure for the operation boundary.
    """
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: {operation} specialization failed: {error}",
        help=f"ensure the active backend provides working {operation} capabilities",
        related=(f"{operation} specialization", "backend capability"),
        data={"operation": operation},
    )


def bind_runtime_backend(
    context: RuntimeSpecializationContext,
    /,
    *,
    operation: str,
    required_namespace_methods: tuple[str, ...],
    bind_namespace_when_backend_ops_available: bool,
) -> RuntimeBackendBinding:
    """Resolve backend ops and optional namespace binding for one step."""
    backend_profile = context.backend_profile
    if backend_profile is None:
        return RuntimeBackendBinding(backend_ops=None, xp=None)

    try:
        backend_ops = get_backend_array_ops(backend_profile.backend_family)
        xp: ArrayNamespace | None = None
        if bind_namespace_when_backend_ops_available or backend_ops is None:
            namespace_binding = ArrayNamespaceBinding(
                namespace=backend_profile.namespace
            )
            xp = namespace_binding.as_array_namespace(required_namespace_methods)
    except TensorOpError:
        raise
    except Exception as error:
        raise backend_specialization_error(
            operation=operation,
            error=error,
        ) from error
    return RuntimeBackendBinding(backend_ops=backend_ops, xp=xp)


__all__ = [
    "coerce_step_outputs",
    "runtime_output_has_shape",
    "validate_runtime_output_shape",
]
