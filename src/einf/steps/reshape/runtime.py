from math import prod

from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    bind_array_namespace,
)
from einf.backend.namespace import derive_namespace_id, is_namespace_family
from einf.backend.runtime import BackendRuntimeUnavailable, load_backend_module
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError, ValidationError
from einf.steps.runtime import (
    FALLBACK_ELIGIBLE_BACKEND_ERRORS,
    runtime_output_has_shape,
)
from einf.tensor_types import TensorLike

from .constants import (
    RESHAPE_REQUIRED_NAMESPACE_METHODS,
    ZERO_COPY_ALLOWED_RESHAPE_MODE,
    ZERO_COPY_REQUIRED_RESHAPE_MODE,
    ZeroCopyReshapeMode,
)
from .model import ReshapeCompiledProgram
from .resolve import resolve_reshape_target_shape


def validate_rearrange_numel(
    *,
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> None:
    """Validate rearrange numel preservation for one unary reshape transform."""
    input_total = prod(input_shape, start=1)
    output_total = prod(target_shape, start=1)
    if output_total > input_total:
        raise ValidationError(
            code=ErrorCode.NUMEL_MISMATCH_GROW,
            message=(
                "numel mismatch grow: rearrange output numel is larger than input numel"
            ),
            help="keep output numel equal to input",
            related=("rearrange schema",),
            data={"operation": "rearrange"},
        )
    if output_total < input_total:
        raise ValidationError(
            code=ErrorCode.NUMEL_MISMATCH_SHRINK,
            message=(
                "numel mismatch shrink: rearrange output numel is smaller than input numel"
            ),
            help="keep output numel equal to input",
            related=("rearrange schema",),
            data={"operation": "rearrange"},
        )


def run_reshape_program(
    *,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
    backend_ops: BackendArrayOps | None,
    xp: ArrayNamespace | None,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
) -> TensorLike:
    """Run one unary reshape program via backend-native reshape primitives."""
    output = try_reshape_to_shape(
        tensor=tensor,
        target_shape=target_shape,
        backend_ops=backend_ops,
        xp=xp,
        zero_copy_mode=zero_copy_mode,
    )
    if output is not None:
        return output

    if zero_copy_mode == ZERO_COPY_REQUIRED_RESHAPE_MODE:
        raise ValidationError(
            code=ErrorCode.NOT_A_VIEW,
            message="not a view: reshape backend could not produce a zero-copy output",
            help="use a layout and mapping supported as a zero-copy backend view",
            related=("view affine mapping", "reshape runtime"),
            data={"operation": "view"},
        )

    raise ExecutionError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: reshape backend could not produce the "
            "requested output shape"
        ),
        help="ensure the backend supports reshape for the requested tensor layout",
        related=("reshape runtime",),
        data={
            "operation": "reshape",
            "expected_shape": repr(target_shape),
        },
    )


def try_reshape_to_shape(
    *,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
    backend_ops: BackendArrayOps | None,
    xp: ArrayNamespace | None,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
) -> TensorLike | None:
    """Try validated reshape routes.

    Parameters
    ----------
    tensor
        Tensor to reshape.
    target_shape
        Required output shape.
    backend_ops
        Backend-native primitives when available.
    xp
        Array namespace fallback when available.
    zero_copy_mode
        Whether the route may copy storage.

    Returns
    -------
    TensorLike or None
        A shape-valid output, or ``None`` when no route can satisfy the request.

    Raises
    ------
    TensorOpError
        A backend route violates a structured runtime contract.
    """
    if tensor.shape == target_shape:
        return tensor

    strict_numpy_route = False
    if zero_copy_mode == ZERO_COPY_REQUIRED_RESHAPE_MODE:
        strict_numpy_route = (
            backend_ops is not None and backend_ops.backend_family == "numpy"
        ) or _is_numpy_namespace(xp)
        if strict_numpy_route:
            strict_output = _try_strict_numpy_reshape(
                tensor=tensor,
                target_shape=target_shape,
            )
            if strict_output is not None:
                return strict_output
            return None

    if backend_ops is not None and not strict_numpy_route:
        try:
            output = backend_ops.reshape(tensor, target_shape)
        except TensorOpError:
            raise
        except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
            if xp is None:
                return None
        except Exception as error:
            raise reshape_route_error(error) from error
        else:
            if runtime_output_has_shape(
                output,
                target_shape,
                operation="reshape",
            ):
                return output
            if xp is None:
                return None

    if xp is None:
        return None
    return _try_namespace_reshape(
        tensor=tensor,
        target_shape=target_shape,
        xp=xp,
    )


def _is_numpy_namespace(xp: ArrayNamespace | None, /) -> bool:
    """Return whether one optional namespace belongs to NumPy."""
    if xp is None:
        return False
    try:
        namespace_id = derive_namespace_id(xp)
    except TensorOpError:
        raise
    except TypeError:
        return False
    except Exception as error:
        raise reshape_route_error(error) from error
    return is_namespace_family(namespace_id, "numpy")


def _try_strict_numpy_reshape(
    *,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
) -> TensorLike | None:
    """Try the NumPy order-preserving reshape route once."""
    try:
        numpy_module = load_backend_module("numpy")
    except BackendRuntimeUnavailable:
        return None
    except TensorOpError:
        raise
    except Exception as error:
        raise reshape_route_error(error) from error

    try:
        output = numpy_module.reshape(tensor, target_shape, order="A")
    except TensorOpError:
        raise
    except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
        return None
    except Exception as error:
        raise reshape_route_error(error) from error

    if runtime_output_has_shape(output, target_shape, operation="reshape"):
        return output
    return None


def _try_namespace_reshape(
    *,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
    xp: ArrayNamespace,
) -> TensorLike | None:
    """Try one namespace reshape route."""
    if tensor.shape == target_shape:
        return tensor
    try:
        output = xp.reshape(tensor, target_shape)
    except TensorOpError:
        raise
    except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
        return None
    except Exception as error:
        raise reshape_route_error(error) from error
    if runtime_output_has_shape(output, target_shape, operation="reshape"):
        return output
    return None


def reshape_route_error(error: Exception) -> ExecutionError:
    """Build one structured error for an unexpected reshape route failure."""
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: reshape backend failed: {error}",
        help=(
            "ensure the backend supports the required reshape "
            "operation on the given tensor layout"
        ),
        related=("reshape runtime",),
        data={"operation": "reshape"},
    )


def try_run_reshape_program(
    *,
    tensor: TensorLike,
    explicit_sizes: dict[str, int],
    program: ReshapeCompiledProgram,
    backend_ops: BackendArrayOps | None = None,
    xp: ArrayNamespace | None = None,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
) -> TensorLike | None:
    """Try symbolic unary-reshape runtime; return None when ineligible."""
    try:
        target_shape = resolve_reshape_target_shape(
            tensor=tensor,
            explicit_sizes=explicit_sizes,
            program=program,
        )
    except TensorOpError:
        raise
    except Exception as error:
        raise reshape_route_error(error) from error
    if target_shape is None:
        return None
    validate_rearrange_numel(
        input_shape=tensor.shape,
        target_shape=target_shape,
    )

    namespace_xp = xp
    if namespace_xp is None and backend_ops is None:
        try:
            namespace_binding = bind_array_namespace(tensor)
        except TensorOpError:
            raise
        except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
            return None
        namespace_xp = namespace_binding.as_array_namespace(
            RESHAPE_REQUIRED_NAMESPACE_METHODS
        )
        if namespace_xp is None:
            return None

    return try_reshape_to_shape(
        tensor=tensor,
        target_shape=target_shape,
        backend_ops=backend_ops,
        xp=namespace_xp,
        zero_copy_mode=zero_copy_mode,
    )


__all__ = [
    "run_reshape_program",
    "try_reshape_to_shape",
    "try_run_reshape_program",
]
