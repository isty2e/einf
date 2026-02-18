from einf.backend import (
    BackendProfile,
    is_namespace_family,
    numpy_shares_memory,
    tensor_numel,
    torch_storage_ptr,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.tensor_types import TensorLike

from .constants import RESHAPE_STRUCTURAL_ERROR_CODES


def reshape_shares_memory(
    *,
    lhs: TensorLike,
    rhs: TensorLike,
    backend_profile: BackendProfile,
) -> bool | None:
    """Return whether one reshape output aliases input storage under one backend."""
    if tensor_numel(rhs) == 0:
        return True

    if is_namespace_family(backend_profile.namespace_id, "numpy"):
        numpy_shares = numpy_shares_memory(lhs=lhs, rhs=rhs)
        if numpy_shares is not None:
            return numpy_shares

    if is_namespace_family(backend_profile.namespace_id, "torch"):
        lhs_ptr = torch_storage_ptr(lhs)
        rhs_ptr = torch_storage_ptr(rhs)
        if lhs_ptr is None or rhs_ptr is None:
            return None
        return lhs_ptr == rhs_ptr

    return None


def normalize_zero_copy_reshape_error(
    error: ValidationError,
    *,
    operation: str,
    backend: str | None,
) -> ValidationError:
    """Convert reshape-structural failures into strict view diagnostics."""
    if error.code not in RESHAPE_STRUCTURAL_ERROR_CODES:
        return error
    payload_data: dict[str, str | int | bool] = {"operation": operation}
    if backend is not None:
        payload_data["backend"] = backend
    return ValidationError(
        code=ErrorCode.NOT_A_VIEW,
        message="not a view: reshape mapping is not representable without copying",
        help="restrict view reshape to runtime layouts that preserve zero-copy aliasing",
        related=("view affine mapping", "reshape"),
        data=payload_data,
    )


__all__ = [
    "normalize_zero_copy_reshape_error",
    "reshape_shares_memory",
]
