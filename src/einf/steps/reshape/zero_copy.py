from einf.backend import BackendProfile
from einf.backend.memory_alias import (
    numpy_shares_storage,
    torch_shares_storage,
)
from einf.backend.namespace import is_namespace_family
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
    if is_namespace_family(backend_profile.namespace_id, "numpy"):
        numpy_shares = numpy_shares_storage(lhs=lhs, rhs=rhs)
        if numpy_shares is not None:
            return numpy_shares

    if is_namespace_family(backend_profile.namespace_id, "torch"):
        return torch_shares_storage(lhs=lhs, rhs=rhs)

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
