from typing import Literal

from einf.diagnostics import ErrorCode

RESHAPE_REQUIRED_NAMESPACE_METHODS = ("reshape",)
ZERO_COPY_ALLOWED_RESHAPE_MODE = "allow_copy"
ZERO_COPY_REQUIRED_RESHAPE_MODE = "require_zero_copy"
RESHAPE_STRUCTURAL_ERROR_CODES = frozenset(
    (
        ErrorCode.INCONSISTENT_DIMS.value,
        ErrorCode.NUMEL_MISMATCH_GROW.value,
        ErrorCode.NUMEL_MISMATCH_SHRINK.value,
    )
)

ZeroCopyReshapeMode = Literal["allow_copy", "require_zero_copy"]


__all__ = [
    "ZeroCopyReshapeMode",
]
