from collections.abc import Callable
from math import prod

from einf.tensor_types import TensorLike

from .runtime import load_backend_module


def _try_storage_data_ptr(storage_factory: Callable[[], object]) -> int | None:
    try:
        storage = storage_factory()
        data_ptr_fn = getattr(storage, "data_ptr", None)
        if not callable(data_ptr_fn):
            return None
        data_ptr = data_ptr_fn()
    except (AttributeError, RuntimeError, TypeError):
        return None
    return data_ptr if isinstance(data_ptr, int) else None


def torch_storage_ptr(tensor: TensorLike) -> int | None:
    """Return torch storage base pointer when available."""
    untyped_storage = getattr(tensor, "untyped_storage", None)
    if callable(untyped_storage):
        data_ptr = _try_storage_data_ptr(untyped_storage)
        if data_ptr is not None:
            return data_ptr

    storage = getattr(tensor, "storage", None)
    if callable(storage):
        return _try_storage_data_ptr(storage)

    return None


def numpy_shares_memory(*, lhs: TensorLike, rhs: TensorLike) -> bool | None:
    """Run NumPy shares-memory check when numpy runtime module is available."""
    numpy_module = load_backend_module("numpy")
    try:
        return bool(numpy_module.shares_memory(lhs, rhs))
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return None


def tensor_numel(tensor: TensorLike) -> int:
    """Return element count for one tensor shape."""
    return prod(tensor.shape, start=1)


__all__ = [
    "numpy_shares_memory",
    "tensor_numel",
    "torch_storage_ptr",
]
