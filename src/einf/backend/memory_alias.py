from math import prod

from einf.tensor_types import TensorLike

from .runtime import load_backend_module


def torch_storage_ptr(tensor: TensorLike) -> int | None:
    """Return torch storage base pointer when available."""
    untyped_storage = getattr(tensor, "untyped_storage", None)
    if callable(untyped_storage):
        try:
            storage = untyped_storage()
            data_ptr = storage.data_ptr()
            if isinstance(data_ptr, int):
                return data_ptr
        except Exception:
            pass

    storage = getattr(tensor, "storage", None)
    if callable(storage):
        try:
            storage_obj = storage()
            data_ptr = storage_obj.data_ptr()
            if isinstance(data_ptr, int):
                return data_ptr
        except Exception:
            pass

    return None


def numpy_shares_memory(*, lhs: TensorLike, rhs: TensorLike) -> bool | None:
    """Run NumPy shares-memory check when numpy runtime module is available."""
    numpy_module = load_backend_module("numpy")
    try:
        return bool(numpy_module.shares_memory(lhs, rhs))
    except Exception:
        return None


def tensor_numel(tensor: TensorLike) -> int:
    """Return element count for one tensor shape."""
    return prod(tensor.shape, start=1)


__all__ = [
    "numpy_shares_memory",
    "tensor_numel",
    "torch_storage_ptr",
]
