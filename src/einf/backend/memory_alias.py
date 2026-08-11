from collections.abc import Callable
from math import prod

from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError
from einf.tensor_types import TensorLike, trusted_tensor_family

from .runtime import BackendRuntimeUnavailable, load_backend_module


def _try_storage_state(
    storage_factory: Callable[[], object],
) -> tuple[object, int | None] | None:
    storage = storage_factory()
    if storage is None:
        return None
    data_ptr_fn = getattr(storage, "data_ptr", None)
    data_ptr = data_ptr_fn() if callable(data_ptr_fn) else None
    if not isinstance(data_ptr, int) or data_ptr == 0:
        data_ptr = None
    return storage, data_ptr


def _torch_storage_state(
    tensor: TensorLike,
    tensor_type: type[object],
    /,
) -> tuple[object, int | None] | None:
    untyped_storage = getattr(tensor_type, "untyped_storage", None)
    if callable(untyped_storage):
        return _try_storage_state(lambda: untyped_storage(tensor))

    storage = getattr(tensor_type, "storage", None)
    if callable(storage):
        return _try_storage_state(lambda: storage(tensor))

    return None


def torch_shares_storage(*, lhs: TensorLike, rhs: TensorLike) -> bool | None:
    """Return whether Torch tensors provably share backing storage.

    Parameters
    ----------
    lhs
        First tensor in the storage comparison.
    rhs
        Second tensor in the storage comparison.

    Returns
    -------
    bool or None
        Whether the tensors share storage, or ``None`` when storage identity
        cannot be established.
    """
    try:
        torch_module = load_backend_module("torch")
    except BackendRuntimeUnavailable:
        return None
    except TensorOpError:
        raise
    except Exception as error:
        raise _torch_alias_error(error) from error

    try:
        tensor_type = getattr(torch_module, "Tensor", None)
        if not isinstance(tensor_type, type):
            return None
        if type(lhs) is not tensor_type or type(rhs) is not tensor_type:
            return None
        lhs_state = _torch_storage_state(lhs, tensor_type)
        rhs_state = _torch_storage_state(rhs, tensor_type)
    except TensorOpError:
        raise
    except Exception as error:
        raise _torch_alias_error(error) from error
    if lhs_state is None or rhs_state is None:
        return None

    lhs_storage, lhs_ptr = lhs_state
    rhs_storage, rhs_ptr = rhs_state
    if lhs_ptr is not None and rhs_ptr is not None:
        return lhs_ptr == rhs_ptr
    if lhs_storage is rhs_storage:
        return True
    return None


def numpy_shares_storage(*, lhs: TensorLike, rhs: TensorLike) -> bool | None:
    """Return whether NumPy tensors provably share backing storage.

    Parameters
    ----------
    lhs
        First tensor in the storage comparison.
    rhs
        Second tensor in the storage comparison.

    Returns
    -------
    bool or None
        Whether the tensors share storage, or ``None`` when the NumPy runtime
        cannot establish the relationship.

    Raises
    ------
    ExecutionError
        The NumPy runtime or its storage-inspection capability is broken.
    """
    if not (
        _is_exact_trusted_family_tensor(lhs, "numpy")
        and _is_exact_trusted_family_tensor(rhs, "numpy")
    ):
        return None

    try:
        numpy_module = load_backend_module("numpy")
    except BackendRuntimeUnavailable:
        return None
    except TensorOpError:
        raise
    except Exception as error:
        raise _numpy_alias_error(error) from error

    try:
        shares_memory = getattr(numpy_module, "shares_memory", None)
    except TensorOpError:
        raise
    except Exception as error:
        raise _numpy_alias_error(error) from error
    if not callable(shares_memory):
        error = AttributeError("numpy runtime has no callable shares_memory")
        raise _numpy_alias_error(error) from error

    if tensor_numel(rhs) == 0:
        try:
            array_type = getattr(numpy_module, "ndarray", None)
        except TensorOpError:
            raise
        except Exception as error:
            raise _numpy_alias_error(error) from error
        if (
            not isinstance(array_type, type)
            or type(lhs) is not array_type
            or type(rhs) is not array_type
        ):
            return None
        lhs_bases = _numpy_base_chain(lhs)
        rhs_bases = _numpy_base_chain(rhs)
        if lhs_bases is None or rhs_bases is None:
            return None
        return not lhs_bases.isdisjoint(rhs_bases)

    try:
        return bool(shares_memory(lhs, rhs))
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return None
    except Exception as error:
        raise _numpy_alias_error(error) from error


def _is_exact_trusted_family_tensor(tensor: TensorLike, family: str, /) -> bool:
    """Return whether one value has the exact trusted tensor type for a family."""
    return trusted_tensor_family(type(tensor)) == family


def _numpy_base_chain(array: object, /) -> frozenset[int] | None:
    """Return object identities along one NumPy base chain."""
    identities: set[int] = set()
    current: object | None = array
    while current is not None and id(current) not in identities:
        identities.add(id(current))
        try:
            current = getattr(current, "base", None)
        except Exception:  # noqa: BLE001 - an opaque base chain proves nothing
            return None
    return frozenset(identities)


def _numpy_alias_error(error: Exception, /) -> ExecutionError:
    """Build one structured NumPy alias-capability failure."""
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: NumPy shares_memory capability failed: {error}",
        help="ensure the NumPy runtime provides a working shares_memory capability",
        related=("view zero-copy check", "backend capability"),
        data={"operation": "view"},
    )


def _torch_alias_error(error: Exception, /) -> ExecutionError:
    """Build one structured Torch storage-capability failure."""
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: Torch storage capability failed: {error}",
        help="ensure the Torch runtime provides working tensor storage capabilities",
        related=("view zero-copy check", "backend capability"),
        data={"operation": "view"},
    )


def tensor_numel(tensor: TensorLike) -> int:
    """Return element count for one tensor shape."""
    return prod(tensor.shape, start=1)


__all__ = [
    "numpy_shares_storage",
    "tensor_numel",
    "torch_shares_storage",
]
