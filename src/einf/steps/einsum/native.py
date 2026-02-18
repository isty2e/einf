from einf.backend.namespace import ArrayNamespaceLike
from einf.backend.runtime import load_backend_module
from einf.tensor_types import TensorLike


def try_native_contract_einsum(
    *,
    equation: str,
    tensors: tuple[TensorLike, ...],
    namespace: ArrayNamespaceLike,
) -> TensorLike | None:
    """Run native backend einsum for contract execution."""
    namespace_einsum = getattr(namespace, "einsum", None)
    if callable(namespace_einsum):
        try:
            return namespace_einsum(equation, *tensors)
        except Exception:
            return None

    torch_module = load_backend_module("torch")
    torch_einsum = getattr(torch_module, "einsum", None)
    if not callable(torch_einsum):
        return None

    try:
        return torch_einsum(equation, *tensors)
    except Exception:
        return None


__all__ = ["try_native_contract_einsum"]
