from collections.abc import Callable
from typing import cast

from einf.backend.namespace import ArrayNamespaceLike
from einf.backend.runtime import load_backend_module
from einf.tensor_types import TensorLike

EINSUM_FALLBACK_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


def try_native_contract_einsum(
    *,
    equation: str,
    tensors: tuple[TensorLike, ...],
    namespace: ArrayNamespaceLike,
) -> TensorLike | None:
    """Run native backend einsum for contract execution."""
    namespace_einsum_candidate = getattr(namespace, "einsum", None)
    if callable(namespace_einsum_candidate):
        namespace_einsum = cast(Callable[..., TensorLike], namespace_einsum_candidate)
        try:
            return namespace_einsum(equation, *tensors)
        except EINSUM_FALLBACK_ERRORS:
            return None

    torch_module = load_backend_module("torch")
    torch_einsum_candidate = getattr(torch_module, "einsum", None)
    if not callable(torch_einsum_candidate):
        return None
    torch_einsum = cast(Callable[..., TensorLike], torch_einsum_candidate)

    try:
        return torch_einsum(equation, *tensors)
    except EINSUM_FALLBACK_ERRORS:
        return None


__all__ = ["try_native_contract_einsum"]
