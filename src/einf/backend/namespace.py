from typing import Protocol, TypeAlias

BackendFamily: TypeAlias = str

_BACKEND_FAMILY_CANONICAL: dict[str, BackendFamily] = {
    "numpy": "numpy",
    "torch": "torch",
    "jax": "jax",
    "tensorflow": "tensorflow",
    "cupy": "cupy",
    "dask": "dask",
    "autograd": "autograd",
    "theano": "theano",
    "mlx": "mlx.core",
}


class ArrayNamespaceLike(Protocol):
    """Minimal Array API namespace protocol."""

    __name__: str


def derive_namespace_id(namespace: ArrayNamespaceLike) -> str:
    """Derive stable namespace identifier for modules and class-like namespaces."""
    namespace_name = getattr(namespace, "__name__", None)
    if not isinstance(namespace_name, str):
        raise TypeError("array namespace must define string __name__")

    stripped_name = namespace_name.strip()
    if not stripped_name:
        raise TypeError("array namespace __name__ cannot be empty")

    if "." in stripped_name:
        return stripped_name

    namespace_module = getattr(namespace, "__module__", "")
    if namespace_module is None:
        return stripped_name

    if not isinstance(namespace_module, str):
        raise TypeError("array namespace must define string __module__")

    stripped_module = namespace_module.strip()
    if not stripped_module:
        return stripped_name
    return f"{stripped_module}.{stripped_name}"


def infer_backend_family(namespace_id: str) -> BackendFamily | None:
    """Infer canonical backend family from namespace identifier."""
    normalized_namespace_id = namespace_id
    if normalized_namespace_id.startswith("array_api_compat."):
        normalized_namespace_id = normalized_namespace_id.removeprefix(
            "array_api_compat."
        )

    candidate = normalized_namespace_id.split(".")[0]
    return _BACKEND_FAMILY_CANONICAL.get(candidate)


def derive_family_key(namespace_id: str) -> str:
    """Derive backend-family key used for mixed-family validation."""
    backend_family = infer_backend_family(namespace_id)
    if backend_family is not None:
        return f"known:{backend_family}"
    return f"namespace:{namespace_id}"


def is_namespace_family(namespace_id: str, backend_family: BackendFamily) -> bool:
    """Return whether namespace id belongs to one canonical backend family."""
    return infer_backend_family(namespace_id) == backend_family
