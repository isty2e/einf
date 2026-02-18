from dataclasses import dataclass

from einf.backend import (
    ArrayNamespace,
    ArrayNamespaceBinding,
    BackendArrayOps,
    get_backend_array_ops,
)
from einf.steps.base import RuntimeSpecializationContext
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class _RuntimeBackendBinding:
    """Internal backend binding resolved during step specialization."""

    backend_ops: BackendArrayOps | None
    xp: ArrayNamespace | None


def coerce_step_outputs(
    raw_outputs: TensorLike | tuple[TensorLike, ...] | list[TensorLike],
    /,
) -> tuple[TensorLike, ...]:
    """Normalize runtime outputs to runtime-step tuple protocol."""
    if isinstance(raw_outputs, tuple):
        return raw_outputs
    if isinstance(raw_outputs, list):
        return tuple(raw_outputs)
    return (raw_outputs,)


def bind_runtime_backend(
    context: RuntimeSpecializationContext,
    /,
    *,
    required_namespace_methods: tuple[str, ...],
    bind_namespace_when_backend_ops_available: bool,
) -> _RuntimeBackendBinding:
    """Resolve backend ops and optional namespace binding for one step."""
    backend_profile = context.backend_profile
    if backend_profile is None:
        return _RuntimeBackendBinding(backend_ops=None, xp=None)

    backend_ops = get_backend_array_ops(backend_profile.backend_family)
    xp: ArrayNamespace | None = None
    if bind_namespace_when_backend_ops_available or backend_ops is None:
        namespace_binding = ArrayNamespaceBinding(namespace=backend_profile.namespace)
        xp = namespace_binding.as_array_namespace(required_namespace_methods)
    return _RuntimeBackendBinding(backend_ops=backend_ops, xp=xp)


__all__ = [
    "coerce_step_outputs",
]
