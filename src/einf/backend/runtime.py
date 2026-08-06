import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Literal, Protocol, TypeGuard, cast, runtime_checkable

from array_api_compat import array_namespace

from ..tensor_types import TensorLike, trusted_tensor_family
from .namespace import ArrayNamespaceLike, BackendFamily, derive_namespace_id

ReducerFn = Callable[[TensorLike, tuple[int, ...]], TensorLike]
AxisKeyword = Literal["axis", "dim"]


class BackendRuntimeUnavailable(ImportError):
    """Signal that an optional backend-native runtime route is unavailable."""


@dataclass(frozen=True, slots=True)
class _BackendRuntimeSpec:
    """Backend runtime callable specification."""

    module_name: str
    reshape_name: str
    permute_name: str
    expand_dims_name: str
    expand_dims_axis_keyword: AxisKeyword | None
    broadcast_to_name: str
    concat_name: str
    concat_axis_keyword: AxisKeyword
    reducer_name_map: dict[str, str]
    reducer_axis_keyword: AxisKeyword


_BACKEND_RUNTIME_SPECS: dict[BackendFamily, _BackendRuntimeSpec] = {
    "numpy": _BackendRuntimeSpec(
        module_name="numpy",
        reshape_name="reshape",
        permute_name="transpose",
        expand_dims_name="expand_dims",
        expand_dims_axis_keyword=None,
        broadcast_to_name="broadcast_to",
        concat_name="concatenate",
        concat_axis_keyword="axis",
        reducer_name_map={
            "sum": "sum",
            "prod": "prod",
            "mean": "mean",
            "max": "max",
            "min": "min",
            "all": "all",
            "any": "any",
        },
        reducer_axis_keyword="axis",
    ),
    "torch": _BackendRuntimeSpec(
        module_name="torch",
        reshape_name="reshape",
        permute_name="permute",
        expand_dims_name="unsqueeze",
        expand_dims_axis_keyword="dim",
        broadcast_to_name="broadcast_to",
        concat_name="cat",
        concat_axis_keyword="dim",
        reducer_name_map={
            "sum": "sum",
            "prod": "prod",
            "mean": "mean",
            "max": "amax",
            "min": "amin",
            "all": "all",
            "any": "any",
        },
        reducer_axis_keyword="dim",
    ),
}


def supports_native_array_ops(backend_family: BackendFamily | None, /) -> bool:
    """Return whether a family has a backend-native array adapter.

    Parameters
    ----------
    backend_family
        Canonical backend family, if one was resolved.

    Returns
    -------
    bool
        Whether the family has a registered native adapter.
    """
    return backend_family in _BACKEND_RUNTIME_SPECS


@dataclass(frozen=True, slots=True)
class BackendArrayOps:
    """Backend-native array primitive call surface."""

    backend_family: BackendFamily
    reshape: Callable[[TensorLike, tuple[int, ...]], TensorLike]
    permute: Callable[[TensorLike, tuple[int, ...]], TensorLike]
    expand_dims: Callable[[TensorLike, int], TensorLike]
    broadcast_to: Callable[[TensorLike, tuple[int, ...]], TensorLike]
    concat: Callable[[list[TensorLike], int], TensorLike]
    reducers: dict[str, ReducerFn]

    def reduce(
        self,
        *,
        reducer_name: str,
        tensor: TensorLike,
        axes: tuple[int, ...],
    ) -> TensorLike:
        """Run one backend-native string reducer."""
        reducer = self.reducers.get(reducer_name)
        if reducer is None:
            raise ValueError(f"unsupported reducer {reducer_name!r}")
        return reducer(tensor, axes)


@runtime_checkable
class ArrayNamespace(Protocol):
    """Array namespace protocol required by execution internals."""

    __name__: str

    def reshape(self, tensor: TensorLike, shape: tuple[int, ...], /) -> TensorLike: ...

    def permute_dims(
        self, tensor: TensorLike, axes: tuple[int, ...], /
    ) -> TensorLike: ...

    def expand_dims(self, tensor: TensorLike, /, *, axis: int) -> TensorLike: ...

    def broadcast_to(
        self, tensor: TensorLike, shape: tuple[int, ...], /
    ) -> TensorLike: ...

    def concat(self, tensors: list[TensorLike], /, *, axis: int) -> TensorLike: ...

    def asarray(self, value: bool | complex, /) -> TensorLike: ...

    def sum(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def prod(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def mean(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def max(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def min(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def all(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...

    def any(self, tensor: TensorLike, /, *, axis: tuple[int, ...]) -> TensorLike: ...


def _is_array_namespace_with_methods(
    namespace: ArrayNamespaceLike,
    required_methods: tuple[str, ...],
) -> TypeGuard[ArrayNamespace]:
    """Return whether namespace supports required Array API methods."""
    for method_name in required_methods:
        method = getattr(namespace, method_name, None)
        if not callable(method):
            return False
    return True


@dataclass(frozen=True, slots=True)
class ArrayNamespaceBinding:
    """Resolved runtime namespace with capability checks."""

    namespace: ArrayNamespaceLike

    @property
    def namespace_id(self) -> str:
        """Return stable namespace identifier."""
        return derive_namespace_id(self.namespace)

    def missing_methods(self, required_methods: tuple[str, ...]) -> tuple[str, ...]:
        """Return missing namespace methods from required method set."""
        return tuple(
            method_name
            for method_name in required_methods
            if not callable(getattr(self.namespace, method_name, None))
        )

    def supports(self, required_methods: tuple[str, ...]) -> bool:
        """Return whether namespace supports required method set."""
        return not self.missing_methods(required_methods)

    def as_array_namespace(
        self,
        required_methods: tuple[str, ...],
    ) -> ArrayNamespace | None:
        """Return namespace as ArrayNamespace when required methods are present."""
        namespace = self.namespace
        if _is_array_namespace_with_methods(namespace, required_methods):
            return namespace
        return None


def bind_array_namespace(*tensors: TensorLike) -> ArrayNamespaceBinding:
    """Resolve Array API namespace binding from runtime tensors."""
    namespace = array_namespace(*tensors)
    return ArrayNamespaceBinding(namespace=namespace)


@cache
def resolve_backend_array_ops(
    backend_family: BackendFamily,
    /,
) -> BackendArrayOps:
    """Resolve backend-native primitive callables for one family."""
    spec = _BACKEND_RUNTIME_SPECS.get(backend_family)
    if spec is None:
        raise BackendRuntimeUnavailable(
            f"backend family has no native array adapter: {backend_family!r}"
        )

    backend_module = load_backend_module(backend_family)
    reducers: dict[str, ReducerFn] = {}
    for reducer_name, module_reducer_name in spec.reducer_name_map.items():
        if backend_family == "torch" and reducer_name == "prod":
            reducers[reducer_name] = _bind_torch_prod_reducer(
                backend_module,
                module_reducer_name=module_reducer_name,
            )
        else:
            reducers[reducer_name] = _bind_reducer_op(
                backend_module,
                module_reducer_name=module_reducer_name,
                axis_keyword=spec.reducer_axis_keyword,
            )

    if backend_family == "torch":
        module_reshape = _bind_tensor_shape_op(
            backend_module, module_op_name=spec.reshape_name
        )
        module_permute = _bind_tensor_axes_op(
            backend_module, module_op_name=spec.permute_name
        )
        module_expand_dims = _bind_expand_dims_op(
            backend_module,
            module_op_name=spec.expand_dims_name,
            axis_keyword=spec.expand_dims_axis_keyword,
        )
        module_broadcast_to = _bind_tensor_shape_op(
            backend_module, module_op_name=spec.broadcast_to_name
        )

        torch_tensor_type = getattr(backend_module, "Tensor", None)
        tensor_reshape = (
            getattr(torch_tensor_type, "reshape", None)
            if isinstance(torch_tensor_type, type)
            else None
        )
        tensor_permute = (
            getattr(torch_tensor_type, "permute", None)
            if isinstance(torch_tensor_type, type)
            else None
        )
        tensor_unsqueeze = (
            getattr(torch_tensor_type, "unsqueeze", None)
            if isinstance(torch_tensor_type, type)
            else None
        )
        tensor_expand = (
            getattr(torch_tensor_type, "expand", None)
            if isinstance(torch_tensor_type, type)
            else None
        )

        if (
            callable(tensor_reshape)
            and callable(tensor_permute)
            and callable(tensor_unsqueeze)
            and callable(tensor_expand)
        ):
            tensor_reshape_op = cast(Callable[..., TensorLike], tensor_reshape)
            tensor_permute_op = cast(Callable[..., TensorLike], tensor_permute)
            tensor_unsqueeze_op = cast(Callable[..., TensorLike], tensor_unsqueeze)
            tensor_expand_op = cast(Callable[..., TensorLike], tensor_expand)
            return BackendArrayOps(
                backend_family=backend_family,
                reshape=lambda tensor, shape: tensor_reshape_op(tensor, shape),
                permute=lambda tensor, axes: tensor_permute_op(tensor, *axes),
                expand_dims=lambda tensor, axis: tensor_unsqueeze_op(tensor, axis),
                broadcast_to=lambda tensor, shape: tensor_expand_op(tensor, *shape),
                concat=_bind_concat_op(
                    backend_module,
                    module_op_name=spec.concat_name,
                    axis_keyword=spec.concat_axis_keyword,
                ),
                reducers=reducers,
            )

        return BackendArrayOps(
            backend_family=backend_family,
            reshape=module_reshape,
            permute=module_permute,
            expand_dims=module_expand_dims,
            broadcast_to=module_broadcast_to,
            concat=_bind_concat_op(
                backend_module,
                module_op_name=spec.concat_name,
                axis_keyword=spec.concat_axis_keyword,
            ),
            reducers=reducers,
        )

    return BackendArrayOps(
        backend_family=backend_family,
        reshape=_bind_tensor_shape_op(
            backend_module,
            module_op_name=spec.reshape_name,
        ),
        permute=_bind_tensor_axes_op(
            backend_module,
            module_op_name=spec.permute_name,
        ),
        expand_dims=_bind_expand_dims_op(
            backend_module,
            module_op_name=spec.expand_dims_name,
            axis_keyword=spec.expand_dims_axis_keyword,
        ),
        broadcast_to=_bind_tensor_shape_op(
            backend_module,
            module_op_name=spec.broadcast_to_name,
        ),
        concat=_bind_concat_op(
            backend_module,
            module_op_name=spec.concat_name,
            axis_keyword=spec.concat_axis_keyword,
        ),
        reducers=reducers,
    )


def get_backend_array_ops(
    backend_family: BackendFamily | None,
    /,
) -> BackendArrayOps | None:
    """Resolve backend-native primitive callables when family is supported."""
    if not supports_native_array_ops(backend_family):
        return None
    try:
        return resolve_backend_array_ops(backend_family)
    except BackendRuntimeUnavailable:
        return None


def is_trusted_backend_array_ops(
    *,
    backend_ops: BackendArrayOps,
    tensor: TensorLike,
) -> bool:
    """Return whether an adapter and tensor form one canonical native route.

    Parameters
    ----------
    backend_ops
        Backend adapter selected for the route.
    tensor
        Runtime tensor passed to the adapter.

    Returns
    -------
    bool
        Whether both objects match one canonical built-in runtime route.
    """
    family = trusted_tensor_family(type(tensor))
    if family is None:
        return False
    try:
        canonical_ops = resolve_backend_array_ops(family)
    except BackendRuntimeUnavailable:
        return False
    return backend_ops is canonical_ops and is_backend_runtime_uninterposed(family)


def is_backend_runtime_uninterposed(backend_family: BackendFamily, /) -> bool:
    """Return whether native calls cannot be intercepted by a backend mode.

    Parameters
    ----------
    backend_family
        Canonical backend family for the native call.

    Returns
    -------
    bool
        Whether the runtime has no active user override mode.
    """
    if backend_family != "torch":
        return True

    try:
        torch_module = load_backend_module("torch")
        torch_c = torch_module._C
        function_mode_enabled = torch_c._is_torch_function_mode_enabled
        if not callable(function_mode_enabled) or function_mode_enabled():
            return False

        python_dispatch = sys.modules.get("torch.utils._python_dispatch")
        if python_dispatch is None:
            return True
        dispatch_stack_length = getattr(
            python_dispatch,
            "_len_torch_dispatch_stack",
            None,
        )
        return callable(dispatch_stack_length) and dispatch_stack_length() == 0
    except Exception:  # noqa: BLE001 - missing proof selects the validated route
        return False


@cache
def load_backend_module(backend_family: BackendFamily) -> ModuleType:
    """Load one backend runtime module lazily.

    Parameters
    ----------
    backend_family
        Registered backend family whose native runtime module is required.

    Returns
    -------
    ModuleType
        Imported backend runtime module.

    Raises
    ------
    BackendRuntimeUnavailable
        The optional module or one of its loader dependencies is unavailable.
    RuntimeError
        The backend module fails during runtime initialization.
    Notes
    -----
    Backend initialization errors such as ``RuntimeError`` are preserved. They
    indicate a broken runtime rather than an unavailable optional route.
    """
    runtime_spec = _BACKEND_RUNTIME_SPECS.get(backend_family)
    if runtime_spec is None:
        raise BackendRuntimeUnavailable(
            f"backend family has no native runtime adapter: {backend_family!r}"
        )
    module_name = runtime_spec.module_name
    try:
        return import_module(module_name)
    except (ImportError, OSError) as error:
        raise BackendRuntimeUnavailable(
            f"backend runtime module is unavailable: {module_name!r}"
        ) from error


def _resolve_module_op(
    backend_module: ModuleType,
    /,
    *,
    module_op_name: str,
) -> Callable[..., TensorLike]:
    """Resolve one backend module callable and fail when unavailable."""
    try:
        op_candidate = getattr(backend_module, module_op_name, None)
    except (ImportError, OSError) as error:
        raise BackendRuntimeUnavailable(
            "backend runtime module callable is unavailable: "
            f"{backend_module.__name__}.{module_op_name}"
        ) from error
    if not callable(op_candidate):
        raise BackendRuntimeUnavailable(
            "backend runtime module callable is unavailable: "
            f"{backend_module.__name__}.{module_op_name}"
        )
    return cast(Callable[..., TensorLike], op_candidate)


def _bind_tensor_shape_op(
    backend_module: ModuleType,
    /,
    *,
    module_op_name: str,
) -> Callable[[TensorLike, tuple[int, ...]], TensorLike]:
    """Bind tensor-shape callable `(tensor, shape)`."""
    op = _resolve_module_op(
        backend_module,
        module_op_name=module_op_name,
    )
    return lambda tensor, shape, op=op: op(tensor, shape)


def _bind_tensor_axes_op(
    backend_module: ModuleType,
    /,
    *,
    module_op_name: str,
) -> Callable[[TensorLike, tuple[int, ...]], TensorLike]:
    """Bind tensor-axes callable `(tensor, axes)`."""
    op = _resolve_module_op(
        backend_module,
        module_op_name=module_op_name,
    )
    return lambda tensor, axes, op=op: op(tensor, axes)


def _bind_expand_dims_op(
    backend_module: ModuleType,
    /,
    *,
    module_op_name: str,
    axis_keyword: AxisKeyword | None,
) -> Callable[[TensorLike, int], TensorLike]:
    """Bind expand-dims callable using positional or keyword axis."""
    op = _resolve_module_op(
        backend_module,
        module_op_name=module_op_name,
    )
    if axis_keyword is None:
        return lambda tensor, axis, op=op: op(tensor, axis)
    if axis_keyword == "axis":
        return lambda tensor, axis, op=op: op(tensor, axis=axis)
    return lambda tensor, axis, op=op: op(tensor, dim=axis)


def _bind_concat_op(
    backend_module: ModuleType,
    /,
    *,
    module_op_name: str,
    axis_keyword: AxisKeyword,
) -> Callable[[list[TensorLike], int], TensorLike]:
    """Bind concat callable using backend-specific axis keyword."""
    op = _resolve_module_op(
        backend_module,
        module_op_name=module_op_name,
    )
    if axis_keyword == "axis":
        return lambda tensors, axis, op=op: op(tensors, axis=axis)
    return lambda tensors, axis, op=op: op(tensors, dim=axis)


def _bind_reducer_op(
    backend_module: ModuleType,
    /,
    *,
    module_reducer_name: str,
    axis_keyword: AxisKeyword,
) -> ReducerFn:
    """Bind reducer callable using backend-specific axis keyword."""
    reducer = _resolve_module_op(
        backend_module,
        module_op_name=module_reducer_name,
    )
    if axis_keyword == "axis":
        return lambda tensor, axes, reducer=reducer: reducer(tensor, axis=axes)
    return lambda tensor, axes, reducer=reducer: reducer(tensor, dim=axes)


def _bind_torch_prod_reducer(
    backend_module: ModuleType,
    /,
    *,
    module_reducer_name: str,
) -> ReducerFn:
    """Bind Torch product reduction across one or more canonical axes."""
    reducer = _resolve_module_op(
        backend_module,
        module_op_name=module_reducer_name,
    )

    def reduce_prod(tensor: TensorLike, axes: tuple[int, ...]) -> TensorLike:
        reduced = tensor
        for axis in sorted(axes, reverse=True):
            reduced = reducer(reduced, dim=axis)
        return reduced

    return reduce_prod
