from dataclasses import dataclass

import numpy as np
import pytest

from einf import ExecutionError, ax, axes, packs
from einf.backend import BACKEND_RESOLVER, ArrayNamespaceLike, BackendProfile
from einf.diagnostics import ErrorCode, ValidationError
from einf.operations import rearrange, reduce, repeat, view
from einf.steps.reduce import build as reduce_build_module


class _SwitchingNamespace:
    def __init__(self, namespace_id: str, marker: str) -> None:
        self.__name__ = namespace_id
        self.marker = marker

    def permute_dims(
        self,
        tensor: "_SwitchingNamespaceTensor",
        permutation: tuple[int, ...],
    ) -> "_SwitchingNamespaceTensor":
        return _SwitchingNamespaceTensor(
            shape=tuple(tensor.shape[index] for index in permutation),
            namespace=self,
            marker=self.marker,
        )

    def expand_dims(
        self,
        tensor: "_SwitchingNamespaceTensor",
        *,
        axis: int,
    ) -> "_SwitchingNamespaceTensor":
        shape = list(tensor.shape)
        shape.insert(axis, 1)
        return _SwitchingNamespaceTensor(
            shape=tuple(shape),
            namespace=self,
            marker=self.marker,
        )

    def broadcast_to(
        self,
        tensor: "_SwitchingNamespaceTensor",
        shape: tuple[int, ...],
    ) -> "_SwitchingNamespaceTensor":
        return _SwitchingNamespaceTensor(
            shape=shape,
            namespace=self,
            marker=self.marker,
        )

    def asarray(self, value: object) -> "_SwitchingNamespaceTensor":
        _ = value
        return _SwitchingNamespaceTensor(shape=(), namespace=self, marker=self.marker)

    def _reduce(
        self,
        tensor: "_SwitchingNamespaceTensor",
        *,
        axis: tuple[int, ...],
    ) -> "_SwitchingNamespaceTensor":
        return _SwitchingNamespaceTensor(
            shape=tuple(
                dimension
                for index, dimension in enumerate(tensor.shape)
                if index not in axis
            ),
            namespace=self,
            marker=self.marker,
        )

    sum = _reduce
    prod = _reduce
    mean = _reduce
    max = _reduce
    min = _reduce
    all = _reduce
    any = _reduce


@dataclass(frozen=True, slots=True)
class _SwitchingNamespaceTensor:
    shape: tuple[int, ...]
    namespace: _SwitchingNamespace
    marker: str = "input"

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> ArrayNamespaceLike:
        _ = api_version
        return self.namespace

    def __getitem__(self, key: object) -> "_SwitchingNamespaceTensor":
        _ = key
        return self


def _namespace_sum(
    tensor: _SwitchingNamespaceTensor,
    *,
    axis: tuple[int, ...],
) -> _SwitchingNamespaceTensor:
    return tensor.namespace.sum(tensor, axis=axis)


class _AlternatingOutputNamespace:
    __name__ = "custom.alternating_output"

    def __init__(self) -> None:
        self.call_count = 0

    def permute_dims(
        self,
        tensor: "_AlternatingOutputTensor",
        permutation: tuple[int, ...],
    ) -> object:
        self.call_count += 1
        if self.call_count > 1:
            return object()
        return _AlternatingOutputTensor(
            shape=tuple(tensor.shape[index] for index in permutation),
            namespace=self,
        )


@dataclass(frozen=True, slots=True)
class _AlternatingOutputTensor:
    shape: tuple[int, ...]
    namespace: _AlternatingOutputNamespace

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> ArrayNamespaceLike:
        _ = api_version
        return self.namespace

    def __getitem__(self, key: object) -> "_AlternatingOutputTensor":
        _ = key
        return self


def test_backend_profile_resolution_distinguishes_namespaces_for_one_tensor_type() -> (
    None
):
    b, c = axes("profile_b", "profile_c")
    op = rearrange(ax[b, c], ax[c, b])
    namespace_a = _SwitchingNamespace("custom.backend_a", "a")
    namespace_b = _SwitchingNamespace("custom.backend_b", "b")
    tensor_a = _SwitchingNamespaceTensor((2, 3), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((2, 3), namespace_b)

    profile_a = op.abstract_plan.resolve_backend_profile((tensor_a,))
    profile_b = op.abstract_plan.resolve_backend_profile((tensor_b,))

    assert profile_a.namespace_id == "custom.backend_a"
    assert profile_b.namespace_id == "custom.backend_b"
    assert profile_a.execution_identity != profile_b.execution_identity


def test_shape_free_runner_cache_distinguishes_namespaces_for_one_tensor_type() -> None:
    b, c = axes("runner_b", "runner_c")
    op = rearrange(ax[b, c], ax[c, b])
    namespace_a = _SwitchingNamespace("custom.backend_a", "a")
    namespace_b = _SwitchingNamespace("custom.backend_b", "b")
    tensor_a = _SwitchingNamespaceTensor((2, 3), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((2, 3), namespace_b)

    output_a = op(tensor_a)
    output_b = op(tensor_b)

    assert isinstance(output_a, _SwitchingNamespaceTensor)
    assert isinstance(output_b, _SwitchingNamespaceTensor)
    assert output_a.marker == "a"
    assert output_b.marker == "b"


def test_shape_free_runner_validates_every_untrusted_backend_output() -> None:
    b, c = axes("untrusted_output_b", "untrusted_output_c")
    op = rearrange(ax[b, c], ax[c, b])
    namespace = _AlternatingOutputNamespace()
    tensor = _AlternatingOutputTensor((2, 3), namespace)

    first_output = op(tensor)

    assert isinstance(first_output, _AlternatingOutputTensor)
    with pytest.raises(ExecutionError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_shape_dependent_runner_cache_distinguishes_namespaces_for_one_tensor_type() -> (
    None
):
    (a,) = axes("dynamic_backend_a")
    op = repeat(ax[(1 + a)], ax[(1 + a), a])
    namespace_a = _SwitchingNamespace("custom.backend_a", "a")
    namespace_b = _SwitchingNamespace("custom.backend_b", "b")
    tensor_a = _SwitchingNamespaceTensor((3,), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((3,), namespace_b)

    output_a = op(tensor_a)
    output_b = op(tensor_b)

    assert isinstance(output_a, _SwitchingNamespaceTensor)
    assert isinstance(output_b, _SwitchingNamespaceTensor)
    assert output_a.shape == (3, 2)
    assert output_b.shape == (3, 2)
    assert output_a.marker == "a"
    assert output_b.marker == "b"


def test_runner_cache_distinguishes_namespace_bindings_with_same_declared_id() -> None:
    b, c = axes("binding_b", "binding_c")
    op = rearrange(ax[b, c], ax[c, b])
    namespace_a = _SwitchingNamespace("custom.shared", "a")
    namespace_b = _SwitchingNamespace("custom.shared", "b")
    tensor_a = _SwitchingNamespaceTensor((2, 3), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((2, 3), namespace_b)

    output_a = op(tensor_a)
    output_b = op(tensor_b)

    assert isinstance(output_a, _SwitchingNamespaceTensor)
    assert isinstance(output_b, _SwitchingNamespaceTensor)
    assert output_a.marker == "a"
    assert output_b.marker == "b"


def test_reduce_compile_cache_distinguishes_namespaces_with_same_declared_id() -> None:
    (batch_axes,) = packs("reducer_cache_batch")
    (feature,) = axes("reducer_cache_feature")
    op = reduce(ax[batch_axes, feature], ax[batch_axes])
    namespace_a = _SwitchingNamespace("custom.reducer_shared", "a")
    namespace_b = _SwitchingNamespace("custom.reducer_shared", "b")
    tensor_a = _SwitchingNamespaceTensor((2, 3, 4), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((2, 3, 4), namespace_b)
    reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES.clear()
    reduce_build_module._REDUCE_RUNTIME_CACHE_ORDER.clear()

    output_a = op(tensor_a)
    output_b = op(tensor_b)

    assert isinstance(output_a, _SwitchingNamespaceTensor)
    assert isinstance(output_b, _SwitchingNamespaceTensor)
    assert output_a.shape == (2, 3)
    assert output_b.shape == (2, 3)
    assert output_a.marker == "a"
    assert output_b.marker == "b"
    assert len(reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES) == 2


def test_reduce_compile_cache_shares_backend_independent_callable_reducer() -> None:
    (batch_axes,) = packs("callable_cache_batch")
    (feature,) = axes("callable_cache_feature")
    op = reduce(ax[batch_axes, feature], ax[batch_axes]).reduce_by(_namespace_sum)
    namespace_a = _SwitchingNamespace("custom.callable_shared", "a")
    namespace_b = _SwitchingNamespace("custom.callable_shared", "b")
    tensor_a = _SwitchingNamespaceTensor((2, 3, 4), namespace_a)
    tensor_b = _SwitchingNamespaceTensor((2, 3, 4), namespace_b)
    reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES.clear()
    reduce_build_module._REDUCE_RUNTIME_CACHE_ORDER.clear()

    output_a = op(tensor_a)
    output_b = op(tensor_b)

    assert isinstance(output_a, _SwitchingNamespaceTensor)
    assert isinstance(output_b, _SwitchingNamespaceTensor)
    assert output_a.marker == "a"
    assert output_b.marker == "b"
    assert len(reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES) == 1


def test_shape_free_runner_cache_does_not_bypass_view_capability_validation() -> None:
    (b,) = axes("view_backend_b")
    op = view(ax[b], ax[b])
    strict_namespace = _SwitchingNamespace("array_api_compat.numpy", "strict")
    non_view_namespace = _SwitchingNamespace("array_api_compat.jax", "non-view")
    strict_tensor = _SwitchingNamespaceTensor((3,), strict_namespace)
    non_view_tensor = _SwitchingNamespaceTensor((3,), non_view_namespace)

    assert op(strict_tensor) is strict_tensor
    with pytest.raises(ValidationError) as error:
        _ = op(non_view_tensor)

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_shape_free_runner_cache_resolves_backend_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (b,) = axes("backend_resolution_count_b")
    op = rearrange(ax[b], ax[b])
    namespace = _SwitchingNamespace("custom.counted", "counted")
    tensor = _SwitchingNamespaceTensor((3,), namespace)
    original_resolve = BACKEND_RESOLVER.resolve
    resolve_calls = 0

    def counting_resolve(*tensors: object, op_name: str) -> BackendProfile:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(*tensors, op_name=op_name)

    monkeypatch.setattr(BACKEND_RESOLVER, "resolve", counting_resolve)

    assert op(tensor) is tensor
    assert op(tensor) is tensor
    assert resolve_calls == 2


def test_backend_profile_derives_identity_from_namespace() -> None:
    profile = BackendProfile(
        namespace=np,
    )

    assert profile.namespace_id == "numpy"
    assert profile.backend_family == "numpy"
    assert profile.execution_identity.namespace is np
    assert profile.execution_identity.namespace_id == "numpy"
    assert profile.execution_identity.backend_family == "numpy"


def test_backend_profile_excludes_route_capabilities() -> None:
    profile = BackendProfile(namespace=np)

    assert not hasattr(profile, "supports_einsum")
    assert not hasattr(profile, "supports_strict_view")
