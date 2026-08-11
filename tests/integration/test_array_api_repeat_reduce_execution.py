import functools
import inspect
from dataclasses import dataclass
from functools import partial, partialmethod, wraps
from types import MappingProxyType, MethodType
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from einf import (
    ErrorCode,
    ExecutionError,
    ValidationError,
    ax,
    axes,
    packs,
    reduce,
    repeat,
)
from einf.backend import (
    BACKEND_RESOLVER,
    ArrayNamespaceLike,
    BackendArrayOps,
    BackendFamily,
)
from einf.reduction.schema import CanonicalReducer
from einf.steps.expand import step as expand_step_module
from einf.steps.reduce import build as reduce_build_module
from einf.steps.reduce import runtime as reduce_runtime_module
from einf.steps.reduce import step as reduce_step_module

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class _SelectedReducerNamespace:
    __name__ = "custom.selected_reducer"

    def __init__(self) -> None:
        self.sum_calls = 0
        self.asarray_calls = 0

    def asarray(self, value: bool | complex) -> "_SelectedReducerTensor":
        self.asarray_calls += 1
        return _SelectedReducerTensor(np.asarray(value), self)

    def sum(
        self,
        tensor: "_SelectedReducerTensor",
        *,
        axis: tuple[int, ...],
    ) -> int | float | complex:
        self.sum_calls += 1
        return np.asarray(np.sum(tensor.value, axis=axis)).item()


class _NoInventoryReducerNamespace(_SelectedReducerNamespace):
    __name__ = "custom.no_inventory_reducer"

    def __getattribute__(self, name: str) -> object:
        if name in {"all", "any", "max", "mean", "min", "prod"}:
            raise AssertionError(f"unselected reducer {name!r} was inspected")
        return super().__getattribute__(name)


class _BrokenReducerLookupNamespace(_SelectedReducerNamespace):
    __name__ = "custom.broken_reducer_lookup"

    def __getattribute__(self, name: str) -> object:
        if name == "sum":
            raise OSError("selected reducer lookup failed")
        return super().__getattribute__(name)


class _FailingSelectedReducerNamespace(_SelectedReducerNamespace):
    __name__ = "custom.failing_selected_reducer"

    def sum(
        self,
        tensor: "_SelectedReducerTensor",
        *,
        axis: tuple[int, ...],
    ) -> int | float | complex:
        del tensor, axis
        self.sum_calls += 1
        raise OSError("selected reducer invocation failed")


@dataclass(frozen=True, slots=True)
class _SelectedReducerTensor:
    value: np.ndarray
    namespace: _SelectedReducerNamespace

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> _SelectedReducerNamespace:
        del api_version
        return self.namespace

    def __getitem__(self, key: object) -> "_SelectedReducerTensor":
        del key
        return self


class _KnownFamilyReducerNamespace:
    __name__ = "array_api_compat.numpy.custom"

    def __init__(self) -> None:
        self.sum_calls = 0

    def asarray(self, value: bool | complex) -> "_KnownFamilyReducerTensor":
        return _KnownFamilyReducerTensor(np.asarray(value), self)

    def sum(
        self,
        tensor: "_KnownFamilyReducerTensor",
        *,
        axis: tuple[int, ...],
    ) -> "_KnownFamilyReducerTensor":
        self.sum_calls += 1
        return _KnownFamilyReducerTensor(np.sum(tensor.value, axis=axis), self)


class _KnownFamilyReducerTensor:
    def __init__(
        self,
        value: np.ndarray,
        namespace: _KnownFamilyReducerNamespace,
    ) -> None:
        self.value = value
        self.namespace = namespace
        self.method_calls: list[None] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> _KnownFamilyReducerNamespace:
        del api_version
        return self.namespace

    def __getitem__(self, key: object) -> "_KnownFamilyReducerTensor":
        return _KnownFamilyReducerTensor(self.value[key], self.namespace)

    def sum(self, *, axis: tuple[int, ...]) -> "_KnownFamilyReducerTensor":
        del axis
        self.method_calls.append(None)
        wrong = np.full((self.shape[0],), -100)
        return _KnownFamilyReducerTensor(wrong, self.namespace)


def _explode_native_contract_einsum(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native contract einsum should not be called in this path")


def test_inflate_rejects_multi_input_lhs_with_diagnostic_code() -> None:
    (b,) = axes("b")

    with pytest.raises(ValidationError) as error:
        _ = repeat.__call__((ax[b], ax[b]), ax[b])

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value


def test_reduce_requires_only_scalar_coercion_and_selected_reducer() -> None:
    b, h = axes("selected_b", "selected_h")
    namespace = _NoInventoryReducerNamespace()
    tensor = _SelectedReducerTensor(np.arange(2 * 3).reshape(2, 3), namespace)

    result = reduce(ax[b, h], ax[()]).reduce_by("sum")(tensor)

    assert isinstance(result, _SelectedReducerTensor)
    assert result.shape == ()
    assert result.value.item() == 15
    assert namespace.sum_calls == 1
    assert namespace.asarray_calls == 1


def test_reduce_reports_missing_selected_reducer() -> None:
    b, h = axes("missing_b", "missing_h")
    namespace = _SelectedReducerNamespace()
    tensor = _SelectedReducerTensor(np.arange(2 * 3).reshape(2, 3), namespace)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by("prod")(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "backend reducer 'prod' is unavailable" in error.value.message


def test_reduce_projects_selected_reducer_lookup_failure() -> None:
    b, h = axes("lookup_b", "lookup_h")
    namespace = _BrokenReducerLookupNamespace()
    tensor = _SelectedReducerTensor(np.arange(2 * 3).reshape(2, 3), namespace)

    with pytest.raises(ExecutionError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by("sum")(tensor)

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value
    assert "selected reducer lookup failed" in error.value.message


def test_reduce_projects_selected_reducer_invocation_failure() -> None:
    b, h = axes("invocation_b", "invocation_h")
    namespace = _FailingSelectedReducerNamespace()
    tensor = _SelectedReducerTensor(np.arange(2 * 3).reshape(2, 3), namespace)

    with pytest.raises(ExecutionError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by("sum")(tensor)

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value
    assert "selected reducer invocation failed" in error.value.message
    assert namespace.sum_calls == 1


def test_reduce_known_family_custom_tensor_uses_selected_namespace_reducer() -> None:
    b, h = axes("known_family_b", "known_family_h")
    namespace = _KnownFamilyReducerNamespace()
    tensor = _KnownFamilyReducerTensor(
        np.arange(2 * 3).reshape(2, 3),
        namespace,
    )

    result = reduce(ax[b, h], ax[b]).reduce_by("sum")(tensor)

    assert isinstance(result, _KnownFamilyReducerTensor)
    np.testing.assert_array_equal(result.value, np.array([3, 12]))
    assert namespace.sum_calls == 1
    assert tensor.method_calls == []


def test_dynamic_reduce_known_family_tensor_uses_selected_namespace_reducer() -> None:
    (batch_axes,) = packs("known_family_dynamic_batch")
    (feature,) = axes("known_family_dynamic_feature")
    namespace = _KnownFamilyReducerNamespace()
    tensor = _KnownFamilyReducerTensor(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        namespace,
    )

    result = reduce(ax[batch_axes, feature], ax[batch_axes]).reduce_by("sum")(tensor)

    assert isinstance(result, _KnownFamilyReducerTensor)
    np.testing.assert_array_equal(result.value, np.sum(tensor.value, axis=2))
    assert namespace.sum_calls == 1
    assert tensor.method_calls == []


def test_reduce_exact_numpy_tensor_keeps_native_method_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h = axes("native_route_b", "native_route_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    profile = BACKEND_RESOLVER.resolve(tensor, op_name="reduce")
    namespace_calls: list[None] = []

    def wrong_namespace_sum(*_args: object, **_kwargs: object) -> np.ndarray:
        namespace_calls.append(None)
        return np.full((2,), -100)

    monkeypatch.setattr(profile.namespace, "sum", wrong_namespace_sum)

    result = reduce(ax[b, h], ax[b]).reduce_by("sum")(tensor)

    np.testing.assert_array_equal(result, np.array([3, 12]))
    assert namespace_calls == []


def test_inflate_appends_axis_and_broadcasts_values() -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_repeat_appends_axis_and_broadcasts_values_with_torch() -> None:
    assert torch is not None
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = torch.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, tensor.unsqueeze(2).expand(2, 3, 4))


def test_repeat_uses_symbolic_fastpath_without_reindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _explode_native_contract_einsum,
    )

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


def test_inflate_rank_zero_input_to_vector() -> None:
    op = repeat(ax[()], ax[3])

    tensor = np.asarray(7)
    result = op(tensor)

    expected = np.asarray([7, 7, 7])
    np.testing.assert_array_equal(result, expected)


def test_inflate_evaluable_expression_repeat_treats_expr_as_atomic() -> None:
    a, b, c = axes("a", "b", "c")
    op = repeat(ax[(a + a)], ax[b, (a + (a * c)), (c * c)]).with_sizes(a=2, b=1, c=2)

    result = op(np.zeros((4,)))

    assert result.shape == (1, 6, 4)


def test_inflate_singleton_axes_resolve_with_labeled_axis_preference() -> None:
    (n,) = axes("n")
    op = repeat(ax[n], ax[1, n, 1]).with_sizes(n=1)

    result = op(np.arange(1))

    expected = np.array([[[0]]])
    np.testing.assert_array_equal(result, expected)


def test_inflate_labeled_preference_falls_back_when_only_nonpreserving_path_is_viable() -> (
    None
):
    (a,) = axes("a")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a=2)

    result = op(np.array([5, 7, 9]))

    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_repeat_uses_solver_fast_path_for_expression_lhs_with_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (a,) = axes("a_solver_ws")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a_solver_ws=2)
    called = {"value": False}
    original_solve_expand_program_from_input_shape = (
        expand_step_module.solve_expand_program_from_input_shape
    )

    def _spy_solve_expand_program_from_input_shape(*args, **kwargs):
        called["value"] = True
        return original_solve_expand_program_from_input_shape(*args, **kwargs)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _spy_solve_expand_program_from_input_shape,
    )

    result = op(np.array([5, 7, 9]))
    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert called["value"]


def test_repeat_uses_solver_fast_path_without_explicit_sizes_when_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (a,) = axes("a_solver_wo")
    op = repeat(ax[(1 + a)], ax[(1 + a), a])
    called = {"value": False}
    original_solve_expand_program_from_input_shape = (
        expand_step_module.solve_expand_program_from_input_shape
    )

    def _spy_solve_expand_program_from_input_shape(*args, **kwargs):
        called["value"] = True
        return original_solve_expand_program_from_input_shape(*args, **kwargs)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _spy_solve_expand_program_from_input_shape,
    )

    result = op(np.array([5, 7, 9]))
    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert called["value"]


def test_repeat_expression_solver_reports_ambiguity_when_non_unique() -> None:
    a, b = axes("a", "b")
    op = repeat(ax[(a + b)], ax[(a + b), a])

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_repeat_missing_explicit_axis_reports_validation_error() -> None:
    a, b = axes("a", "b")
    op = repeat(ax[a], ax[a, b])

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_repeat_expression_solver_reports_inconsistent_when_constraints_conflict() -> (
    None
):
    (a,) = axes("a")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a=5)

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_default_sum_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_default_sum_uses_compiled_phase_without_reindex() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_default_sum_uses_static_runner_without_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    def explode_context_normalization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reduce runtime context normalization should be called")

    monkeypatch.setattr(
        reduce_step_module,
        "build_runtime_execution_context",
        explode_context_normalization,
    )

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)
    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_reuses_cached_compiled_runtime_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(op(tensor), expected)
    monkeypatch.setattr(
        reduce_build_module,
        "_compile_reduce_runtime_phase",
        _explode_native_contract_einsum,
    )
    np.testing.assert_array_equal(op(tensor), expected)


def test_dynamic_reduce_binds_backend_capabilities_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (batch_axes,) = packs("binding_batch")
    (feature,) = axes("binding_feature")
    op = reduce(ax[batch_axes, feature], ax[batch_axes])
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    capability_calls = {"namespace": 0, "adapter": 0}
    original_bind = reduce_step_module.bind_reducer_namespace
    original_get_backend_ops = reduce_step_module.get_backend_array_ops

    def count_bind(
        namespace: ArrayNamespaceLike,
    ) -> reduce_runtime_module.ReducerArrayNamespace:
        capability_calls["namespace"] += 1
        return original_bind(namespace)

    def count_backend_ops(
        backend_family: BackendFamily | None,
    ) -> BackendArrayOps | None:
        capability_calls["adapter"] += 1
        return original_get_backend_ops(backend_family)

    monkeypatch.setattr(
        reduce_step_module,
        "bind_reducer_namespace",
        count_bind,
    )
    monkeypatch.setattr(
        reduce_build_module,
        "bind_reducer_namespace",
        count_bind,
        raising=False,
    )
    monkeypatch.setattr(
        reduce_step_module,
        "get_backend_array_ops",
        count_backend_ops,
    )
    monkeypatch.setattr(
        reduce_build_module,
        "get_backend_array_ops",
        count_backend_ops,
        raising=False,
    )

    expected = np.sum(tensor, axis=2)
    for _ in range(3):
        np.testing.assert_array_equal(op(tensor), expected)

    assert capability_calls == {"namespace": 1, "adapter": 1}


def test_reduce_compile_invariant_rejects_mismatched_output_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[d, b]).reduce_by(
        lambda x, *, axis: np.sum(x, axis=axis)
    )
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES.clear()
    reduce_build_module._REDUCE_RUNTIME_CACHE_ORDER.clear()
    compile_reduce_runtime_phase = reduce_build_module._compile_reduce_runtime_phase

    def compile_with_wrong_output_terms(
        *,
        lhs_terms: reduce_build_module.ScalarAxisTerms,
        reduce_axes: reduce_build_module.AxisTerms,
        reducer: CanonicalReducer,
        pack_sizes: dict[str, tuple[int, ...]],
        axis_sizes: dict[str, int],
        xp: reduce_runtime_module.ReducerArrayNamespace,
    ) -> tuple[
        tuple[int, ...],
        reduce_build_module.CompiledReducer,
        reduce_build_module.ScalarAxisTerms,
    ]:
        axes, compiled_reducer, output_terms = compile_reduce_runtime_phase(
            lhs_terms=lhs_terms,
            reduce_axes=reduce_axes,
            reducer=reducer,
            pack_sizes=pack_sizes,
            axis_sizes=axis_sizes,
            xp=xp,
        )
        wrong_terms = reduce_build_module.ScalarAxisTerms(tuple(reversed(output_terms)))
        return axes, compiled_reducer, wrong_terms

    monkeypatch.setattr(
        reduce_build_module,
        "_compile_reduce_runtime_phase",
        compile_with_wrong_output_terms,
    )

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "reduce lowering invariant violated" in error.value.message


def test_reduce_duplicate_axis_partial_multiplicity_is_ambiguous() -> None:
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[b])

    tensor = np.arange(12).reshape(2, 2, 3)
    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_reduce_duplicate_axis_full_reduction_executes() -> None:
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[()])

    tensor = np.arange(12).reshape(2, 2, 3)
    result = op(tensor)

    expected = np.sum(tensor, axis=(0, 1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_reorders_remaining_axes_to_rhs_order() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[d, b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.transpose(np.sum(tensor, axis=1), (1, 0))
    np.testing.assert_array_equal(result, expected)


def test_reduce_ordered_phase_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "prod"))

    tensor = np.arange(1, (2 * 3 * 4) + 1).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.prod(np.sum(tensor, axis=1), axis=1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_accepts_one_axis() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b, d]).reduce_by("prod")
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    torch.testing.assert_close(result, tensor.prod(dim=1))


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_accepts_multiple_axes() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by("prod")
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    expected = tensor.prod(dim=2).prod(dim=1)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_preserves_nonconsecutive_axis_positions() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[h]).reduce_by((ax[b, d], "prod"))
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    expected = tensor.prod(dim=2).prod(dim=0)
    torch.testing.assert_close(result, expected)


def test_reduce_ordered_phase_preserves_declared_axis_order() -> None:
    b, h, d = axes("b", "h", "d")
    seen_axes: list[tuple[int, ...]] = []

    def reducer(tensor: NDArray[np.int64], *, axis: tuple[int, ...]):
        seen_axes.append(axis)
        return np.sum(tensor, axis=axis)

    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[d, h], reducer))

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    assert seen_axes == [(2, 1)]
    expected = np.sum(tensor, axis=(2, 1))
    np.testing.assert_array_equal(result, expected)


def test_reduce_callable_reducer_executes() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(tensor: NDArray[np.int64], axis: tuple[int, ...]):
        return np.max(tensor, axis=axis)

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.max(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_builtin_numpy_sum_callable_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by(np.sum)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_quantile_partial_preserves_open_tensor_parameter() -> None:
    b, h, d = axes("quantile_partial_b", "quantile_partial_h", "quantile_partial_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    reducer = partial(np.quantile, q=0.5)

    result = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)(tensor)

    np.testing.assert_array_equal(result, np.quantile(tensor, q=0.5, axis=(1, 2)))


def test_reduce_opaque_or_variadic_signatures_are_rejected_before_invocation() -> None:
    calls: list[str] = []

    def invalid_signature(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("invalid")
        return np.sum(values, axis=reducer_axes)

    invalid_signature.__signature__ = "invalid"  # type: ignore[attr-defined]

    def variadic(*args: object, **kwargs: object) -> NDArray[np.int64]:
        calls.append("variadic")
        values, reducer_axes = args
        assert isinstance(values, np.ndarray)
        assert isinstance(reducer_axes, tuple)
        return np.sum(values, axis=reducer_axes)

    class BrokenSignature:
        @property
        def __signature__(self) -> object:
            raise OSError("metadata unavailable")

        def __call__(self, values: NDArray[np.int64]) -> NDArray[np.int64]:
            calls.append("broken")
            return values

    b, h = axes("opaque_b", "opaque_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    for reducer in (invalid_signature, variadic, BrokenSignature()):
        with pytest.raises(ValidationError) as error:
            _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

        assert "callable reducer is ambiguous" in str(error.value)

    assert calls == []


def test_reduce_explicit_wrapper_remediates_opaque_callable() -> None:
    class OpaqueReducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("signature unavailable")

        def __call__(
            self,
            values: NDArray[np.int64],
            reducer_axes: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=reducer_axes)

    opaque = OpaqueReducer()

    def reducer(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return opaque(values, axis)

    b, h = axes("explicit_wrapper_b", "explicit_wrapper_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_positional_axes_reducer_executes() -> None:
    seen_axes: list[tuple[int, ...]] = []

    def reducer(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        seen_axes.append(reducer_axes)
        return np.sum(values, axis=reducer_axes)

    b, h, d = axes("positional_b", "positional_h", "positional_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    result = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)(tensor)

    assert seen_axes == [(1, 2)]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=(1, 2)))


def test_reduce_regular_axes_default_is_not_preconfigured() -> None:
    seen_axes: list[tuple[int, ...]] = []

    def reducer(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...] = (0,),
    ) -> NDArray[np.int64]:
        seen_axes.append(axis)
        return np.sum(values, axis=axis)

    b, h, d = axes("default_b", "default_h", "default_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    result = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)(tensor)

    assert seen_axes == [(1, 2)]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=(1, 2)))


@pytest.mark.parametrize("variadic_kind", ["args", "kwargs"])
def test_reduce_variadic_parameters_are_not_axes_authority(
    variadic_kind: str,
) -> None:
    if variadic_kind == "args":

        def reducer(values: NDArray[np.int64], *axes: object) -> np.int64:
            assert axes == ()
            return np.sum(values)

    else:

        def reducer(  # type: ignore[no-redef]
            values: NDArray[np.int64],
            **axis: object,
        ) -> np.int64:
            assert axis == {}
            return np.sum(values)

    b, h = axes(f"variadic_{variadic_kind}_b", f"variadic_{variadic_kind}_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[()]).reduce_by(reducer)(tensor)

    assert np.asarray(result).item() == np.sum(tensor)


def test_reduce_single_parameter_named_axis_is_tensor_only() -> None:
    def reducer(axis: NDArray[np.int64]) -> np.int64:
        return np.sum(axis)

    b, h = axes("axis_tensor_b", "axis_tensor_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[()]).reduce_by(reducer)(tensor)

    assert np.asarray(result).item() == np.sum(tensor)


def test_reduce_accurate_wrapped_signature_is_effective_contract() -> None:
    calls: list[tuple[object, ...]] = []

    def positional_reducer(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return np.sum(values, axis=reducer_axes)

    @wraps(positional_reducer)
    def forwarding(*args: object, **kwargs: object) -> NDArray[np.int64]:
        calls.append(args)
        return positional_reducer(*args, **kwargs)  # type: ignore[arg-type]

    b, h = axes("wrapped_b", "wrapped_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(forwarding)(tensor)

    assert calls == [(tensor, (1,))]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_scalar_partial_configuration_preserves_tensor_slot() -> None:
    seen_scales: list[int] = []

    def scaled_sum(
        scale: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        seen_scales.append(scale)
        return np.sum(values, axis=axis) * scale

    b, h = axes("partial_config_b", "partial_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    reducer = partial(scaled_sum, 2)

    result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    assert seen_scales == [2]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_tensor_only_scalar_partial_preserves_tensor_slot() -> None:
    def scaled_total(scale: int, values: NDArray[np.int64]) -> np.int64:
        return np.sum(values) * scale

    b, h = axes("tensor_only_partial_b", "tensor_only_partial_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    reducer = partial(scaled_total, 2)

    result = reduce(ax[b, h], ax[()]).reduce_by(reducer)(tensor)

    assert np.asarray(result).item() == np.sum(tensor) * 2


def test_reduce_required_scalar_keyword_partial_preserves_tensor_slot() -> None:
    def scaled_sum(
        values: NDArray[np.int64],
        scale: int,
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return np.sum(values, axis=axis) * scale

    b, h = axes("keyword_partial_b", "keyword_partial_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    reducer = partial(scaled_sum, scale=2)

    result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_partial_configuration_is_captured_at_plan_ingress() -> None:
    b, h = axes("partial_snapshot_b", "partial_snapshot_h")
    tensor = np.arange(2 * 2).reshape(2, 2)
    reducer = partial(np.sum, axis=(1,))
    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)

    assert reducer.keywords is not None
    reducer.keywords["axis"] = (0,)
    result = op(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_partial_subclass_is_rejected_at_plan_ingress() -> None:
    calls: list[str] = []

    class ReducerPartial(functools.partial):
        def __call__(self, *args: object, **kwargs: object) -> object:
            calls.append("called")
            return super().__call__(*args, **kwargs)

    b, h = axes("partial_subclass_b", "partial_subclass_h")
    reducer = ReducerPartial(np.sum, axis=(1,))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)

    assert calls == []
    assert "partial subclasses are unsupported" in str(error.value)


def test_reduce_nested_partial_subclass_uses_observable_partial_provenance() -> None:
    calls: list[str] = []

    class ReducerPartial(functools.partial):
        def __call__(self, *args: object, **kwargs: object) -> object:
            calls.append("called")
            return super().__call__(*args, **kwargs)

    b, h = axes("nested_partial_subclass_b", "nested_partial_subclass_h")
    inner = ReducerPartial(np.max, axis=(1,))
    reducer = partial(inner)
    tensor = np.arange(2 * 3).reshape(2, 3)

    if reducer.func is inner:
        with pytest.raises(ValidationError) as error:
            _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)

        assert calls == []
        assert "partial subclasses are unsupported" in str(error.value)
        return

    expected = reducer(tensor)
    default_result = np.sum(tensor, axis=1)
    result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    assert reducer.func is np.max
    assert calls == []
    assert not np.array_equal(expected, default_result)
    np.testing.assert_array_equal(result, expected)


def test_reduce_partial_keyword_option_after_tensor_is_supported() -> None:
    seen_modes: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        mode: str = "plain",
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        seen_modes.append(mode)
        return np.sum(values, axis=axis)

    b, h = axes("partial_option_b", "partial_option_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, mode="checked")

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert seen_modes == ["checked"]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_tensor_like_partial_value_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: NDArray[np.int64],
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values + configuration, axis=axis)

    b, h = axes("tensor_config_b", "tensor_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, np.ones_like(tensor))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_nested_tensor_config_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: tuple[NDArray[np.int64]],
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(configuration[0] + values * 0, axis=axis)

    b, h = axes("nested_tensor_config_b", "nested_tensor_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    stale = np.full_like(tensor, 100)
    configured = partial(reducer, (stale,))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_namedtuple_tensor_config_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    class Configuration(NamedTuple):
        source: NDArray[np.int64]

    def reducer(
        configuration: Configuration,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(configuration.source, axis=axis)

    b, h = axes("namedtuple_tensor_config_b", "namedtuple_tensor_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    stale = np.full_like(tensor, 100)
    configured = partial(reducer, Configuration(stale))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_frozenset_subclass_tensor_config_is_rejected_before_invocation() -> (
    None
):
    calls: list[str] = []

    class TensorConfiguration:
        shape = (1,)

        def __getitem__(self, index: int) -> int:
            return index

    class Configuration(frozenset[object]):
        pass

    def reducer(
        configuration: Configuration,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    b, h = axes("frozenset_tensor_config_b", "frozenset_tensor_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, Configuration({TensorConfiguration()}))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_mutable_pre_axes_configuration_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: list[object],
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    configuration: list[object] = []
    configuration.append(configuration)
    configured = partial(reducer, configuration)
    b, h = axes("cyclic_config_b", "cyclic_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "mutable or live container" in str(error.value)


def test_reduce_nested_mutable_pre_axes_configuration_is_rejected() -> None:
    calls: list[str] = []

    def reducer(
        configuration: tuple[list[int]],
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis) * configuration[0][0]

    configured = partial(reducer, ([2],))
    b, h = axes("nested_mutable_config_b", "nested_mutable_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "mutable or live container" in str(error.value)


def test_reduce_mapping_proxy_configuration_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: object,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    backing = {"scale": 2}
    configured = partial(reducer, MappingProxyType(backing))
    b, h = axes("mapping_proxy_config_b", "mapping_proxy_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "live container" in str(error.value)


def test_reduce_mapping_view_configuration_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: object,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    backing = {"scale": 2}
    configured = partial(reducer, backing.values())
    b, h = axes("mapping_view_config_b", "mapping_view_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "live container" in str(error.value)


def test_reduce_post_axes_tensor_like_partial_configuration_is_supported() -> None:
    b, h, d = axes("partial_where_b", "partial_where_h", "partial_where_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    where = (tensor % 2) == 0
    reducer = partial(np.sum, where=where)

    result = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)(tensor)

    np.testing.assert_array_equal(
        result,
        np.sum(tensor, axis=(1, 2), where=where),
    )


def test_reduce_post_axes_nested_tensor_like_configuration_is_supported() -> None:
    def reducer(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
        configuration: tuple[NDArray[np.bool_]],
    ) -> NDArray[np.int64]:
        return np.sum(values, axis=reducer_axes, where=configuration[0])

    b, h = axes("nested_where_b", "nested_where_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    where = (tensor % 2) == 0
    configured = partial(reducer, configuration=(where,))

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(
        result,
        np.sum(tensor, axis=1, where=where),
    )


def test_reduce_post_axes_mutable_configuration_remains_external_state() -> None:
    def reducer(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
        configuration: list[int],
    ) -> NDArray[np.int64]:
        return np.sum(values, axis=reducer_axes) * configuration[0]

    b, h = axes("mutable_post_axes_b", "mutable_post_axes_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configuration = [2]
    op = reduce(ax[b, h], ax[b]).reduce_by(
        partial(reducer, configuration=configuration)
    )

    first = op(tensor)
    configuration[0] = 3
    second = op(tensor)

    np.testing.assert_array_equal(first, np.sum(tensor, axis=1) * 2)
    np.testing.assert_array_equal(second, np.sum(tensor, axis=1) * 3)


def test_reduce_scalar_var_keyword_configuration_is_supported() -> None:
    def reducer(
        values: NDArray[np.int64],
        **configuration: int,
    ) -> np.int64:
        return np.sum(values) * configuration.get("scale", 1)

    b, h = axes("var_keyword_config_b", "var_keyword_config_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, scale=2)

    result = reduce(ax[b, h], ax[()]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(result, np.asarray(np.sum(tensor) * 2))


def test_reduce_var_keyword_tensor_config_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        **configuration: object,
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=1)

    b, h = axes("var_keyword_tensor_b", "var_keyword_tensor_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, source=np.ones_like(tensor))

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_var_keyword_mutable_config_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        **configuration: object,
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=1) * len(configuration)

    b, h = axes("var_keyword_mutable_b", "var_keyword_mutable_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, options=[1])

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "mutable or live container" in str(error.value)


def test_reduce_partial_bound_tensor_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        configuration: str,
        data: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append(configuration)
        return np.sum(data, axis=axis)

    b, h = axes("bound_tensor_b", "bound_tensor_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, "configured", data=tensor)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_partial_bound_axes_selects_tensor_only_call() -> None:
    seen_axes: list[tuple[int, ...]] = []

    def reducer(
        values: NDArray[np.int64],
        reducer_axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        seen_axes.append(reducer_axes)
        return np.sum(values, axis=reducer_axes)

    b, h, d = axes("bound_axes_b", "bound_axes_h", "bound_axes_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    configured = partial(reducer, reducer_axes=(1, 2))

    result = reduce(ax[b, h, d], ax[b]).reduce_by(configured)(tensor)

    assert seen_axes == [(1, 2)]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=(1, 2)))


@pytest.mark.parametrize("configured_axes", [(0, 2), [1, 2]])
def test_reduce_partial_bound_axes_must_be_exact(
    configured_axes: object,
) -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    b, h, d = axes("mismatch_b", "mismatch_h", "mismatch_d")
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    configured = partial(reducer, axis=configured_axes)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h, d], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "configures different reduction axes" in str(error.value)


def test_reduce_bound_method_uses_effective_signature() -> None:
    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis)

    b, h = axes("method_b", "method_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(Reducers().total)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_callable_object_uses_effective_signature() -> None:
    class Reducer:
        def __call__(
            self,
            values: NDArray[np.int64],
            reducer_axes: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=reducer_axes)

    b, h = axes("callable_b", "callable_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(Reducer())(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_static_and_class_callable_descriptors_use_bound_signature() -> None:
    class StaticReducer:
        @staticmethod
        def __call__(
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis)

    class ClassReducer:
        @classmethod
        def __call__(
            cls,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis)

    b, h = axes("call_descriptor_b", "call_descriptor_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    for reducer in (StaticReducer(), ClassReducer()):
        result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

        np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_partial_callable_descriptor_preserves_tensor_slot() -> None:
    class Reducer:
        @staticmethod
        def __call__(
            values: NDArray[np.int64],
            scale: int = 1,
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis) * scale

    b, h = axes("partial_call_descriptor_b", "partial_call_descriptor_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(Reducer(), scale=2)

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_bound_partialmethod_uses_effective_signature() -> None:
    class Reducers:
        def scaled_sum(
            self,
            scale: int,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis) * scale

        double_sum = partialmethod(scaled_sum, 2)

    b, h = axes("partialmethod_b", "partialmethod_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(Reducers().double_sum)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_partialmethod_requires_descriptor_binding() -> None:
    calls: list[str] = []

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values, axis=axis)

        forwarding = partialmethod(total)

    reducers = Reducers()
    configured = partial(Reducers.forwarding, reducers)
    b, h = axes("bound_partialmethod_b", "bound_partialmethod_h")

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)

    assert calls == []
    assert "partialmethod reducer must use descriptor binding" in str(error.value)


def test_reduce_unbound_partialmethod_is_rejected_before_invocation() -> None:
    calls: list[str] = []
    stale = np.full((2, 3), 100)

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values, axis=axis)

        stale_total = partialmethod(total, stale)

    b, h = axes("unbound_partialmethod_b", "unbound_partialmethod_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(Reducers.stale_total)(tensor)

    assert calls == []
    assert "partialmethod reducer must use descriptor binding" in str(error.value)


def test_reduce_bound_partialmethod_cannot_capture_stale_tensor() -> None:
    calls: list[str] = []
    stale = np.full((2, 3), 100)

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values, axis=axis)

        stale_total = partialmethod(total, stale)

    b, h = axes("bound_stale_partialmethod_b", "bound_stale_partialmethod_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(Reducers().stale_total)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_receiver_partial_cannot_hide_partialmethod_configuration() -> None:
    calls: list[str] = []
    stale = np.full((2, 3), 100)

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            bias: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values + bias, axis=axis)

        stale_total = partialmethod(total, bias=stale)

    b, h = axes("receiver_partialmethod_b", "receiver_partialmethod_h")
    configured = partial(Reducers.stale_total, Reducers())

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)

    assert calls == []
    assert "partialmethod reducer must use descriptor binding" in str(error.value)


@pytest.mark.parametrize("binding_style", ["method_type", "descriptor_get"])
def test_reduce_manual_method_binding_cannot_hide_partialmethod_configuration(
    binding_style: str,
) -> None:
    calls: list[str] = []
    stale = np.full((2, 3), 100)

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            bias: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values + bias, axis=axis)

        stale_total = partialmethod(total, bias=stale)

    reducers = Reducers()
    if binding_style == "method_type":
        configured = MethodType(Reducers.stale_total, reducers)
    else:
        configured = Reducers.stale_total.__get__(reducers, Reducers)
    b, h = axes(f"manual_{binding_style}_b", f"manual_{binding_style}_h")

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)

    assert calls == []
    assert "partialmethod" in str(error.value)


def test_reduce_nested_bound_partialmethod_cannot_hide_stale_tensor() -> None:
    calls: list[str] = []
    stale = np.full((2, 3), 100)

    class Reducers:
        def total(
            self,
            values: NDArray[np.int64],
            bias: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values + bias, axis=axis)

        biased_total = partialmethod(total, bias=stale)

    b, h = axes("nested_stale_partialmethod_b", "nested_stale_partialmethod_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    reducer = partial(Reducers().biased_total)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    assert calls == []
    assert "tensor-like value" in str(error.value)


def test_reduce_nested_partial_configuration_is_captured_recursively() -> None:
    class Reducers:
        def scaled_sum(
            self,
            values: NDArray[np.int64],
            *,
            scale: int,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            return np.sum(values, axis=axis) * scale

        double_sum = partialmethod(scaled_sum, scale=2)

    b, h = axes("nested_snapshot_b", "nested_snapshot_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    inner = Reducers().double_sum
    reducer = partial(inner)
    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)

    assert inner.keywords is not None
    inner.keywords["scale"] = 3
    result = op(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_partial_uses_its_visible_signature_metadata() -> None:
    calls: list[str] = []

    def visible(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        raise AssertionError("signature authority is not the runtime target")

    def forwarding(*args: object, **kwargs: object) -> NDArray[np.int64]:
        calls.append("called")
        values = args[0]
        assert isinstance(values, np.ndarray)
        axis = kwargs["axis"]
        assert isinstance(axis, tuple)
        return np.sum(values, axis=axis)

    b, h = axes("partial_visible_signature_b", "partial_visible_signature_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    explicit_signature = partial(forwarding)
    explicit_signature.__signature__ = inspect.signature(visible)
    wrapped_signature = partial(forwarding)
    wrapped_signature.__wrapped__ = visible

    for reducer in (explicit_signature, wrapped_signature):
        result = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)
        np.testing.assert_array_equal(result, np.sum(tensor, axis=1))

    assert calls == ["called", "called"]


@pytest.mark.parametrize("metadata_kind", ["wrapped", "signature"])
@pytest.mark.parametrize("metadata_layer", ["configured", "inner"])
def test_reduce_configured_partial_chain_cannot_own_signature_metadata(
    metadata_kind: str,
    metadata_layer: str,
) -> None:
    calls: list[str] = []

    def visible(values: NDArray[np.int64]) -> NDArray[np.int64]:
        raise AssertionError("signature authority is not the runtime target")

    def forwarding(*args: object, **kwargs: object) -> NDArray[np.int64]:
        calls.append("called")
        values = args[0]
        assert isinstance(values, np.ndarray)
        axis = kwargs["axis"]
        assert isinstance(axis, tuple)
        return np.sum(values, axis=axis)

    if metadata_layer == "inner":
        metadata_owner = partial(forwarding)
        if metadata_kind == "wrapped":
            functools.update_wrapper(metadata_owner, visible)
        else:
            metadata_owner.__signature__ = inspect.signature(visible)
        configured = partial(metadata_owner, axis=(1,))
    else:
        configured = partial(forwarding, axis=(1,))
        metadata_owner = configured
        if metadata_kind == "wrapped":
            functools.update_wrapper(metadata_owner, visible)
        else:
            metadata_owner.__signature__ = inspect.signature(visible)
    b, h = axes("configured_metadata_b", "configured_metadata_h")

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[h]).reduce_by(configured)

    assert calls == []
    assert "configured partial signature metadata" in str(error.value)


def test_reduce_configured_partial_uses_target_signature_metadata() -> None:
    def visible(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        raise AssertionError("signature authority is not the runtime target")

    def forwarding(*args: object, **kwargs: object) -> NDArray[np.int64]:
        values = args[0]
        assert isinstance(values, np.ndarray)
        axis = kwargs["axis"]
        assert isinstance(axis, tuple)
        return np.sum(values, axis=axis)

    functools.update_wrapper(forwarding, visible)
    configured = partial(forwarding, axis=(1,))
    b, h = axes("target_metadata_b", "target_metadata_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_target_signature_metadata_rejects_configured_axes_mismatch() -> None:
    calls: list[str] = []

    def visible(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        raise AssertionError("signature authority is not the runtime target")

    def forwarding(*args: object, **kwargs: object) -> NDArray[np.int64]:
        calls.append("called")
        values = args[0]
        assert isinstance(values, np.ndarray)
        axis = kwargs["axis"]
        assert isinstance(axis, tuple)
        return np.sum(values, axis=axis)

    functools.update_wrapper(forwarding, visible)
    configured = partial(forwarding, axis=(0,))
    b, h = axes("target_metadata_mismatch_b", "target_metadata_mismatch_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "configures different reduction axes" in str(error.value)


def test_reduce_invalid_partial_signature_is_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis)

    b, h = axes("invalid_partial_signature_b", "invalid_partial_signature_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer)
    configured.__signature__ = "invalid"

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "callable reducer is ambiguous" in str(error.value)


def test_reduce_update_wrapper_callable_uses_visible_signature() -> None:
    def target(
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return np.sum(values, axis=axis)

    class Reducer:
        def __init__(self) -> None:
            functools.update_wrapper(self, target)

        def __call__(self, *args: object, **kwargs: object) -> NDArray[np.int64]:
            return target(*args, **kwargs)  # type: ignore[arg-type]

    b, h = axes("update_wrapper_callable_b", "update_wrapper_callable_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(Reducer())(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1))


def test_reduce_singledispatchmethod_requires_explicit_wrapper() -> None:
    calls: list[str] = []

    class Reducers:
        @functools.singledispatchmethod
        def total(
            self,
            values: NDArray[np.int64],
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            calls.append("called")
            return np.sum(values, axis=axis)

    b, h = axes("singledispatchmethod_b", "singledispatchmethod_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(Reducers().total)(tensor)

    assert calls == []
    assert "reducer signature is unsupported" in str(error.value)


def test_reduce_wrapped_scalar_partial_preserves_effective_signature() -> None:
    seen_scales: list[int] = []

    def target(
        scale: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        seen_scales.append(scale)
        return np.sum(values, axis=axis) * scale

    @wraps(target)
    def forwarding(scale: int, *args: object, **kwargs: object) -> NDArray[np.int64]:
        return target(scale, *args, **kwargs)  # type: ignore[arg-type]

    b, h = axes("wrapped_partial_b", "wrapped_partial_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(forwarding, 2)

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert seen_scales == [2]
    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_multiple_axes_authorities_are_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        axis: tuple[int, ...],
        axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis) + np.sum(values, axis=axes)

    b, h = axes("multiple_axes_b", "multiple_axes_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    assert calls == []
    assert "multiple axes authorities" in str(error.value)


def test_reduce_unsupported_keyword_only_axes_are_rejected_before_invocation() -> None:
    calls: list[str] = []

    def reducer(
        values: NDArray[np.int64],
        *,
        reducer_axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=reducer_axes)

    b, h = axes("keyword_axes_b", "keyword_axes_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(reducer)(tensor)

    assert calls == []
    assert "reducer signature is unsupported" in str(error.value)


def test_reduce_placeholder_partial_is_rejected_before_invocation() -> None:
    placeholder = getattr(functools, "Placeholder", None)
    if placeholder is None:
        pytest.skip("functools.Placeholder requires Python 3.14")

    calls: list[str] = []

    def reducer(
        configuration: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis) * configuration

    b, h = axes("placeholder_b", "placeholder_h")
    tensor = np.arange(2 * 3).reshape(2, 3)
    configured = partial(reducer, placeholder, 2)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "Placeholder" in str(error.value)


def test_reduce_nested_placeholder_partial_is_rejected_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placeholder = object()
    monkeypatch.setattr(functools, "Placeholder", placeholder, raising=False)
    calls: list[str] = []

    def reducer(
        configuration: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        calls.append("called")
        return np.sum(values, axis=axis) * configuration

    inner = partial(reducer, placeholder, 2)
    inner.marker = "preserve nested partial"
    configured = partial(inner)
    b, h = axes("nested_placeholder_b", "nested_placeholder_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    with pytest.raises(ValidationError) as error:
        _ = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    assert calls == []
    assert "Placeholder" in str(error.value)


def test_reduce_resolved_nested_placeholder_partial_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placeholder = object()
    monkeypatch.setattr(functools, "Placeholder", placeholder, raising=False)

    def reducer(
        configuration: int,
        offset: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return np.sum(values + offset, axis=axis) * configuration

    inner = partial(reducer, placeholder, 0)
    inner.marker = "preserve nested partial"
    configured = partial(inner, 2)
    b, h = axes("resolved_placeholder_b", "resolved_placeholder_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_native_resolved_placeholder_partial_is_supported() -> None:
    placeholder = getattr(functools, "Placeholder", None)
    if placeholder is None:
        pytest.skip("functools.Placeholder requires Python 3.14")

    def reducer(
        configuration: int,
        offset: int,
        values: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        return np.sum(values + offset, axis=axis) * configuration

    inner = partial(reducer, placeholder, 0)
    inner.marker = "preserve nested partial"
    configured = partial(inner, 2)
    b, h = axes("native_resolved_placeholder_b", "native_resolved_placeholder_h")
    tensor = np.arange(2 * 3).reshape(2, 3)

    result = reduce(ax[b, h], ax[b]).reduce_by(configured)(tensor)

    np.testing.assert_array_equal(result, np.sum(tensor, axis=1) * 2)


def test_reduce_callable_scalar_output_is_coerced_to_rank_zero_tensor() -> None:
    b, h = axes("b", "h")

    def reducer(tensor: np.ndarray) -> float:
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[()]).reduce_by(reducer)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    assert result.shape == ()
    assert np.asarray(result).item() == float(np.sum(tensor))


def test_reduce_scalar_custom_output_does_not_silently_broadcast() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> float:
        _ = axis
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)
    tensor = np.arange(6).reshape(2, 3)

    with pytest.raises(ExecutionError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_scalar_custom_output_is_validated_without_reindex() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> float:
        _ = axis
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)

    with pytest.raises(ExecutionError) as error:
        _ = op(np.arange(6).reshape(2, 3))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_callable_typeerror_is_preserved() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64], *, axis: tuple[int, ...]
    ) -> NDArray[np.int64]:
        _ = tensor
        _ = axis
        raise TypeError("user boom")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="user boom"):
        _ = op(tensor)


def test_reduce_callable_valueerror_is_normalized_to_validation_error() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.float64], *, axis: tuple[int, ...]
    ) -> NDArray[np.float64]:
        _ = tensor
        _ = axis
        raise ValueError("domain exploded")

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)
    tensor = np.zeros((2, 0), dtype=np.float64)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "custom reducer failed" in str(error.value)


def test_reduce_string_reducer_empty_domain_is_normalized_to_validation_error() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[()]).reduce_by("max")

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 0)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "backend reducer 'max' failed" in str(error.value)


def test_reduce_builtin_numpy_max_callable_empty_domain_is_normalized() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[()]).reduce_by(np.max)

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 0)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "custom reducer failed" in str(error.value)


def test_reduce_callable_with_unsupported_signature_raises_validation_error() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64],
        axis: tuple[int, ...],
        axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        _ = axis
        _ = axes
        return tensor

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "multiple axes authorities" in str(error.value)


@pytest.mark.parametrize("signature_kind", ["empty", "axis_only"])
def test_reduce_concrete_invalid_signature_is_not_reported_as_ambiguous(
    signature_kind: str,
) -> None:
    if signature_kind == "empty":

        def reducer() -> np.int64:
            return np.int64(0)

    else:

        def reducer(  # type: ignore[no-redef]
            *,
            axis: tuple[int, ...],
        ) -> np.int64:
            _ = axis
            return np.int64(0)

    b, h = axes(f"invalid_{signature_kind}_b", f"invalid_{signature_kind}_h")
    op = reduce(ax[b, h], ax[()]).reduce_by(reducer)

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(2 * 3).reshape(2, 3))

    assert "reducer signature is unsupported" in str(error.value)
    assert "signature is unavailable" not in str(error.value)
