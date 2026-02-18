from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

import einf.plans.abstract as abstract_plan_module
import einf.steps.einsum as einsum_step_module
import einf.steps.einsum.step as einsum_step_impl
import einf.steps.permute as permute_step_module
from einf import (
    ErrorCode,
    ValidationError,
    ax,
    axes,
    contract,
    einop,
    packs,
    rearrange,
    reduce,
    repeat,
    view,
)
from einf.ir.routing import runtime as routing_runtime_module
from einf.lowering import einop as einop_plan_module
from einf.steps.expand import step as expand_step_module
from einf.steps.reduce import build as reduce_build_module
from einf.steps.reduce import step as reduce_step_module
from einf.tensor_types import TensorLike

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class _NoViewNamespace:
    __name__ = "array_api_compat.jax"


@dataclass(frozen=True, slots=True)
class NoViewTensor:
    shape: tuple[int, ...]

    def __array_namespace__(
        self, api_version: str | None = None
    ) -> type[_NoViewNamespace]:
        _ = api_version
        return _NoViewNamespace

    def __getitem__(self, key: object) -> "NoViewTensor":
        _ = key
        return self


class _MissingOpsNamespace:
    __name__ = "array_api_compat.numpy"


@dataclass(frozen=True, slots=True)
class MissingOpsTensor:
    shape: tuple[int, ...]

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> type[_MissingOpsNamespace]:
        _ = api_version
        return _MissingOpsNamespace

    def __getitem__(self, key: object) -> "MissingOpsTensor":
        _ = key
        return self


class _BadOpsNamespace:
    __name__ = "array_api_compat.numpy"
    permute_dims = None


@dataclass(frozen=True, slots=True)
class BadOpsTensor:
    shape: tuple[int, ...]

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> type[_BadOpsNamespace]:
        _ = api_version
        return _BadOpsNamespace

    def __getitem__(self, key: object) -> "BadOpsTensor":
        _ = key
        return self


class _ExplodingOpsNamespace:
    __name__ = "array_api_compat.numpy"

    @staticmethod
    def permute_dims(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("permute exploded")

    @staticmethod
    def reshape(tensor: object, _shape: tuple[int, ...]) -> object:
        return tensor

    @staticmethod
    def expand_dims(tensor: object, *, axis: int) -> object:
        _ = axis
        return tensor

    @staticmethod
    def broadcast_to(tensor: object, _shape: tuple[int, ...]) -> object:
        return tensor

    @staticmethod
    def concat(tensors: list[object], *, axis: int) -> object:
        _ = axis
        return tensors[0]

    @staticmethod
    def asarray(value: bool | int | float | complex) -> object:
        return np.asarray(value)

    @staticmethod
    def sum(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def prod(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def mean(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def max(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def min(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def all(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor

    @staticmethod
    def any(tensor: object, *, axis: tuple[int, ...]) -> object:
        _ = axis
        return tensor


@dataclass(frozen=True, slots=True)
class ExplodingOpsTensor:
    shape: tuple[int, ...]

    def __array_namespace__(
        self,
        api_version: str | None = None,
    ) -> type[_ExplodingOpsNamespace]:
        _ = api_version
        return _ExplodingOpsNamespace

    def __getitem__(self, key: object) -> "ExplodingOpsTensor":
        _ = key
        return self


def test_view_identity_executes_as_zero_copy_numpy_view() -> None:
    b, c = axes("b", "c")
    op = view(ax[b, c], ax[b, c])

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    assert np.shares_memory(result, tensor)
    np.testing.assert_array_equal(result, tensor)


def test_view_split_outputs_are_zero_copy_and_non_overlapping() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = view(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d])).with_sizes(n=2, m=3)

    tensor = np.arange(1 * 5 * 2).reshape(1, 5, 2)
    out_left, out_right = op(tensor)

    assert np.shares_memory(out_left, tensor)
    assert np.shares_memory(out_right, tensor)
    assert not np.shares_memory(out_left, out_right)
    np.testing.assert_array_equal(out_left, tensor[:, :2, :])
    np.testing.assert_array_equal(out_right, tensor[:, 2:, :])


def test_view_identity_with_concrete_plus_axis_is_zero_copy() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = view(ax[b, (n + m), d], ax[b, (n + m), d]).with_sizes(n=2, m=3)

    tensor = np.arange(1 * 5 * 2).reshape(1, 5, 2)
    result = op(tensor)

    assert np.shares_memory(result, tensor)
    np.testing.assert_array_equal(result, tensor)


def test_view_transpose_with_concrete_plus_axis_is_zero_copy() -> None:
    a, b = axes("a", "b")
    op = view(ax[(a + b), 2], ax[2, (a + b)]).with_sizes(a=2, b=3)

    tensor = np.arange(10).reshape(5, 2)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 0))
    assert np.shares_memory(result, tensor)
    np.testing.assert_array_equal(result, expected)


def test_view_rejects_axis_drop_mapping_as_not_a_view() -> None:
    b, c = axes("b", "c")
    op = view(ax[b, c], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 3)))

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_view_rejects_broadcast_like_mapping_as_not_a_view() -> None:
    b, c = axes("b", "c")
    op = view(ax[b, c], ax[b, c, 1])

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 3)))

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_view_rejects_overlapping_outputs_as_not_a_view() -> None:
    n, m = axes("n", "m")
    op = view(ax[(n + m)], (ax[(n + m)], ax[n])).with_sizes(n=2, m=3)

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(5))

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_view_torch_rejects_overlapping_outputs_as_not_a_view() -> None:
    assert torch is not None
    n, m = axes("n", "m")
    op = view(ax[(n + m)], (ax[(n + m)], ax[n])).with_sizes(n=2, m=3)

    with pytest.raises(ValidationError) as error:
        _ = op(torch.arange(5))

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_view_torch_split_outputs_are_zero_copy_and_non_overlapping() -> None:
    assert torch is not None
    n, m = axes("n", "m")
    op = view(ax[(n + m)], (ax[n], ax[m])).with_sizes(n=2, m=3)

    tensor = torch.arange(5)
    out_left, out_right = op(tensor)

    assert out_left.shape == (2,)
    assert out_right.shape == (3,)
    assert torch.equal(out_left, tensor[:2])
    assert torch.equal(out_right, tensor[2:])


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_view_torch_allows_disjoint_strided_outputs() -> None:
    assert torch is not None
    b, n, m = axes("b", "n", "m")
    op = view(ax[b, (n + m)], (ax[n, b], ax[m, b])).with_sizes(n=1, m=1)

    tensor = torch.arange(4).reshape(2, 2)
    out_left, out_right = op(tensor)

    assert torch.equal(out_left, torch.tensor([[0, 2]]))
    assert torch.equal(out_right, torch.tensor([[1, 3]]))


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_view_torch_rejects_large_stride_zero_overlapping_split_outputs() -> None:
    assert torch is not None
    n, m = axes("n", "m")
    op = view(ax[(n + m)], (ax[n], ax[m])).with_sizes(n=100_001, m=100_001)

    tensor = torch.tensor([1]).expand(200_002)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_view_split_with_empty_prefix_segment_is_allowed() -> None:
    n, m = axes("n", "m")
    op = view(ax[(n + m)], (ax[n], ax[m])).with_sizes(n=0, m=5)

    out_left, out_right = op(np.arange(5))

    assert out_left.shape == (0,)
    assert out_right.shape == (5,)


def test_view_rejects_non_contiguous_reshape_that_requires_copy() -> None:
    a, b, c = axes("a", "b", "c")
    op = view(ax[a, b, c], ax[(a * b), c]).with_sizes(a=3, b=2)

    tensor = np.arange(24).reshape(2, 3, 4).transpose(1, 0, 2)
    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_view_accepts_fortran_contiguous_merge_when_zero_copy_is_possible() -> None:
    a, b, c = axes("a", "b", "c")
    op = view(ax[a, b, c], ax[(a * b), c]).with_sizes(a=2, b=3)

    tensor = np.asfortranarray(np.arange(24).reshape(2, 3, 4))
    result = op(tensor)

    assert result.shape == (6, 4)
    assert np.shares_memory(result, tensor)


def test_view_rejects_backends_without_strict_view_capability() -> None:
    (b,) = axes("b")
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(NoViewTensor(shape=(3,)))

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_view_rejects_multi_input_lhs_with_diagnostic_code() -> None:
    (b,) = axes("b")

    with pytest.raises(ValidationError) as error:
        _ = getattr(view, "__call__")((ax[b], ax[b]), ax[b])

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value


def test_inflate_rejects_multi_input_lhs_with_diagnostic_code() -> None:
    (b,) = axes("b")

    with pytest.raises(ValidationError) as error:
        _ = getattr(repeat, "__call__")((ax[b], ax[b]), ax[b])

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value


def test_rearrange_transpose_executes_with_numpy() -> None:
    b, h, d = axes("b", "h", "d")
    op = rearrange(ax[b, h, d], ax[h, b, d])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 0, 2))
    np.testing.assert_array_equal(result, expected)


def test_rearrange_literal_rectangular_transpose_round_trip() -> None:
    op = rearrange(ax[2, 3], ax[3, 2])
    inverse = rearrange(ax[3, 2], ax[2, 3])

    tensor = np.arange(6).reshape(2, 3)
    transformed = op(tensor)
    restored = inverse(transformed)

    expected = np.transpose(tensor, (1, 0))
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(restored, tensor)


def test_rearrange_literal_square_identity_is_ambiguous() -> None:
    op = rearrange(ax[2, 2], ax[2, 2])

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(4).reshape(2, 2))

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_rearrange_allows_rhs_literal_to_consume_named_input_axis() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[2, b])

    tensor = np.arange(6).reshape(3, 2)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 0))
    np.testing.assert_array_equal(result, expected)


def test_rearrange_allows_named_output_axis_to_consume_lhs_literal_with_size_binding() -> (
    None
):
    b, c = axes("b", "c")
    op = rearrange(ax[2, b], ax[c, b]).with_sizes(c=2)

    tensor = np.arange(6).reshape(2, 3)
    result = op(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_rearrange_prefers_named_axis_match_over_literal_fallback() -> None:
    (a,) = axes("a")
    op = rearrange(ax[a, 2], ax[2, a]).with_sizes(a=2)

    tensor = np.arange(4).reshape(2, 2)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 0))
    np.testing.assert_array_equal(result, expected)


def test_rearrange_literal_fallback_backtracks_to_preserve_future_named_match() -> None:
    a, b = axes("a", "b")
    op = rearrange(ax[a, b], ax[2, a]).with_sizes(a=2, b=2)

    tensor = np.arange(4).reshape(2, 2)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 0))
    np.testing.assert_array_equal(result, expected)


def test_rearrange_piece_assignment_ambiguity_raises_ambiguous_dims() -> None:
    a, b = axes("a", "b")
    op = rearrange(
        (ax[a, b, 1], ax[a, a, a]),
        (ax[1, a, 1], ax[1, b, a]),
    ).with_sizes(a=1, b=1)

    left = np.array([[[11]]], dtype=np.int64)
    right = np.array([[[22]]], dtype=np.int64)

    with pytest.raises(ValidationError) as error:
        _ = op(left, right)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_rearrange_duplicate_input_output_shapes_are_ambiguous() -> None:
    (n,) = axes("n")
    op = rearrange((ax[n], ax[n]), (ax[n], ax[n])).with_sizes(n=3)

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(3), np.arange(3) + 10)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_rearrange_route_fastpath_skips_reindex_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n, m = axes("n", "m")
    op = rearrange((ax[n], ax[m]), (ax[m], ax[n])).with_sizes(n=2, m=3)

    monkeypatch.setattr(
        routing_runtime_module,
        "build_runtime_execution_context",
        _explode_native_contract_einsum,
    )

    first = np.arange(2)
    second = np.arange(3) + 10
    out_first, out_second = op(first, second)

    np.testing.assert_array_equal(out_first, second)
    np.testing.assert_array_equal(out_second, first)


def test_rearrange_multi_input_concat_prefers_exact_labeled_piece_match() -> None:
    b, c = axes("b", "c")
    op = rearrange(
        (ax[(3 + 1)], ax[(c + 1 + b)]),
        ax[((3 + 1) + (c + 1 + b))],
    ).with_sizes(b=3, c=1)

    left = np.arange(4)
    right = np.arange(4, 9)
    result = op(left, right)

    expected = np.concatenate((left, right), axis=0)
    np.testing.assert_array_equal(result, expected)


def test_rearrange_atomic_first_multi_input_equal_extents_can_be_ambiguous() -> None:
    (a,) = axes("a")
    op = rearrange((ax[(a + 1)], ax[3]), (ax[(a + 1)], ax[3])).with_sizes(a=2)

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(3), np.arange(3) + 10)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_rearrange_atomic_first_collapse_avoids_piecewise_ambiguity() -> None:
    a, b = axes("a", "b")
    op = rearrange(
        ax[((2 + a) + 2), (b + 2)],
        ax[(b + 2), ((2 + a) + 2)],
    ).with_sizes(a=1, b=2)
    inverse = rearrange(
        ax[(b + 2), ((2 + a) + 2)],
        ax[((2 + a) + 2), (b + 2)],
    ).with_sizes(a=1, b=2)

    tensor = np.arange(5 * 4).reshape(5, 4)
    transformed = op(tensor)
    restored = inverse(transformed)

    expected = np.transpose(tensor, (1, 0))
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(restored, tensor)


def test_rearrange_atomic_first_collapses_plus_to_literal_identity() -> None:
    a, b = axes("a", "b")
    op = rearrange(ax[(a + b)], ax[5]).with_sizes(a=2, b=3)

    tensor = np.arange(5)
    result = op(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_rearrange_piecewise_split_executes_under_canonical_arithmetic() -> None:
    (n,) = axes("n")
    op = rearrange(ax[(n + (n + n))], (ax[n], ax[(n + n)])).with_sizes(n=1)
    tensor = np.arange(3)

    left, right = op(tensor)

    np.testing.assert_array_equal(left, np.array([0]))
    np.testing.assert_array_equal(right, np.array([1, 2]))


def test_rearrange_fortran_input_uses_logical_axis_mapping() -> None:
    a, b, c = axes("a", "b", "c")
    op = rearrange(ax[a, b, c], ax[(a * b), c]).with_sizes(a=2, b=3)

    tensor = np.asfortranarray(np.arange(24).reshape(2, 3, 4))
    result = op(tensor)

    expected = np.reshape(tensor, (6, 4), order="C")
    np.testing.assert_array_equal(result, expected)


def test_rearrange_missing_backend_primitives_raises_backend_dispatch_error() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[c, b])

    with pytest.raises(ValidationError) as error:
        _ = op(MissingOpsTensor(shape=(2, 3)))

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_rearrange_non_callable_backend_primitive_raises_backend_dispatch_error() -> (
    None
):
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[c, b])

    with pytest.raises(ValidationError) as error:
        _ = op(BadOpsTensor(shape=(2, 3)))

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_rearrange_exploding_backend_primitive_is_normalized() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[c, b])

    with pytest.raises(ValidationError) as error:
        _ = op(ExplodingOpsTensor(shape=(2, 3)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "backend primitive failed during reindex execution" in str(error.value)


def test_rearrange_split_output_tuple_executes() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d])).with_sizes(n=2, m=3)

    tensor = np.arange(1 * 5 * 2).reshape(1, 5, 2)
    out_left, out_right = op(tensor)

    np.testing.assert_array_equal(out_left, tensor[:, :2, :])
    np.testing.assert_array_equal(out_right, tensor[:, 2:, :])


def test_rearrange_concat_multi_input_executes() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange((ax[b, n, d], ax[b, m, d]), ax[b, (n + m), d]).with_sizes(n=2, m=3)

    left = np.arange(1 * 2 * 2).reshape(1, 2, 2)
    right = np.arange(100, 100 + (1 * 3 * 2)).reshape(1, 3, 2)
    result = op(left, right)

    expected = np.concatenate((left, right), axis=1)
    np.testing.assert_array_equal(result, expected)


def test_rearrange_concat_allows_distributive_equivalence() -> None:
    b, h1, h2, h3, c = axes("b", "h1", "h2", "h3", "c")
    op = rearrange(
        (ax[b, (h1 * h3), c], ax[b, (h2 * h3), c]),
        ax[b, ((h1 + h2) * h3), c],
    ).with_sizes(h1=1, h2=2, h3=3)

    left = np.arange(1 * 3 * 1).reshape(1, 3, 1)
    right = np.arange(100, 100 + (1 * 6 * 1)).reshape(1, 6, 1)
    result = op(left, right)

    expected = np.concatenate((left, right), axis=1)
    np.testing.assert_array_equal(result, expected)


def test_rearrange_axis_slice_allows_distributive_equivalence() -> None:
    b, h1, h2, h3, c = axes("b", "h1", "h2", "h3", "c")
    op = rearrange(
        ax[b, ((h1 + h2) * h3), c],
        (ax[b, (h1 * h3), c], ax[b, (h2 * h3), c]),
    ).with_sizes(h1=1, h2=2, h3=3)

    tensor = np.arange(1 * 9 * 1).reshape(1, 9, 1)
    left, right = op(tensor)

    np.testing.assert_array_equal(left, tensor[:, :3, :])
    np.testing.assert_array_equal(right, tensor[:, 3:, :])


def test_rearrange_allows_nested_concat_regrouping_across_input_boundaries() -> None:
    b, h1, h2, h3, c = axes("b", "h1", "h2", "h3", "c")
    op = rearrange(
        (ax[b, (h1 + h2), c], ax[b, h3, c]),
        ax[b, (h1 + (h2 + h3)), c],
    ).with_sizes(h1=1, h2=1, h3=2)

    left = np.arange(1 * 2 * 1).reshape(1, 2, 1)
    right = np.arange(100, 100 + (1 * 2 * 1)).reshape(1, 2, 1)
    result = op(left, right)

    expected = np.concatenate((left, right), axis=1)
    np.testing.assert_array_equal(result, expected)


def test_rearrange_allows_commutative_nested_concat_regrouping() -> None:
    b, n1, n2, n3, n4, d = axes("b", "n1", "n2", "n3", "n4", "d")
    op = rearrange(
        (ax[b, (n1 + n2), d], ax[b, n3, d], ax[b, n4, d]),
        ax[b, ((n1 + n3) + (n2 + n4)), d],
    ).with_sizes(n1=1, n2=1, n3=1, n4=1)

    first = np.arange(1 * 2 * 1).reshape(1, 2, 1)
    second = np.arange(100, 100 + (1 * 1 * 1)).reshape(1, 1, 1)
    third = np.arange(200, 200 + (1 * 1 * 1)).reshape(1, 1, 1)
    result = op(first, second, third)

    expected = np.concatenate((first, second, third), axis=1)
    np.testing.assert_array_equal(result, expected)


def test_rearrange_allows_unary_nested_concat_regrouping() -> None:
    b, n1, n2, n3, d = axes("b", "n1", "n2", "n3", "d")
    op = rearrange(
        ax[b, (n1 + (n2 + n3)), d],
        ax[b, ((n1 + n2) + n3), d],
    ).with_sizes(n1=1, n2=1, n3=2)

    tensor = np.arange(1 * 4 * 1).reshape(1, 4, 1)
    result = op(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_rearrange_numel_mismatch_grow_raises_specific_code() -> None:
    b, c, r = axes("b", "c", "r")
    op = rearrange(ax[b, c], ax[b, c, r]).with_sizes(r=2)

    with pytest.raises(ValidationError) as error:
        _ = op(np.ones((2, 3)))

    assert error.value.code == ErrorCode.NUMEL_MISMATCH_GROW.value


def test_rearrange_numel_mismatch_shrink_raises_specific_code() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(np.ones((2, 3)))

    assert error.value.code == ErrorCode.NUMEL_MISMATCH_SHRINK.value


def test_rearrange_pack_reorders_variadic_axes() -> None:
    (b,) = axes("b")
    (tail,) = packs("tail")
    op = rearrange(ax[b, tail], ax[tail, b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.transpose(tensor, (1, 2, 0))
    np.testing.assert_array_equal(result, expected)


def test_rearrange_pack_can_match_empty_axis_sequence() -> None:
    (b,) = axes("b")
    (tail,) = packs("tail")
    op = rearrange(ax[b, tail], ax[tail, b])

    tensor = np.arange(2)
    result = op(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_rearrange_repeated_pack_must_match_across_inputs() -> None:
    b, c = axes("b", "c")
    (tail,) = packs("tail")
    with pytest.raises(ValidationError) as error:
        rearrange((ax[tail, b], ax[tail, c]), ax[tail, b, c])

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_rearrange_rejects_axis_names_colliding_with_internal_pack_names() -> None:
    (head,) = axes("_einf_pack_tail_0")
    (tail,) = packs("tail")
    op = rearrange(ax[head, tail], ax[tail, head])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "collide with internal pack expansion names" in str(error.value)


def test_contract_matrix_multiply_executes_with_numpy() -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def _explode_opt_einsum_contract(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "opt_einsum.contract should not be called on torch native contract path"
    )


def _explode_native_contract_einsum(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native contract einsum should not be called in this path")


def test_contract_matrix_multiply_numpy_prefers_native_matmul_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_contract_matrix_multiply_torch_uses_native_einsum_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    assert torch is not None
    left = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    right = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    result = op(left, right)

    expected = left @ right
    assert torch.equal(result, expected)


def test_contract_three_inputs_reuses_cached_contract_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, j, k, out = axes("i", "j", "k", "out")
    op = contract((ax[i, j], ax[j, k], ax[k, out]), ax[i, out])
    compiled_calls = {"value": 0}

    def _spy_contract_expression(
        equation: str,
        *operand_shapes: tuple[int, ...],
        optimize: str,
    ) -> Callable[..., np.ndarray]:
        compiled_calls["value"] += 1
        _ = operand_shapes
        _ = optimize

        def _run(*operands: np.ndarray) -> np.ndarray:
            return np.einsum(equation, *operands, optimize=True)

        return _run

    einsum_step_impl._cached_contract_expression.cache_clear()

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract_expression",
        _spy_contract_expression,
    )
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_native_contract_einsum,
    )

    left = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    middle = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    right = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
    first = op(left, middle, right)
    second = op(left, middle, right)

    expected = left @ middle @ right
    assert compiled_calls["value"] == 1
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_einop_chain_two_input_torch_uses_native_contract_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_native_contract_einsum,
    )

    assert torch is not None
    lhs = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
    rhs = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    out_left, out_right = op(lhs, rhs)

    expected = torch.einsum("bmn,nd->bmd", lhs, rhs)
    assert torch.equal(out_left, expected[:, :6, :])
    assert torch.equal(out_right, expected[:, 6:, :])


def test_einop_concat_matches_multi_input_rearrange_concat() -> None:
    a, b, c = axes("a", "b", "c")
    op = einop((ax[a, b], ax[c, b]), ax[(a + c), b])

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(4 * 3).reshape(4, 3) + 100
    result = op(left, right)

    expected = np.concatenate((left, right), axis=0)
    np.testing.assert_array_equal(result, expected)


def test_einop_factorized_contract_executes_with_numpy() -> None:
    b, h, d, j = axes("b", "h", "d", "j")
    op = einop((ax[b, (h * d)], ax[(h * d), j]), ax[b, j]).with_sizes(h=2, d=3)

    left = np.arange(2 * 6).reshape(2, 6)
    right = np.arange(6 * 4).reshape(6, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def test_einop_atomic_contract_skips_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    op = einop((ax[b, n, d], ax[d, j]), ax[b, n, j])

    monkeypatch.setattr(
        permute_step_module,
        "build_runtime_execution_context",
        _explode_native_contract_einsum,
    )

    left = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    right = np.arange(4 * 5).reshape(4, 5)
    result = op(left, right)

    expected = np.einsum("bnd,dj->bnj", left, right, optimize=True)
    np.testing.assert_array_equal(result, expected)


def test_einop_factorized_contract_with_pack_executes_with_numpy() -> None:
    h, d, j = axes("h", "d", "j")
    (tail,) = packs("tail")
    op = einop((ax[tail, (h * d)], ax[(h * d), j]), ax[tail, j]).with_sizes(h=2, d=3)

    left = np.arange(2 * 5 * 6).reshape(2, 5, 6)
    right = np.arange(6 * 4).reshape(6, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def test_einop_repeat_like_path_does_not_use_tensorop_contract_fastpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, c, r = axes("b", "c", "r")
    op = einop(ax[b, c], ax[b, c, r]).with_sizes(r=4)
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

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    assert not called["value"]
    np.testing.assert_array_equal(result, expected)


def test_einop_contract_then_split_outputs_executes_with_numpy() -> None:
    b, h, w, d, j = axes("b", "h", "w", "d", "j")
    op = einop((ax[b, (h + w), d], ax[d, j]), (ax[b, h, j], ax[b, w, j])).with_sizes(
        h=2, w=1
    )

    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    out_left, out_right = op(left, right)

    intermediate = np.einsum("bnd,dj->bnj", left, right)
    expected_left = intermediate[:, :2, :]
    expected_right = intermediate[:, 2:, :]
    np.testing.assert_array_equal(out_left, expected_left)
    np.testing.assert_array_equal(out_right, expected_right)


def test_einop_three_input_contract_then_split_outputs_executes_with_numpy() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    middle = np.arange(5 * 4).reshape(5, 4)
    right = np.arange(4 * 6).reshape(4, 6)
    out_left, out_right = op(left, middle, right)

    intermediate = np.einsum("bnd,dj,jk->bnk", left, middle, right)
    expected_left = intermediate[:, :2, :]
    expected_right = intermediate[:, 2:, :]
    np.testing.assert_array_equal(out_left, expected_left)
    np.testing.assert_array_equal(out_right, expected_right)


def test_einop_cached_chain_split_skips_nested_stage_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    monkeypatch.setattr(
        einop_plan_module,
        "build_einop_execution_plan",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_einop_cached_chain_split_skips_tensor_op_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_einop_cached_chain_split_uses_direct_slice_tail_fastpath() -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_shape_free_single_runner_cache_is_arity_agnostic_for_ternary_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d, j, k = axes("b", "n", "d", "j", "k")
    op = contract((ax[b, n, d], ax[d, j], ax[j, k]), ax[b, n, k])

    first_lhs = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    first_mid = np.arange(4 * 5).reshape(4, 5)
    first_rhs = np.arange(5 * 6).reshape(5, 6)
    np.testing.assert_array_equal(
        op(first_lhs, first_mid, first_rhs),
        np.einsum("bnd,dj,jk->bnk", first_lhs, first_mid, first_rhs),
    )

    monkeypatch.setattr(
        abstract_plan_module.AbstractPlan,
        "resolve_single_output_runner",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 3 * 4).reshape(3, 3, 4)
    second_mid = np.arange(4 * 5).reshape(4, 5)
    second_rhs = np.arange(5 * 6).reshape(5, 6)
    np.testing.assert_array_equal(
        op(second_lhs, second_mid, second_rhs),
        np.einsum("bnd,dj,jk->bnk", second_lhs, second_mid, second_rhs),
    )


def test_shape_free_tuple_runner_cache_is_arity_agnostic_for_ternary_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    first_lhs = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    first_mid = np.arange(5 * 4).reshape(5, 4)
    first_rhs = np.arange(4 * 6).reshape(4, 6)
    first_out_left, first_out_right = op(first_lhs, first_mid, first_rhs)
    first_intermediate = np.einsum("bnd,dj,jk->bnk", first_lhs, first_mid, first_rhs)
    np.testing.assert_array_equal(first_out_left, first_intermediate[:, :2, :])
    np.testing.assert_array_equal(first_out_right, first_intermediate[:, 2:, :])

    monkeypatch.setattr(
        abstract_plan_module.AbstractPlan,
        "resolve_tuple_runner",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 3 * 5).reshape(3, 3, 5)
    second_mid = np.arange(5 * 4).reshape(5, 4)
    second_rhs = np.arange(4 * 6).reshape(4, 6)
    second_out_left, second_out_right = op(second_lhs, second_mid, second_rhs)
    second_intermediate = np.einsum(
        "bnd,dj,jk->bnk", second_lhs, second_mid, second_rhs
    )
    np.testing.assert_array_equal(second_out_left, second_intermediate[:, :2, :])
    np.testing.assert_array_equal(second_out_right, second_intermediate[:, 2:, :])


def test_einop_inflate_like_broadcast_matches_inflate() -> None:
    b, c, r = axes("b", "c", "r")
    op = einop(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


def test_einop_reduce_default_sum_matches_reduce() -> None:
    b, c = axes("b", "c")
    op = einop(ax[b, c], ax[b])

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.sum(tensor, axis=1)
    np.testing.assert_array_equal(result, expected)


def test_einop_reduce_by_callable_matches_reduce_plus_inflate_pipeline() -> None:
    b, c, r = axes("b", "c", "r")
    op = (
        einop(ax[b, c], ax[b, r])
        .with_sizes(r=2)
        .reduce_by(lambda x, *, axis: np.sum(x, axis=axis))
    )

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    reduced = np.sum(tensor, axis=1)
    expected = np.broadcast_to(np.expand_dims(reduced, axis=1), (2, 2))
    np.testing.assert_array_equal(result, expected)


def test_contract_dot_product_to_scalar_executes_with_numpy() -> None:
    i = axes("i")[0]
    op = contract((ax[i], ax[i]), ax[()])

    left = np.arange(5)
    right = np.arange(5)
    result = op(left, right)

    expected = np.asarray(np.dot(left, right))
    assert result.shape == ()
    np.testing.assert_array_equal(result, expected)


def test_contract_rank_zero_identity_executes_with_numpy() -> None:
    op = contract(ax[()], ax[()])

    tensor = np.asarray(7)
    result = op(tensor)

    assert result.shape == ()
    np.testing.assert_array_equal(result, tensor)


def test_contract_trace_and_diagonal_match_numpy() -> None:
    i = axes("i")[0]
    op_trace = contract(ax[i, i], ax[()])
    op_diag = contract(ax[i, i], ax[i])

    tensor = np.arange(16).reshape(4, 4)
    trace_result = op_trace(tensor)
    diag_result = op_diag(tensor)

    trace_expected = np.einsum("ii->", tensor)
    diag_expected = np.einsum("ii->i", tensor)
    np.testing.assert_array_equal(trace_result, trace_expected)
    np.testing.assert_array_equal(diag_result, diag_expected)


def test_contract_three_input_chain_executes_with_numpy() -> None:
    a, b, c, d = axes("a", "b", "c", "d")
    op = contract((ax[a, b], ax[b, c], ax[c, d]), ax[a, d])

    x = np.arange(2 * 3).reshape(2, 3)
    y = np.arange(3 * 4).reshape(3, 4)
    z = np.arange(4 * 5).reshape(4, 5)
    result = op(x, y, z)

    expected = x @ y @ z
    np.testing.assert_array_equal(result, expected)


def test_contract_rejects_non_atomic_axis_expression_at_constructor() -> None:
    b, h, w, c = axes("b", "h", "w", "c")

    with pytest.raises(ValidationError) as error:
        _ = contract((ax[b, (h * w)], ax[(h * w), c]), ax[b, c])

    assert error.value.code == ErrorCode.CONTRACT_NON_ATOMIC_AXIS.value


def test_contract_rejects_output_axis_missing_from_inputs() -> None:
    i, j = axes("i", "j")
    op = contract(ax[i], ax[j]).with_sizes(j=3)
    tensor = np.arange(3)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_contract_rejects_duplicate_output_axis_names() -> None:
    i = axes("i")[0]
    op = contract(ax[i], ax[i, i])
    tensor = np.arange(3)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_rearrange_two_pack_split_is_ambiguous_without_constraints() -> None:
    b = axes("b")[0]
    t1, t2 = packs("t1", "t2")
    op = rearrange(ax[t1, b, t2], ax[t2, b, t1])
    tensor = np.zeros((2, 3, 4), dtype=np.int64)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_rearrange_allows_pack_prefix_axis_name_when_no_pack_is_used() -> None:
    (head,) = axes("_einf_pack_tail_0")
    op = rearrange(ax[head], ax[head])

    tensor = np.arange(3)
    result = op(tensor)
    np.testing.assert_array_equal(result, tensor)


def test_inflate_appends_axis_and_broadcasts_values() -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


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


def test_rearrange_multi_output_split_uses_slice_fastpath() -> None:
    b, h1, h2, d = axes("b", "h1", "h2", "d")
    op = rearrange(
        ax[b, (h1 + h2), d],
        (ax[b, h1, d], ax[b, h2, d]),
    ).with_sizes(h1=2, h2=1)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    left, right = op(tensor)

    np.testing.assert_array_equal(left, tensor[:, :2, :])
    np.testing.assert_array_equal(right, tensor[:, 2:, :])


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
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a=2)
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
        reducer: reduce_build_module.Reducer,
        pack_sizes: dict[str, tuple[int, ...]],
        axis_sizes: dict[str, int],
        tensor: TensorLike,
        xp: reduce_build_module.ArrayNamespace,
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
            tensor=tensor,
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


def test_reduce_uninspectable_callable_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by(np.add.reduce)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_uninspectable_callable_typeerror_is_not_swallowed() -> None:
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            /,
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = tensor
            _ = axis
            raise TypeError("user boom")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="user boom"):
        _ = op(tensor)


def test_reduce_uninspectable_callable_typeerror_with_binding_like_message_is_not_swallowed() -> (
    None
):
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            /,
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = tensor
            _ = axis
            raise TypeError("missing 1 required positional argument: 'x'")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _ = op(tensor)


def test_reduce_uninspectable_function_typeerror_is_not_swallowed() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        _ = tensor
        _ = axis
        raise TypeError("missing 1 required positional argument: 'x'")

    setattr(reducer, "__signature__", object())
    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _ = op(tensor)


def test_reduce_uninspectable_callable_unsupported_signature_is_normalized() -> None:
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            axis: tuple[int, ...],
            axes: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = axis
            _ = axes
            return tensor

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "reducer signature is unsupported" in str(error.value)


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

    with pytest.raises(ValidationError) as error:
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

    with pytest.raises(ValidationError) as error:
        _ = op(np.arange(6).reshape(2, 3))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_rearrange_split_handles_zero_length_plus_segments() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d])).with_sizes(n=0, m=3)

    tensor = np.arange(1 * 3 * 2).reshape(1, 3, 2)
    out_left, out_right = op(tensor)

    assert out_left.shape == (1, 0, 2)
    np.testing.assert_array_equal(out_right, tensor)


def test_reduce_callable_typeerror_is_not_swallowed_by_fallback_dispatch() -> None:
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
    assert "reducer signature is unsupported" in str(error.value)
