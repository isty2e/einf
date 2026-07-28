from dataclasses import dataclass

import numpy as np
import pytest

import einf.plans.routing as plan_routing_module
from einf import ErrorCode, ValidationError, ax, axes, packs, rearrange


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
    def asarray(value: bool | complex) -> object:
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


def _explode_native_contract_einsum(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native contract einsum should not be called in this path")


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
        plan_routing_module,
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


def test_rearrange_split_handles_zero_length_plus_segments() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d])).with_sizes(n=0, m=3)

    tensor = np.arange(1 * 3 * 2).reshape(1, 3, 2)
    out_left, out_right = op(tensor)

    assert out_left.shape == (1, 0, 2)
    np.testing.assert_array_equal(out_right, tensor)
