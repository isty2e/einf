from dataclasses import dataclass

import numpy as np
import pytest

from einf import ErrorCode, ValidationError, ax, axes, view

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
