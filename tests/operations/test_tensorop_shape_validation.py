from dataclasses import dataclass

import numpy as np
import pytest
from array_api_compat import numpy as array_api_numpy

from einf import ErrorCode, ValidationError, ax, axes, contract, packs, rearrange
from einf.plans import abstract as abstract_plan_module
from einf.signature import Signature


@dataclass(frozen=True, slots=True)
class ArrayApiShapeTensor:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy

    def __getitem__(self, key: object) -> "ArrayApiShapeTensor":
        _ = key
        return self


def test_tensorop_call_revalidates_literal_dims_after_runner_cache_hit() -> None:
    op = rearrange(ax[2, 3], ax[3, 2])
    valid = np.arange(6, dtype=np.float32).reshape(2, 3)
    invalid = valid.reshape(3, 2)

    np.testing.assert_array_equal(op(valid), valid.T)
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_shared_dim_conflict_before_backend() -> None:
    b, k, m = axes("b", "k", "m")
    op = contract((ax[b, k], ax[k, m]), ax[b, m])
    left = np.ones((2, 3), dtype=np.float32)
    right = np.ones((4, 5), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(left, right)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_revalidates_with_sizes_after_runner_cache_hit() -> None:
    b, k, m = axes("b", "k", "m")
    op = contract((ax[b, k], ax[k, m]), ax[b, m]).with_sizes(k=3)
    valid_left = np.ones((2, 3), dtype=np.float32)
    valid_right = np.ones((3, 5), dtype=np.float32)
    invalid_left = np.ones((2, 4), dtype=np.float32)
    invalid_right = np.ones((4, 5), dtype=np.float32)

    np.testing.assert_array_equal(
        op(valid_left, valid_right),
        valid_left @ valid_right,
    )
    with pytest.raises(ValidationError) as error:
        _ = op(invalid_left, invalid_right)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_rank_mismatch_after_runner_cache_hit() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[c, b])
    valid = np.ones((2, 3), dtype=np.float32)
    invalid = np.ones((6,), dtype=np.float32)

    np.testing.assert_array_equal(op(valid), valid.T)
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_repeated_axis_conflict() -> None:
    (n,) = axes("n")
    op = rearrange(ax[n, n], ax[n, n])
    invalid = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_tuple_output_split_size_conflict() -> None:
    n, m = axes("n", "m")
    op = rearrange(ax[(n + m)], (ax[n], ax[m])).with_sizes(n=2, m=3)
    invalid = np.ones((6,), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_multi_input_route_shared_dim_conflict() -> None:
    (n,) = axes("n")
    op = rearrange((ax[n], ax[n]), (ax[n], ax[n]))
    left = np.ones((3,), dtype=np.float32)
    right = np.ones((4,), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(left, right)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_accepts_zero_length_literal_axis() -> None:
    op = rearrange(ax[0, 3], ax[3, 0])
    tensor = np.empty((0, 3), dtype=np.float32)

    result = op(tensor)

    assert result.shape == (3, 0)


def test_tensorop_call_accepts_scalar_shape_contract() -> None:
    op = rearrange(ax[()], ax[()])
    tensor = np.asarray(3.0, dtype=np.float32)

    result = op(tensor)

    assert result is tensor


def test_tensorop_call_rejects_unused_with_sizes_binding() -> None:
    (n,) = axes("n")
    op = rearrange(ax[n], ax[n]).with_sizes(unused=3)
    tensor = np.ones((3,), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_scalar_binding_for_axis_pack() -> None:
    (tail,) = packs("tail")
    op = rearrange(ax[tail], ax[tail]).with_sizes(tail=3)
    tensor = np.ones((3,), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_negative_custom_shape_entry() -> None:
    (n,) = axes("n")
    op = rearrange(ax[n], ax[n])
    tensor = ArrayApiShapeTensor(shape=(-1,))

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_revalidates_expression_dim_after_runner_cache_hit() -> None:
    (n,) = axes("n")
    op = rearrange(ax[(2 * n)], ax[n, 2]).with_sizes(n=3)
    valid = np.ones((6,), dtype=np.float32)
    invalid = np.ones((8,), dtype=np.float32)

    assert op(valid).shape == (3, 2)
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_accepts_zero_width_contraction() -> None:
    b, k, m = axes("b", "k", "m")
    op = contract((ax[b, k], ax[k, m]), ax[b, m])
    left = np.empty((2, 0), dtype=np.float32)
    right = np.empty((0, 3), dtype=np.float32)

    result = op(left, right)

    np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=np.float32))


def test_tensorop_call_reuses_shape_free_runner_across_valid_shapes() -> None:
    b, c = axes("b", "c")
    op = rearrange(ax[b, c], ax[c, b])
    first = np.arange(4, dtype=np.float32).reshape(1, 4)
    second = np.arange(3, dtype=np.float32).reshape(3, 1)

    np.testing.assert_array_equal(op(first), first.T)
    np.testing.assert_array_equal(op(second), second.T)


def test_tensorop_call_reuses_last_successful_shape_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, c = axes("validation_cache_b", "validation_cache_c")
    op = rearrange(ax[b, c], ax[c, b])
    tensor = np.ones((2, 3), dtype=np.float32)
    validation_call_count = 0
    validate_dimensions = abstract_plan_module.validate_dimensions

    def count_validation_calls(
        signature: Signature,
        input_shapes: tuple[tuple[int, ...], ...],
        *,
        explicit_sizes: dict[str, int] | None = None,
    ) -> None:
        nonlocal validation_call_count
        validation_call_count += 1
        validate_dimensions(
            signature,
            input_shapes,
            explicit_sizes=explicit_sizes,
        )

    monkeypatch.setattr(
        abstract_plan_module,
        "validate_dimensions",
        count_validation_calls,
    )

    _ = op(tensor)
    _ = op(tensor)

    assert validation_call_count == 1


def test_tensorop_call_validates_custom_type_runner_cache_hit() -> None:
    op = rearrange(ax[3], ax[3])
    valid = ArrayApiShapeTensor(shape=(3,))
    invalid = ArrayApiShapeTensor(shape=(4,))

    assert op(valid) is valid
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_revalidates_tuple_runner_cache_hit() -> None:
    op = rearrange(ax[5], (ax[2], ax[3]))
    valid = np.arange(5, dtype=np.float32)
    invalid = np.arange(6, dtype=np.float32)

    first, second = op(valid)
    np.testing.assert_array_equal(first, valid[:2])
    np.testing.assert_array_equal(second, valid[2:])
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_three_input_shared_dim_conflict() -> None:
    a, b, c, d = axes("a", "b", "c", "d")
    op = contract(
        (ax[a, b], ax[b, c], ax[c, d]),
        ax[a, d],
    )
    first = np.ones((2, 3), dtype=np.float32)
    second = np.ones((3, 4), dtype=np.float32)
    third = np.ones((5, 6), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(first, second, third)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_revalidates_zero_with_sizes_binding() -> None:
    (n,) = axes("n")
    op = rearrange(ax[n], ax[n]).with_sizes(n=0)
    valid = np.empty((0,), dtype=np.float32)
    invalid = np.ones((1,), dtype=np.float32)

    assert op(valid) is valid
    with pytest.raises(ValidationError) as error:
        _ = op(invalid)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_preserves_expression_ambiguity_diagnostic() -> None:
    h, w = axes("h", "w")
    op = rearrange(ax[(h * w)], ax[h, w])
    tensor = np.ones((12,), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_tensorop_call_supports_axis_pack_rank_changes() -> None:
    b = axes("b")[0]
    tail = packs("tail")[0]
    op = rearrange(ax[b, tail], ax[tail, b])
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = np.arange(30, dtype=np.float32).reshape(5, 6)

    np.testing.assert_array_equal(op(first), np.transpose(first, (1, 2, 0)))
    np.testing.assert_array_equal(op(second), second.T)
