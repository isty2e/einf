import numpy as np
import pytest

from einf import ValidationError, ax, axes, einop, rearrange, repeat


def test_unary_einop_preserves_reordered_composite_axis_layout() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_multi_input_einop_aligns_reordered_composite_contraction_axis() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.arange(12).reshape(2, 6)
    rhs = np.arange(24).reshape(6, 4)
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        ax[b, d],
    ).with_sizes(h=2, w=3)
    align_rhs = rearrange(ax[w * h, d], ax[h * w, d]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ align_rhs(rhs))


def test_einop_preserves_composite_association_without_layout_change() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24)
    op = einop(ax[(h * w) * d], ax[h * (w * d)]).with_sizes(h=2, w=3, d=4)

    np.testing.assert_array_equal(op(tensor), tensor)


def test_multi_input_einop_emits_requested_composite_output_layout() -> None:
    b, h, w = axes("b", "h", "w")
    tensor = np.arange(12).reshape(2, 6)
    weights = np.array([2, 3])
    op = einop(
        (ax[b, h * w], ax[b]),
        ax[w * h],
    ).with_sizes(h=2, w=3)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor, weights), output_layout(weights @ tensor))


def test_matching_composite_contraction_keeps_flat_einsum_path() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    op = einop(
        (ax[b, h * w], ax[h * w, d]),
        ax[b, d],
    ).with_sizes(h=2, w=3)

    plan = op.plan_dict()

    assert plan["kind"] == "einsum"
    assert plan["steps"] == [{"op": "einsum", "equation": "ab,bc->ac"}]


def test_unary_einop_reorders_literal_composite_factor() -> None:
    (h,) = axes("h")
    tensor = np.arange(6)
    op = einop(ax[h * 2], ax[2 * h]).with_sizes(h=3)
    reference = rearrange(ax[h * 2], ax[2 * h]).with_sizes(h=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_einop_reorders_composite_layout_after_reduction() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24).reshape(6, 4)
    op = einop(ax[h * w, d], ax[w * h]).with_sizes(h=2, w=3)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), output_layout(tensor.sum(axis=1)))


def test_einop_reorders_composite_layout_before_repeat() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(6)
    op = einop(ax[h * w], ax[w * h, d]).with_sizes(h=2, w=3, d=4)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    expected = np.broadcast_to(output_layout(tensor)[:, None], (6, 4))

    np.testing.assert_array_equal(op(tensor), expected)


def test_einop_reorders_composite_layout_after_custom_reduction() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24).reshape(6, 4)
    op = einop(ax[h * w, d], ax[w * h]).with_sizes(h=2, w=3).reduce_by("max")
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), output_layout(tensor.max(axis=1)))


def test_einop_reorders_multiple_composite_extent_classes() -> None:
    h, w, d, k = axes("h", "w", "d", "k")
    tensor = np.arange(48).reshape(6, 8)
    op = einop(
        ax[h * w, d * k],
        ax[w * h, k * d],
    ).with_sizes(h=2, w=3, d=2, k=4)
    reference = rearrange(
        ax[h * w, d * k],
        ax[w * h, k * d],
    ).with_sizes(h=2, w=3, d=2, k=4)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_equal_numeric_extents_do_not_merge_unrelated_composites() -> None:
    h, w, d, k = axes("h", "w", "d", "k")
    op = einop(ax[h * w], ax[d * k]).with_sizes(h=2, w=3, d=2, k=3)

    assert op.plan_dict()["kind"] != "layout_normalized"


def test_einop_routes_multiple_requested_composite_output_layouts() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(
        ax[h * w],
        (ax[h * w], ax[w * h]),
    ).with_sizes(h=2, w=3)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    outputs = op(tensor)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[0], tensor)
    np.testing.assert_array_equal(outputs[1], output_layout(tensor))


def test_layout_normalization_reuses_runner_across_inferred_factor_sizes() -> None:
    h, w = axes("h", "w")
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=2)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2)

    for factor_size in (3, 4):
        tensor = np.arange(2 * factor_size)
        np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_accepts_non_contiguous_input() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(48).reshape(6, 8)[:, ::2]
    assert not tensor.flags.c_contiguous
    op = einop(ax[h * w, d], ax[w * h, d]).with_sizes(h=2, w=3)
    reference = rearrange(ax[h * w, d], ax[w * h, d]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_handles_empty_composite_extent() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(0)
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=0, w=3)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=0, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_handles_singleton_factor() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(5)
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=1, w=5)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=1, w=5)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_reverses_three_composite_factors() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24)
    op = einop(ax[h * w * d], ax[d * w * h]).with_sizes(h=2, w=3, d=4)
    reference = rearrange(ax[h * w * d], ax[d * w * h]).with_sizes(
        h=2,
        w=3,
        d=4,
    )

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_three_input_einop_aligns_one_reordered_composite_axis() -> None:
    b, h, w, d, k = axes("b", "h", "w", "d", "k")
    lhs = np.arange(12).reshape(2, 6)
    middle = np.arange(24).reshape(6, 4)
    rhs = np.arange(12).reshape(4, 3)
    op = einop(
        (ax[b, h * w], ax[w * h, d], ax[d, k]),
        ax[b, k],
    ).with_sizes(h=2, w=3)
    align_middle = rearrange(ax[w * h, d], ax[h * w, d]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(
        op(lhs, middle, rhs),
        lhs @ align_middle(middle) @ rhs,
    )


def test_layout_normalized_contraction_supports_multiple_outputs() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.arange(12).reshape(2, 6)
    rhs = np.arange(24).reshape(6, 4)
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        (ax[b, d], ax[d, b]),
    ).with_sizes(h=2, w=3)
    align_rhs = rearrange(ax[w * h, d], ax[h * w, d]).with_sizes(h=2, w=3)
    expected = lhs @ align_rhs(rhs)

    outputs = op(lhs, rhs)

    assert isinstance(outputs, tuple)
    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(outputs[1], expected.T)


def test_layout_normalized_einop_maps_multiple_composite_outputs() -> None:
    b, h, w = axes("b", "h", "w")
    tensor = np.arange(12).reshape(2, 6)
    weights = np.array([2, 3])
    op = einop(
        (ax[b, h * w], ax[b]),
        (ax[h * w], ax[w * h]),
    ).with_sizes(h=2, w=3)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    expected = weights @ tensor

    outputs = op(tensor, weights)

    assert isinstance(outputs, tuple)
    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(outputs[1], output_layout(expected))


def test_layout_normalization_preserves_boolean_values() -> None:
    h, w = axes("h", "w")
    tensor = np.array([True, False, True, True, False, False])
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_accepts_negative_stride_input() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)[::-1]
    assert tensor.strides[0] < 0
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    reference = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_rejects_underconstrained_factor_sizes() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(ax[h * w], ax[w * h])

    with pytest.raises(ValidationError):
        op(tensor)


def test_layout_normalization_supports_ordered_reducer_phases() -> None:
    h, w, d, k = axes("h", "w", "d", "k")
    tensor = np.arange(36).reshape(6, 2, 3)
    op = (
        einop(ax[h * w, d, k], ax[w * h])
        .with_sizes(h=2, w=3)
        .reduce_by((ax[d], "max"), (ax[k], "sum"))
    )
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    expected = tensor.max(axis=1).sum(axis=1)

    np.testing.assert_array_equal(op(tensor), output_layout(expected))


def test_layout_normalization_reorders_additive_composite_factor() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(20)
    op = einop(ax[(h + w) * d], ax[d * (h + w)]).with_sizes(h=2, w=3, d=4)
    reference = rearrange(ax[(h + w) * d], ax[d * (h + w)]).with_sizes(
        h=2,
        w=3,
        d=4,
    )

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_supports_callable_reducer() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24).reshape(6, 4)

    def reducer(value: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        return np.max(value, axis=axis)

    op = einop(ax[h * w, d], ax[w * h]).with_sizes(h=2, w=3).reduce_by(reducer)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), output_layout(tensor.max(axis=1)))


def test_layout_normalization_preserves_reducer_phase_order() -> None:
    h, w, d, k = axes("h", "w", "d", "k")
    tensor = np.arange(36).reshape(6, 2, 3)
    op = (
        einop(ax[h * w, d, k], ax[w * h])
        .with_sizes(h=2, w=3)
        .reduce_by((ax[k], "sum"), (ax[d], "max"))
    )
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)
    expected = tensor.sum(axis=2).max(axis=1)

    np.testing.assert_array_equal(op(tensor), output_layout(expected))


def test_layout_only_einop_rejects_reducer_configuration() -> None:
    h, w = axes("h", "w")

    with pytest.raises(ValidationError):
        einop(ax[h * w], ax[w * h]).reduce_by("sum")


def test_layout_normalization_preserves_repeated_composite_occurrences() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(36).reshape(6, 6)
    op = einop(
        ax[h * w, w * h],
        ax[h * w, w * h],
    ).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), tensor)


def test_layout_normalization_rejects_ambiguous_composite_occurrence_swap() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(36).reshape(6, 6)

    with pytest.raises(ValidationError):
        op = einop(
            ax[h * w, w * h],
            ax[w * h, h * w],
        ).with_sizes(h=2, w=3)
        op(tensor)


def test_layout_normalized_einop_contracts_to_scalar() -> None:
    h, w = axes("h", "w")
    lhs = np.arange(6)
    rhs = np.arange(6) + 1
    op = einop(
        (ax[h * w], ax[w * h]),
        ax[()],
    ).with_sizes(h=2, w=3)
    align_rhs = rearrange(ax[w * h], ax[h * w]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ align_rhs(rhs))


def test_layout_normalized_contraction_handles_empty_extent() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.empty((2, 0))
    rhs = np.empty((0, 4))
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        ax[b, d],
    ).with_sizes(h=0, w=3)

    np.testing.assert_array_equal(op(lhs, rhs), np.zeros((2, 4)))


def test_layout_normalization_rejects_composite_extent_shape_mismatch() -> None:
    h, w = axes("h", "w")
    op = einop(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    with pytest.raises(ValidationError):
        op(np.arange(5))


def test_layout_normalization_aligns_associated_contraction_axes() -> None:
    b, h, w, d, k = axes("b", "h", "w", "d", "k")
    lhs = np.arange(48).reshape(2, 24)
    rhs = np.arange(72).reshape(24, 3)
    op = einop(
        (ax[b, (h * w) * d], ax[h * (w * d), k]),
        ax[b, k],
    ).with_sizes(h=2, w=3, d=4)

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ rhs)


def test_layout_normalized_contraction_preserves_complex_values() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.arange(12).reshape(2, 6) * (1 + 2j)
    rhs = np.arange(24).reshape(6, 4) * (2 - 1j)
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        ax[b, d],
    ).with_sizes(h=2, w=3)
    align_rhs = rearrange(ax[w * h, d], ax[h * w, d]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ align_rhs(rhs))


def test_layout_normalization_keeps_unvaried_composite_axis_collapsed() -> None:
    b, h, w, d, k, j = axes("b", "h", "w", "d", "k", "j")
    lhs = np.arange(96).reshape(2, 6, 8)
    rhs = np.arange(192).reshape(6, 8, 4)
    op = einop(
        (ax[b, h * w, d * k], ax[w * h, d * k, j]),
        ax[b, j],
    ).with_sizes(h=2, w=3, d=2, k=4)
    align_rhs = rearrange(
        ax[w * h, d * k, j],
        ax[h * w, d * k, j],
    ).with_sizes(h=2, w=3, d=2, k=4)

    np.testing.assert_array_equal(
        op(lhs, rhs),
        np.einsum("bxy,xyj->bj", lhs, align_rhs(rhs)),
    )
    assert op.plan_dict()["steps"][-1] == {
        "op": "einsum",
        "equation": "abcd,cbde->ae",
    }


def test_layout_normalization_regroups_composite_factors() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24)
    op = einop(
        ax[(h * w) * d],
        ax[d, w * h],
    ).with_sizes(h=2, w=3, d=4)
    expected = tensor.reshape(2, 3, 4).transpose(2, 1, 0).reshape(4, 6)

    np.testing.assert_array_equal(op(tensor), expected)


def test_layout_normalization_reduces_one_composite_factor() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(ax[h * w], ax[h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(2, 3).sum(axis=1))


def test_layout_normalization_repeats_into_composite_factor() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(2)
    op = einop(ax[h], ax[w * h]).with_sizes(h=2, w=3)
    expanded = repeat(ax[h], ax[w, h]).with_sizes(w=3)
    flatten = rearrange(ax[w, h], ax[w * h]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), flatten(expanded(tensor)))


def test_layout_normalized_multi_output_runner_reuses_dynamic_shapes() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        (ax[b, d], ax[d, b]),
    ).with_sizes(h=2, w=3)
    align_rhs = rearrange(ax[w * h, d], ax[h * w, d]).with_sizes(h=2, w=3)

    for batch_size, output_size in ((2, 4), (3, 5)):
        lhs = np.arange(batch_size * 6).reshape(batch_size, 6)
        rhs = np.arange(6 * output_size).reshape(6, output_size)
        expected = lhs @ align_rhs(rhs)
        outputs = op(lhs, rhs)

        assert isinstance(outputs, tuple)
        np.testing.assert_array_equal(outputs[0], expected)
        np.testing.assert_array_equal(outputs[1], expected.T)


def test_layout_normalization_accepts_fortran_order_input() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.asfortranarray(np.arange(24).reshape(6, 4))
    assert tensor.flags.f_contiguous
    op = einop(ax[h * w, d], ax[w * h, d]).with_sizes(h=2, w=3)
    reference = rearrange(ax[h * w, d], ax[w * h, d]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_reorders_zero_literal_factor() -> None:
    (h,) = axes("h")
    tensor = np.arange(0)
    op = einop(ax[h * 0], ax[0 * h]).with_sizes(h=2)
    reference = rearrange(ax[h * 0], ax[0 * h]).with_sizes(h=2)

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_einop_identity_with_overlapping_products_keeps_route_path() -> None:
    h, w, d = axes("h", "w", "d")
    op = einop(
        ax[h * w, h * d],
        ax[h * w, h * d],
    ).with_sizes(h=2, w=3, d=4)

    assert op.plan_dict()["kind"] == "route"


def test_layout_normalization_flattens_atomic_factors() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6).reshape(2, 3)
    op = einop(ax[h, w], ax[h * w]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(6))


def test_layout_normalization_unflattens_composite_factor() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(ax[h * w], ax[h, w]).with_sizes(h=2, w=3)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(2, 3))


def test_layout_normalization_reorders_atoms_into_composite_output() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(24).reshape(2, 3, 4)
    op = einop(ax[h, w, d], ax[d, w * h]).with_sizes(h=2, w=3, d=4)
    expected = tensor.transpose(2, 1, 0).reshape(4, 6)

    np.testing.assert_array_equal(op(tensor), expected)


def test_layout_normalized_einop_contracts_one_composite_factor() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.arange(12).reshape(2, 6)
    rhs = np.arange(12).reshape(3, 4)
    op = einop(
        (ax[b, h * w], ax[w, d]),
        ax[b, h, d],
    ).with_sizes(h=2, w=3)
    expected = lhs.reshape(2, 2, 3) @ rhs

    np.testing.assert_array_equal(op(lhs, rhs), expected)


def test_layout_normalization_groups_literal_factor_without_permute() -> None:
    (h,) = axes("h")
    tensor = np.arange(6).reshape(3, 2)
    op = einop(ax[h, 2], ax[h * 2]).with_sizes(h=3)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(6))


def test_einop_identity_fast_path_routes_multiple_tensors() -> None:
    h, w = axes("h", "w")
    lhs = np.arange(2)
    rhs = np.arange(3)
    op = einop(
        (ax[h], ax[w]),
        (ax[h], ax[w]),
    )

    outputs = op(lhs, rhs)

    assert isinstance(outputs, tuple)
    assert outputs[0] is lhs
    assert outputs[1] is rhs
    assert op.plan_dict()["kind"] == "route"


def test_non_composite_einop_keeps_atomic_permute_path() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6).reshape(2, 3)
    op = einop(ax[h, w], ax[w, h])

    np.testing.assert_array_equal(op(tensor), tensor.T)
    assert op.plan_dict()["kind"] == "permute"


def test_non_composite_einop_keeps_atomic_contraction_path() -> None:
    b, h, d = axes("b", "h", "d")
    lhs = np.arange(6).reshape(2, 3)
    rhs = np.arange(12).reshape(3, 4)
    op = einop(
        (ax[b, h], ax[h, d]),
        ax[b, d],
    )

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ rhs)
    assert op.plan_dict()["kind"] == "einsum"


def test_layout_normalization_splits_additive_product_factor() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(20)
    op = einop(ax[(h + w) * d], ax[h + w, d]).with_sizes(h=2, w=3, d=4)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(5, 4))


def test_layout_normalization_transposes_additive_product_factor() -> None:
    h, w, d = axes("h", "w", "d")
    tensor = np.arange(20)
    op = einop(ax[(h + w) * d], ax[d, h + w]).with_sizes(h=2, w=3, d=4)

    np.testing.assert_array_equal(op(tensor), tensor.reshape(5, 4).T)


def test_layout_normalized_reducer_rejects_non_reduced_phase_axis() -> None:
    h, w, d = axes("h", "w", "d")

    with pytest.raises(ValidationError):
        einop(ax[h * w, d], ax[w * h]).reduce_by((ax[h], "sum"))


def test_layout_normalization_reorders_literal_only_product() -> None:
    tensor = np.arange(6)
    op = einop(ax[2 * 3], ax[3 * 2])
    reference = rearrange(ax[2 * 3], ax[3 * 2])

    np.testing.assert_array_equal(op(tensor), reference(tensor))


def test_layout_normalization_preserves_literal_product_association() -> None:
    (h,) = axes("h")
    tensor = np.arange(12)
    op = einop(ax[(2 * h) * 3], ax[2 * (h * 3)]).with_sizes(h=2)

    np.testing.assert_array_equal(op(tensor), tensor)


def test_layout_normalization_rejects_ambiguous_repeated_factor_reorder() -> None:
    h, w = axes("h", "w")

    with pytest.raises(ValidationError):
        einop(ax[h * w * h], ax[h * h * w])


def test_layout_normalized_contraction_handles_singleton_factors() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    lhs = np.arange(2).reshape(2, 1)
    rhs = np.arange(3).reshape(1, 3)
    op = einop(
        (ax[b, h * w], ax[w * h, d]),
        ax[b, d],
    ).with_sizes(h=1, w=1)

    np.testing.assert_array_equal(op(lhs, rhs), lhs @ rhs)


def test_layout_normalization_repeats_into_zero_sized_composite() -> None:
    (h,) = axes("h")
    tensor = np.arange(2)
    op = einop(ax[h], ax[0 * h]).with_sizes(h=2)

    np.testing.assert_array_equal(op(tensor), np.empty((0,), dtype=tensor.dtype))


def test_layout_normalized_einop_combines_reorder_and_scalar_output() -> None:
    h, w = axes("h", "w")
    tensor = np.arange(6)
    op = einop(
        ax[h * w],
        (ax[w * h], ax[()]),
    ).with_sizes(h=2, w=3)
    output_layout = rearrange(ax[h * w], ax[w * h]).with_sizes(h=2, w=3)

    outputs = op(tensor)

    assert isinstance(outputs, tuple)
    np.testing.assert_array_equal(outputs[0], output_layout(tensor))
    np.testing.assert_array_equal(outputs[1], tensor.sum())


def test_exact_identity_einop_rejects_reducer_configuration() -> None:
    (h,) = axes("h")

    with pytest.raises(ValidationError):
        einop(ax[h], ax[h]).reduce_by("sum")


def test_multi_input_layout_rejects_generated_repeated_logical_axes() -> None:
    b, h, w, d = axes("b", "h", "w", "d")

    with pytest.raises(ValidationError):
        einop(
            (ax[b, h * w * h], ax[h * h * w, d]),
            ax[b, d],
        )
