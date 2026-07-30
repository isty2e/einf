import array_api_strict as xp
import pytest

from einf import (
    ErrorCode,
    ValidationError,
    ax,
    axes,
    contract,
    einop,
    rearrange,
    repeat,
    view,
)
from einf import reduce as einf_reduce


def test_standard_array_api_namespace_executes_layout_and_repeat() -> None:
    b, n, d = axes("b", "n", "d")
    tensor = xp.reshape(xp.arange(24, dtype=xp.float32), (2, 3, 4))

    transposed = rearrange(ax[b, n, d], ax[b, d, n])(tensor)
    assert bool(xp.all(transposed == xp.permute_dims(tensor, (0, 2, 1))))

    einop_routed = einop(ax[b, n, d], ax[b, n, d])(tensor)
    assert einop_routed is tensor

    einop_transposed = einop(ax[b, n, d], ax[b, d, n])(tensor)
    assert bool(xp.all(einop_transposed == xp.permute_dims(tensor, (0, 2, 1))))

    repeated = repeat(ax[b, n], ax[b, n, d]).with_sizes(d=4)(tensor[:, :, 0])
    expected_repeated = xp.broadcast_to(
        xp.expand_dims(tensor[:, :, 0], axis=2),
        (2, 3, 4),
    )
    assert bool(xp.all(repeated == expected_repeated))

    einop_repeated = einop(ax[b, n], ax[b, n, d]).with_sizes(d=4)(tensor[:, :, 0])
    assert bool(xp.all(einop_repeated == expected_repeated))


def test_standard_array_api_namespace_executes_rearrange_concat() -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange(
        (ax[b, n, d], ax[b, m, d]),
        ax[b, (n + m), d],
    ).with_sizes(n=2, m=3)
    left = xp.reshape(xp.arange(4, dtype=xp.float32), (1, 2, 2))
    right = xp.reshape(xp.arange(6, dtype=xp.float32), (1, 3, 2))

    result = op(left, right)
    einop_result = einop(
        (ax[b, n, d], ax[b, m, d]),
        ax[b, (n + m), d],
    ).with_sizes(n=2, m=3)(left, right)

    assert bool(xp.all(result == xp.concat((left, right), axis=1)))
    assert bool(xp.all(einop_result == xp.concat((left, right), axis=1)))


def test_standard_array_api_namespace_executes_named_reducers() -> None:
    b, n, d = axes("b", "n", "d")
    tensor = xp.reshape(xp.arange(1, 25, dtype=xp.float32), (2, 3, 4))
    base_op = einf_reduce(ax[b, n, d], ax[b, d])

    summed = base_op.reduce_by("sum")(tensor)
    maximized = base_op.reduce_by("max")(tensor)
    averaged = base_op.reduce_by("mean")(tensor)
    multiplied = base_op.reduce_by("prod")(tensor)
    einop_summed = einop(ax[b, n, d], ax[b, d]).reduce_by("sum")(tensor)

    assert bool(xp.all(summed == xp.sum(tensor, axis=1)))
    assert bool(xp.all(maximized == xp.max(tensor, axis=1)))
    assert bool(xp.all(averaged == xp.mean(tensor, axis=1)))
    assert bool(xp.all(multiplied == xp.prod(tensor, axis=1)))
    assert bool(xp.all(einop_summed == xp.sum(tensor, axis=1)))


def test_standard_array_api_namespace_rejects_strict_view() -> None:
    b, n, d = axes("b", "n", "d")
    tensor = xp.reshape(xp.arange(24, dtype=xp.float32), (2, 3, 4))

    with pytest.raises(ValidationError) as error:
        _ = view(ax[b, n, d], ax[b, d, n])(tensor)

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_standard_array_api_namespace_rejects_contract_without_einsum() -> None:
    b, n, d, o = axes("b", "n", "d", "o")
    lhs = xp.reshape(xp.arange(24, dtype=xp.float32), (2, 3, 4))
    rhs = xp.reshape(xp.arange(20, dtype=xp.float32), (4, 5))

    with pytest.raises(ValidationError) as error:
        _ = contract((ax[b, n, d], ax[d, o]), ax[b, n, o])(lhs, rhs)

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value


def test_standard_array_api_namespace_rejects_contraction_einop_without_einsum() -> (
    None
):
    b, n, d, o = axes("b", "n", "d", "o")
    lhs = xp.reshape(xp.arange(0, dtype=xp.float32), (2, 3, 0))
    rhs = xp.reshape(xp.arange(0, dtype=xp.float32), (0, 5))

    with pytest.raises(ValidationError) as error:
        _ = einop((ax[b, n, d], ax[d, o]), ax[b, n, o])(lhs, rhs)

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value
