import numpy as np
from numpy.typing import NDArray

from einf import ax, axes, einop, rearrange, reduce


def test_parity_guardrail_rearrange_split_roundtrip() -> None:
    b, h, w, d = axes("b", "h", "w", "d")
    flatten = rearrange(ax[b, h, w, d], ax[b, (h * w), d])
    split = rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=3, w=4)

    x: NDArray[np.float32] = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(
        2, 3, 4, 5
    )
    flattened = flatten(x)
    restored = split(flattened)

    np.testing.assert_array_equal(flattened, x.reshape(2, 12, 5))
    np.testing.assert_array_equal(restored, x)


def test_parity_guardrail_reduce_ordered_phases() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "prod"))

    x: NDArray[np.float32] = np.arange(1, (2 * 3 * 4) + 1, dtype=np.float32).reshape(
        2, 3, 4
    )
    result = op(x)
    expected = np.prod(np.sum(x, axis=1), axis=1)
    np.testing.assert_allclose(result, expected)


def test_parity_guardrail_einop_contract_split() -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=3, r=4)

    lhs: NDArray[np.float32] = np.arange(
        2 * ((2 + 3) * 4) * 6, dtype=np.float32
    ).reshape(2, 20, 6)
    rhs: NDArray[np.float32] = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    out_h1, out_h2 = op(lhs, rhs)

    carrier = np.einsum("bqn,nd->bqd", lhs, rhs)
    expected_h1 = carrier[:, : 2 * 4, :]
    expected_h2 = carrier[:, 2 * 4 :, :]
    np.testing.assert_allclose(out_h1, expected_h1)
    np.testing.assert_allclose(out_h2, expected_h2)
