import pytest

from einf import ax, axes, contract, rearrange, repeat
from einf import reduce as einf_reduce


def test_jax_uses_array_namespace_fallback() -> None:
    jnp = pytest.importorskip("jax.numpy")
    batch, width, copy = axes("batch", "width", "copy")
    tensor = jnp.arange(6).reshape(2, 3)

    transposed = rearrange(ax[batch, width], ax[width, batch])(tensor)
    repeated = repeat(ax[batch, width], ax[batch, width, copy]).with_sizes(copy=2)(
        tensor
    )
    reduced = einf_reduce(ax[batch, width], ax[batch]).reduce_by("sum")(tensor)
    contracted = contract(
        (ax[batch, width], ax[width, copy]),
        ax[batch, copy],
    )(tensor, jnp.arange(6).reshape(3, 2))

    assert transposed.shape == (3, 2)
    assert repeated.shape == (2, 3, 2)
    assert reduced.shape == (2,)
    assert contracted.shape == (2, 2)
