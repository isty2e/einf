from collections.abc import Callable

from einf import TensorOp, ax, axes, rearrange

b, n = axes("b", "n")
exact = rearrange(ax[b, n], ax[n, b])
public: TensorOp = exact
callable_op: Callable[..., object] = public

name: str = public.name
sizes: dict[str, int] = public.sizes
sizes_items: tuple[tuple[str, int], ...] = public.sizes_items
supports_reducer: bool = public.supports_reducer
