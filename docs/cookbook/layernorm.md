# LayerNorm

LayerNorm is two phases: compute per-row stats along a feature axis,
then apply a per-feature affine. `einf` carries the axis-named
reductions and the broadcast multiply; NumPy fills in the subtract
and divide, which are not einsum-shaped operations.

## Shared setup

```python
import numpy as np
from einf import ax, axes, einop, reduce

b, t, d = axes("b", "t", "d")
```

## Recipe

```python
def layernorm(x, gamma, beta, eps=1e-5):
    mean_op = reduce(ax[b, t, d], ax[b, t]).reduce_by("mean")
    var_op  = reduce(ax[b, t, d], ax[b, t]).reduce_by(np.var)
    scale   = einop((ax[b, t, d], ax[d]), ax[b, t, d])

    mean = mean_op(x)[..., None]
    var  = var_op(x)[..., None]
    normed = (x - mean) / np.sqrt(var + eps)

    return scale(normed, gamma) + beta
```

### Quick test

```python
x     = np.random.randn(2, 5, 16).astype(np.float32)
gamma = np.random.randn(16).astype(np.float32)
beta  = np.random.randn(16).astype(np.float32)

out = layernorm(x, gamma, beta)
out.shape
# (2, 5, 16)
```

## RMSNorm variant

RMSNorm skips the mean-centering step: normalize by the root-mean-square
of the features instead. One reduce, one multiply.

```python
def rmsnorm(x, gamma, eps=1e-5):
    ms_op = reduce(ax[b, t, d], ax[b, t]).reduce_by("mean")
    scale = einop((ax[b, t, d], ax[d]), ax[b, t, d])

    ms = ms_op(x * x)[..., None]
    return scale(x / np.sqrt(ms + eps), gamma)
```

## Why name the normalized axis

Positional LayerNorm implementations normalize "the last axis". That is
fine until a call site reshapes upstream and the last axis silently
stops being the feature dim — the code still runs, producing garbage.

Here the feature axis is named `d`, and both reductions declare which
axis they remove:

```python
reduce(ax[b, t, d], ax[b, t])
```

If an upstream op changes the layout so `d` is no longer in the input
signature, construction fails loudly at call time rather than silently
reducing the wrong axis. This is the structural payoff of naming axes
by role (`d` = feature dim) rather than by position.

## What each op is doing

| Op | kept | reduced / contracted | reducer |
| --- | --- | --- | --- |
| `mean_op` | `b`, `t` | `d` | `"mean"` |
| `var_op`  | `b`, `t` | `d` | `np.var` |
| `scale`   | `b`, `t`, `d` | — | einop multiply, broadcast `gamma` over `(b, t)` |

The affine multiply is an `einop` rather than plain `*` so that the
broadcast of a feature-only weight `(d,)` against a full tensor
`(b, t, d)` is declared at the axis level — gamma lines up with the
`d` dim by name, not by trailing-axis alignment.

## Where `einf` stops

Subtract and divide are elementwise and do not have a meaningful
axis-signature beyond "same shape in, same shape out", so `einop`
does not express them. Falling out to NumPy for those steps is the
intended split: `einf` covers the reductions and broadcasts that
carry axis information; scalar arithmetic stays in the host array
library.

## Variants not shown here

- **GroupNorm** — split `d` into `(g * dg)` via a `rearrange`, reduce
  mean and variance over `dg` (leaving `g` kept), then merge back.
- **InstanceNorm / BatchNorm** — same reduction shape, different axis
  set. For BatchNorm, the reduced axes are `b` (and spatial axes if
  any); for InstanceNorm, only spatial axes — `b` stays kept.
- **Pre-norm vs post-norm** — structural, not algorithmic: call
  `layernorm` before or after the residual add. No `einf` change.
