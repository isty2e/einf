# Conv / pool (block patterns)

Pooling and stride-equal-kernel convolutions reduce to a single idiom:
unfold a strided axis into `(outer * window)`, then either reduce or
contract over the window. This page covers block pooling in 1D and
2D, plus block convolution.

Sliding windows with *overlapping* strides are out of scope for this
recipe — they need a windowed view (e.g. `np.lib.stride_tricks.sliding_window_view`)
before `einf` takes over.

## Shared setup

```python
import numpy as np
from einf import ax, axes, einop, rearrange, reduce

b, t, k, c = axes("b", "t", "k", "c")
h, w, h1, w1 = axes("h", "w", "h1", "w1")
(co,) = axes("co")
```

## 1D block pool

Window of size `k=4`, stride `k`. Split the long time axis into
`(t * k)`, then reduce the window axis.

```python
split_1d  = rearrange(ax[b, (t * k), c], ax[b, t, k, c]).with_sizes(k=4)
pool_mean = reduce(ax[b, t, k, c], ax[b, t, c]).reduce_by("mean")
pool_max  = reduce(ax[b, t, k, c], ax[b, t, c]).reduce_by("max")
```

### Quick test

```python
x = np.random.randn(2, 16, 8).astype(np.float32)
pool_mean(split_1d(x)).shape
# (2, 4, 8)
```

Mean and max pool share the exact same rearrange — only the reducer
changes. The plan for `split_1d` is shared across both pooling paths.

## 2D block pool

This is the einops idiom
`b c (h h1) (w w1) -> b c h w (h1 w1)` plus a reduce, rewritten as
a signature pair and a single reduction:

```python
split_2d  = rearrange(
    ax[b, c, (h * h1), (w * w1)], ax[b, c, h, w, h1, w1]
).with_sizes(h1=2, w1=2)
pool_mean_2d = reduce(
    ax[b, c, h, w, h1, w1], ax[b, c, h, w]
).reduce_by("mean")
```

### Quick test

```python
img = np.random.randn(2, 3, 8, 8).astype(np.float32)
pool_mean_2d(split_2d(img)).shape
# (2, 3, 4, 4)
```

Compared to the einops rewrite, the einf version keeps `h1` and `w1`
as separate kept axes in the intermediate signature, then reduces both
in one step. No fused `(h1 w1)` axis to flatten first.

## Strided conv as unfold + contract

A convolution with stride equal to kernel size is a contraction over
the window and channel axes after the same unfold:

```python
split_1d = rearrange(ax[b, (t * k), c], ax[b, t, k, c]).with_sizes(k=4)
conv_op  = einop((ax[b, t, k, c], ax[k, c, co]), ax[b, t, co])
```

### Quick test

```python
x = np.random.randn(2, 16, 8).astype(np.float32)    # b, t*k=16, c=8
W = np.random.randn(4, 8, 12).astype(np.float32)    # k=4, c=8, co=12

conv_op(split_1d(x), W).shape
# (2, 4, 12)
```

`k` and `c` both appear on each side of the input signature and are
absent from the output — they are contracted. `co` is introduced by
the weight. See [axis signatures](../concepts/axis-signatures.md) for
the role taxonomy.

## What each op is doing

| Op | kept | reduced / contracted | reducer |
| --- | --- | --- | --- |
| `split_1d`    | `b`, `t`, `k`, `c` | — | — (structural) |
| `split_2d`    | `b`, `c`, `h`, `w`, `h1`, `w1` | — | — (structural) |
| `pool_mean`   | `b`, `t`, `c` | `k` | `"mean"` |
| `pool_mean_2d`| `b`, `c`, `h`, `w` | `h1`, `w1` | `"mean"` |
| `conv_op`     | `b`, `t`, `co` | `k`, `c` | einop (contract) |

## Where this stops

- **Overlapping windows** (stride ≠ kernel). `einf` has no unfolding op
  with a stride parameter. Produce a windowed view first (NumPy:
  `sliding_window_view`; PyTorch: `unfold`), then the same
  `einop`/`reduce` ops work over the unfolded axis.
- **Valid / same / causal padding.** Padding is a boundary decision
  outside `einf`'s axis algebra. Pad the input first; the window ops
  stay the same.
- **Dilated convolutions.** Same story as overlapping windows — the
  unfold step is host-library territory.
