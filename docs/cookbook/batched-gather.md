# Batched gather

Integer-index gather is not an `einf` op — it does not have an
axis-algebraic form. But gather almost always sits in a chain with
reductions and contractions, and that is where named axes pay off:
once the gathered tensor has a signature, the ops downstream compose
exactly as in any other recipe.

This page walks the two shape regimes (unbatched, batched) and closes
with a pure-einf alternative via one-hot × contract.

## Shared setup

```python
import numpy as np
from einf import ax, axes, einop, reduce

v, d, b, t, k = axes("v", "d", "b", "t", "k")
```

## Unbatched gather (embedding lookup)

```python
W   = np.random.randn(100, 16).astype(np.float32)   # (V=100, d=16)
ids = np.array([3, 7, 42, 11, 5])                   # (t=5,)

emb = W[ids]                                        # (t, d)
```

The gather step is fancy indexing — host-library territory. What
follows is einf again: treat `emb` as a `(t, d)` tensor and reduce
over `t` to get a single vector:

```python
pool = reduce(ax[t, d], ax[d]).reduce_by("mean")
sent_repr = pool(emb)
# sent_repr.shape == (16,)
```

## Batched gather

When the index array itself has a batch dim, fancy indexing broadcasts
the gather across it:

```python
W   = np.random.randn(100, 16).astype(np.float32)   # (V, d)
ids = np.random.randint(0, 100, size=(4, 7))        # (b, t)

emb = W[ids]                                        # (b, t, d)
```

No einf in the gather, but the result carries a `(b, t, d)` signature
and the next step is a named-axis einop. For example, score each
token against a per-batch query vector:

```python
q     = np.random.randn(4, 16).astype(np.float32)   # (b, d)
score = einop((ax[b, t, d], ax[b, d]), ax[b, t])

scores = score(emb, q)
# scores.shape == (4, 7)
```

`d` is contracted, `b` and `t` are kept. The axis roles are declared
by the signature — there is no "axis=-1" ambiguity about which
dimension is the contraction dim.

## Per-batch row-gather

When each batch row needs to pick different positions from *its own*
matrix, the index tensor has shape `(b, k)` and is paired with an
explicit batch-index broadcast:

```python
x   = np.random.randn(3, 20, 8).astype(np.float32)  # (b, n, d)
idx = np.random.randint(0, 20, size=(3, 4))         # (b, k)

bidx = np.arange(3)[:, None]          # (b, 1) — lines up against idx
y    = x[bidx, idx]                   # (b, k, d)
```

Downstream einf is unchanged by the gather:

```python
pool = reduce(ax[b, k, d], ax[b, d]).reduce_by("mean")
pool(y).shape
# (3, 8)
```

## Pure-einf alternative: one-hot × contract

For backends without a first-class gather (or code paths where you
need every step to be an `einf` op), you can reach the same result by
multiplying a one-hot matrix against the embedding table:

```python
W   = np.random.randn(50, 16).astype(np.float32)   # (V, d)
ids = np.random.randint(0, 50, size=(2, 5))        # (b, t)

one_hot = np.eye(50, dtype=np.float32)[ids]        # (b, t, V)
lookup  = einop((ax[b, t, v], ax[v, d]), ax[b, t, d])

emb = lookup(one_hot, W)
# emb.shape == (2, 5, 16)
```

This is a contract over `V` — the same shape as an embedding lookup,
expressed as a dense matmul. It is slower than fancy indexing for a
real vocabulary, but it runs anywhere a contraction runs, and its
axis signature is explicit.

## Axis-role summary

| Op | kept | contracted / reduced | notes |
| --- | --- | --- | --- |
| `score` | `b`, `t` | `d` | query broadcast along `t` |
| `pool` (gather) | `d` | `t` | unbatched mean |
| `pool` (batched) | `b`, `d` | `k` | batched mean over gathered rows |
| `lookup` (one-hot) | `b`, `t`, `d` | `v` | dense alternative to fancy indexing |

## What stays outside einf

- **The gather itself** — integer-array indexing is not an einsum-shape
  operation. `einf` picks up the result once the gathered tensor has
  a named-axis signature.
- **Scatter / segment sum** — even more positional; often expressible
  as a masked sum or a one-hot contract but not as a single einf op.
- **Variable-length ragged gathers** — the output shape is not derivable
  from input shapes alone, so there is no plan-level signature.
