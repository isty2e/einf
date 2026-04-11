# Attention

Three self-contained variants: single-head, multi-head, and causal
multi-head. Each function below is complete — run it as-is. The
progressive build is in
[Attention, progressively](../getting-started/attention-progressive.md);
this page is the reference.

## Shared setup

```python
import numpy as np
from einf import ax, axes, einop, rearrange

b, t, s, d, dh, h = axes("b", "t", "s", "d", "dh", "h")

def softmax_over_last(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)
```

## Single-head attention

```python
def attention_single(x, Wq, Wk, Wv):
    project = einop((ax[b, t, d], ax[d, dh]), ax[b, t, dh])
    score   = einop((ax[b, t, dh], ax[b, s, dh]), ax[b, t, s])
    apply_  = einop((ax[b, t, s], ax[b, s, dh]), ax[b, t, dh])

    Q, K, V = project(x, Wq), project(x, Wk), project(x, Wv)
    scale = 1.0 / np.sqrt(Q.shape[-1])
    attn = softmax_over_last(score(Q, K) * scale)
    return apply_(attn, V)
```

### Quick test

```python
x  = np.random.randn(2, 5, 16).astype(np.float32)
Wq = np.random.randn(16, 8).astype(np.float32)
Wk = np.random.randn(16, 8).astype(np.float32)
Wv = np.random.randn(16, 8).astype(np.float32)

attention_single(x, Wq, Wk, Wv).shape
# (2, 5, 8)
```

## Multi-head attention

The only structural difference from the single-head recipe is layout:
split `d` into `(h * dh)`, keep the same contract over the per-head
dim, then merge back.

```python
def attention_multihead(x, Wq, Wk, Wv, Wo, n_heads):
    split_heads = rearrange(ax[b, t, (h * dh)], ax[b, h, t, dh]).with_sizes(h=n_heads)
    merge_heads = rearrange(ax[b, h, t, dh], ax[b, t, (h * dh)])

    project_io = einop((ax[b, t, d], ax[d, d]), ax[b, t, d])
    score_mh   = einop((ax[b, h, t, dh], ax[b, h, s, dh]), ax[b, h, t, s])
    apply_mh   = einop((ax[b, h, t, s], ax[b, h, s, dh]), ax[b, h, t, dh])

    Q = split_heads(project_io(x, Wq))
    K = split_heads(project_io(x, Wk))
    V = split_heads(project_io(x, Wv))

    raw  = score_mh(Q, K) / np.sqrt(Q.shape[-1])
    attn = softmax_over_last(raw)
    out  = apply_mh(attn, V)

    return project_io(merge_heads(out), Wo)
```

### Quick test

```python
x  = np.random.randn(2, 5, 32).astype(np.float32)
Wq = np.random.randn(32, 32).astype(np.float32)
Wk = np.random.randn(32, 32).astype(np.float32)
Wv = np.random.randn(32, 32).astype(np.float32)
Wo = np.random.randn(32, 32).astype(np.float32)

attention_multihead(x, Wq, Wk, Wv, Wo, n_heads=4).shape
# (2, 5, 32)
```

!!! tip "Hoist the ops"
    In real use, hoist `split_heads`, `merge_heads`, `project_io`,
    `score_mh`, and `apply_mh` out of the function body and pass them
    in (or make them module-level constants). This is the
    [first-class op](../concepts/tensorop-as-value.md) idiom — one
    planning pass covers every call.

## Causal multi-head attention

Same structure, one extra step: mask the upper triangle of the score
matrix to `-inf` before softmax so each query only attends to
positions at or before itself.

```python
def attention_causal(x, Wq, Wk, Wv, Wo, n_heads):
    split_heads = rearrange(ax[b, t, (h * dh)], ax[b, h, t, dh]).with_sizes(h=n_heads)
    merge_heads = rearrange(ax[b, h, t, dh], ax[b, t, (h * dh)])

    project_io = einop((ax[b, t, d], ax[d, d]), ax[b, t, d])
    score_mh   = einop((ax[b, h, t, dh], ax[b, h, s, dh]), ax[b, h, t, s])
    apply_mh   = einop((ax[b, h, t, s], ax[b, h, s, dh]), ax[b, h, t, dh])

    Q = split_heads(project_io(x, Wq))
    K = split_heads(project_io(x, Wk))
    V = split_heads(project_io(x, Wv))

    raw = score_mh(Q, K) / np.sqrt(Q.shape[-1])

    # causal mask: positions j > i are blocked for query i
    T = raw.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=np.bool_), k=1)
    raw = np.where(mask, -np.inf, raw)

    attn = softmax_over_last(raw)
    out  = apply_mh(attn, V)
    return project_io(merge_heads(out), Wo)
```

### Sanity check: causality holds

Perturb tokens after position 3 and verify the output at positions
`0..2` is unchanged:

```python
out  = attention_causal(x,  Wq, Wk, Wv, Wo, n_heads=4)

x2 = x.copy()
x2[:, 3:] = np.random.randn(*x2[:, 3:].shape).astype(np.float32)
out2 = attention_causal(x2, Wq, Wk, Wv, Wo, n_heads=4)

np.testing.assert_allclose(out[:, :3], out2[:, :3], atol=1e-5)
# passes — future tokens do not leak into past positions
```

## What each axis role is doing

Cross-referencing [axis roles](../concepts/axis-signatures.md):

| Op | kept | contracted | introduced |
| --- | --- | --- | --- |
| `project_io` | `b`, `t` | `d` (input feature dim) | — |
| `score_mh`   | `b`, `h`, `t`, `s` | `dh` | — |
| `apply_mh`   | `b`, `h`, `t`, `dh` | `s` | — |
| `split_heads` | `b`, `t`, `h`, `dh` | — | — (structural only) |
| `merge_heads` | `b`, `t`, `h`, `dh` | — | — (structural only) |

No axis is introduced in attention — every shape is derived from
inputs or bound via `.with_sizes(h=n_heads)`.

## Variants not shown here

- **Grouped-query attention** — `K` and `V` use fewer heads than `Q`.
  Change `score_mh`'s signature so `K` uses `hk` and replicate with a
  `repeat` on `hk -> h`.
- **Sliding window / block-diagonal masks** — replace the causal
  `np.triu` mask with whatever boolean pattern you need; the rest of
  the recipe is unchanged.
- **Rotary position embeddings (RoPE)** — apply a per-position rotation
  to `Q` and `K` before `score_mh`. RoPE is per-head-dim pairwise
  rotation; no `einf` op changes.
