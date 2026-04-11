# Attention, progressively

This tutorial builds scaled dot-product attention in stages. Each stage
adds one new mechanic, and the code you write in one stage is reused
unchanged in the next. The closing stage shows the same `TensorOp`
instances running on several different batch and head shapes without
re-definition — the concrete payoff of first-class ops.

Assumed: you've read [first ops](first-ops.md) and know the four
axis roles.

## Setup

```python
import numpy as np
from einf import ax, axes, einop, rearrange

b, t, s, d, dh, h = axes("b", "t", "s", "d", "dh", "h")
# b  = batch
# t  = query positions
# s  = key/value positions
# d  = model dimension
# dh = per-head dimension
# h  = number of heads
```

`einf` does not ship a softmax op — softmax needs a max-subtract plus
a normalized sum, which is more than one step. We will use a plain
NumPy helper for it and keep the attention structure inside `einf`:

```python
def softmax_over_last(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)
```

## Stage 1 — project input to Q, K, V

A projection multiplies a `(b, t, d)` input by a `(d, dh)` weight and
produces `(b, t, dh)`. One `einop` captures the contract over `d`:

```python
project = einop((ax[b, t, d], ax[d, dh]), ax[b, t, dh])

x  = np.random.randn(2, 5, 16).astype(np.float32)
Wq = np.random.randn(16, 8).astype(np.float32)
Wk = np.random.randn(16, 8).astype(np.float32)
Wv = np.random.randn(16, 8).astype(np.float32)

Q = project(x, Wq)
K = project(x, Wk)
V = project(x, Wv)
# Q.shape == K.shape == V.shape == (2, 5, 8)
```

This is the reuse hint for later: one `project` op is enough for all
three weight matrices. Nothing forces you to rebuild it.

## Stage 2 — scores via QK contract

Attention scores contract the per-head dim `dh` between queries and
keys, leaving `(b, t, s)` — how much each query position attends to
each key position:

```python
score = einop((ax[b, t, dh], ax[b, s, dh]), ax[b, t, s])

scores = score(Q, K)
# scores.shape == (2, 5, 5)
```

`dh` appears on both inputs and not on the output — it is contracted.
`b`, `t`, `s` are kept. This is exactly the role taxonomy from
[axis signatures](../concepts/axis-signatures.md).

## Stage 3 — softmax, then apply to V

Scale, softmax, and contract the attention weights with `V`. The
softmax step is NumPy; the before-and-after are `einf`.

```python
attn = softmax_over_last(scores / np.sqrt(8.0))

apply_attn = einop((ax[b, t, s], ax[b, s, dh]), ax[b, t, dh])

out = apply_attn(attn, V)
# out.shape == (2, 5, 8)
```

That is single-head attention, end to end.

## Stage 4 — multi-head via rearrange

Going from single-head to multi-head is a layout change, not a
different algorithm. Split the model dim `d` into `(h * dh)` and lift
`h` to its own axis:

```python
split_heads = rearrange(
    ax[b, t, (h * dh)],
    ax[b, h, t, dh],
).with_sizes(h=4)

x_big = np.random.randn(2, 5, 32).astype(np.float32)  # d = 32 = 4 * 8
per_head = split_heads(x_big)
# per_head.shape == (2, 4, 5, 8)
```

The score and apply ops lift the same way — add `h` to every
signature, and the contract over `dh` still works:

```python
score_mh = einop(
    (ax[b, h, t, dh], ax[b, h, s, dh]),
    ax[b, h, t, s],
)
apply_mh = einop(
    (ax[b, h, t, s], ax[b, h, s, dh]),
    ax[b, h, t, dh],
)
```

Run the full block:

```python
Q4 = split_heads(x_big)
K4 = split_heads(x_big)
V4 = split_heads(x_big)

raw  = score_mh(Q4, K4)
attn = softmax_over_last(raw / np.sqrt(8.0))
out  = apply_mh(attn, V4)
# out.shape == (2, 4, 5, 8)
```

Merge heads back for the final projection:

```python
merge_heads = rearrange(ax[b, h, t, dh], ax[b, t, (h * dh)])

final = merge_heads(out)
# final.shape == (2, 5, 32)
```

## Stage 5 — reuse across shapes

Here is the payoff. The `score_mh` op defined above is a `TensorOp`
value. It carries its own compiled plan. You can call it with any
input shapes that satisfy the signature — different batch sizes,
different head counts, different sequence lengths — and the plan is
reused every time.

```python
# tiny: 1 batch, 2 heads, 4 tokens, per-head dim 8
Q_small = np.random.randn(1, 2, 4, 8).astype(np.float32)
K_small = np.random.randn(1, 2, 4, 8).astype(np.float32)
score_mh(Q_small, K_small).shape
# (1, 2, 4, 4)

# large: 8 batch, 16 heads, 64 tokens, per-head dim 8
Q_big = np.random.randn(8, 16, 64, 8).astype(np.float32)
K_big = np.random.randn(8, 16, 64, 8).astype(np.float32)
score_mh(Q_big, K_big).shape
# (8, 16, 64, 64)
```

No re-parsing, no re-lowering, no re-specialization of the structural
plan. In a training loop or a serving hot path, this is the difference
between paying planning cost once per model and paying it once per
batch.

## Where to go next

- [Cookbook: attention](../cookbook/attention.md) completes the
  single-head / multi-head / causal-mask variants as self-contained
  recipes.
- [Concepts: shapes and broadcasting](../concepts/shapes-and-broadcasting.md)
  explains why the same op can run across shapes with no
  re-specialization.
- [Concepts: TensorOp as a value](../concepts/tensorop-as-value.md)
  goes deeper on what "first-class op" buys you structurally.
