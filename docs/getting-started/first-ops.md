# First ops

This page walks the six core operations — `view`, `rearrange`, `repeat`,
`reduce`, `contract`, `einop` — on small concrete NumPy examples. The same
`TensorOp` definitions can run on Array API-compatible inputs when their
namespace satisfies the operation's capability requirements. Tensor
construction and backend callables must use that implementation's equivalents.
See the [runtime backend matrix](install.md#runtime-backend-support) for the
exact contract.

## Setup

```python
import numpy as np
from einf import ax, axes, view, rearrange, repeat, reduce, contract, einop

b, c, n, m, d, h, w = axes("b", "c", "n", "m", "d", "h", "w")
```

`axes(...)` returns named axis placeholders. `ax[...]` builds an axis
term sequence — roughly "here is a shape, named". A lhs / rhs pair of
axis terms is all you need to describe an operation.

## view — structural reinterpretation, zero copy

`view` maps a layout to another layout without moving data. It is
strict: if the requested mapping cannot be expressed as a view of the
input's strides, the call fails with `NOT_A_VIEW` rather than silently
copying.

```python
split_hw = view(ax[b, (h * w), c], ax[b, h, w, c]).with_sizes(h=2)

x = np.arange(3 * 6 * 4, dtype=np.float32).reshape(3, 6, 4)
y = split_hw(x)
# y.shape == (3, 2, 3, 4)
# y and x share memory
```

The `(h * w)` on the left side says "an axis whose size is the product
of named axes `h` and `w`". `.with_sizes(h=2)` binds `h`; `w` is
inferred from the input shape.

## rearrange — move axes, may copy

`rearrange` permutes, splits, and merges axes. Unlike `view`, it is
free to do the work it needs to produce the requested layout.

```python
transpose = rearrange(ax[b, n, d], ax[b, d, n])
y = transpose(np.zeros((2, 3, 4), dtype=np.float32))
# y.shape == (2, 4, 3)
```

`rearrange` also supports tuple outputs for splits:

```python
split_two = rearrange(
    ax[b, (n + m), d],
    (ax[b, n, d], ax[b, m, d]),
).with_sizes(n=1, m=2)

left, right = split_two(np.zeros((2, 3, 4), dtype=np.float32))
# left.shape  == (2, 1, 4)
# right.shape == (2, 2, 4)
```

Here `(n + m)` is a sum-axis: its size is the sum of `n` and `m`, and
the output tuple splits the input into those two pieces.

## repeat — broadcast along a new axis

`repeat` introduces one or more axes not present on the input.

```python
broadcast_r = repeat(ax[b, c], ax[b, c, n]).with_sizes(n=3)

y = broadcast_r(np.ones((2, 4), dtype=np.float32))
# y.shape == (2, 4, 3)
```

Any new axis must have its size bound via `with_sizes(...)` before the
op is callable. Without the binding, `einf` will tell you exactly
which axis is missing.

## reduce — collapse an axis

`reduce` removes axes. The default reducer is sum.

```python
sum_over_n = reduce(ax[b, n, d], ax[b, d])

y = sum_over_n(np.ones((2, 3, 4), dtype=np.float32))
# y.shape == (2, 4); every element equals 3.0
```

Pick a different reducer with `.reduce_by(...)`:

```python
max_over_n = reduce(ax[b, n, d], ax[b, d]).reduce_by(np.max)

y = max_over_n(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
# y.shape == (2, 4)
```

Reducers may be named strings (`"sum"`, `"max"`, `"mean"`, `"prod"`) or
callables that reduce along a given axis. Named reducers dispatch to the active
Array API namespace. A callable receives that namespace's tensor directly, so
the `np.max` example above is NumPy-specific; use a callable implemented with
the active namespace's operations if the same `TensorOp` must run across
implementations.

## contract — pure tensor contraction

`contract` sums over axes that appear on both inputs and not on the
output. It is the building block of dot products, matrix multiplies,
and bilinear forms.

```python
dot = contract((ax[b, n, d], ax[b, n, d]), ax[b, n])

x = np.ones((2, 3, 4), dtype=np.float32)
y = dot(x, x)
# y.shape == (2, 3); every element equals 4.0 (== d)
```

`b` and `n` appear on both sides and on the output — they are kept.
`d` appears on both inputs but not on the output — it is contracted.

## einop — the general operation

`einop` is the superset: it combines multiple inputs, supports any
axis role (kept, contracted, introduced), and accepts a reducer. Most
high-level operations (attention, layernorm, conv unfolding) are one
`einop` call.

```python
matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

y = matmul(
    np.zeros((2, 3, 4), dtype=np.float32),
    np.zeros((4, 5), dtype=np.float32),
)
# y.shape == (2, 3, 5)
```

## Reuse: define once, call many

The point of making ops first-class values is that a single definition
can be reused across calls with different input shapes. The planning
work happens when you build the op; every call reuses that plan.

```python
matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

y1 = matmul(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
y2 = matmul(np.zeros((7, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
# y1.shape == (2, 3, 5)
# y2.shape == (7, 3, 5)
```

This is the idiomatic usage pattern: hoist op construction out of hot
loops and reuse the op across calls.

```python
# good: one-time construction
REDUCE_BD = reduce(ax[b, n, d], ax[b, d])

def forward(x):
    return REDUCE_BD(x)

# avoid: cold construction per call
def slow_forward(x):
    return reduce(ax[b, n, d], ax[b, d])(x)
```

Next: [Attention, progressively](attention-progressive.md) builds
attention step by step and shows the same op being reused across
different batch and head shapes.
