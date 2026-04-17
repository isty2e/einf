# Axis signatures

An axis signature is the user-facing contract of an `einf` operation.
Instead of positional shapes (`(B, N, D)`) and side-channel comments
about which dim means what, an op is defined in terms of named axes on
both sides of a transform. That lets `einf` check intent at definition
time and reuse the same op across calls whose concrete shapes differ.

## Named axes, not positional dims

```python
from einf import ax, axes, rearrange

b, n, d = axes("b", "n", "d")

transpose = rearrange(ax[b, n, d], ax[b, d, n])
```

Reading the signature: "take a tensor whose axes are named `b`, `n`,
`d`, and produce one whose axes are `b`, `d`, `n`". There is no
positional ambiguity — `d` cannot drift from one dim to another
between producer and consumer.

Compare with a positional transpose where the caller has to remember
that axis `1` and axis `2` were the ones to swap. Names scale; indices
do not.

## lhs and rhs

Every op takes two axis-term specifications:

- `lhs` — input layout. May be a single input or a tuple for
  multi-input ops.
- `rhs` — output layout. May be a single output or a tuple for
  multi-output ops (for example `rearrange` splits).

`ax[b, n, d]` builds an axis-term sequence. Tuples are used when
either side has multiple tensors:

```python
# one input, two outputs
(ax[b, n, d])         # lhs: one input
(ax[b, n], ax[b, d])  # rhs: two outputs
```

## Axis roles

Given an op, each axis has one of four roles. Roles are derived from
the signature, not declared separately — which is the point.

| Role | Appears on | Meaning |
| --- | --- | --- |
| kept | lhs and rhs | Passes through; its size is preserved. |
| reduced | lhs only | Collapsed away with a reducer. |
| contracted | multiple lhs inputs, not rhs | Summed over during contraction. |
| introduced | rhs only | Created by broadcast; must be sized via `with_sizes`. |

Examples of each role:

```python
from einf import ax, axes, reduce, einop, repeat

b, n, d, m, r = axes("b", "n", "d", "m", "r")

# kept: b, d     reduced: n
sum_n = reduce(ax[b, n, d], ax[b, d])

# kept: b, n, m  contracted: d
matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

# kept: b, n     introduced: r (needs with_sizes)
broadcast = repeat(ax[b, n], ax[b, n, r]).with_sizes(r=4)
```

If a role that needs a size binding (introduced axis) is missing one,
the op will tell you exactly which axis is unbound rather than failing
inside the backend call.

## Fused axes

An axis term can be a composite of named axes, which is how `einf`
expresses split and merge without new operators.

### Product: merged / split axis

```python
from einf import view

b, h, w, c = axes("b", "h", "w", "c")

# One dim on the lhs whose size is h * w, split into two dims on rhs.
split = view(ax[b, (h * w), c], ax[b, h, w, c]).with_sizes(h=2)

import numpy as np
x = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
split(x).shape  # (2, 2, 3, 3), where w = 3 is inferred
```

Only `h` needs to be explicit — `w` is solved from the input.

### Sum: concatenated / tuple-split axis

```python
from einf import rearrange

b, n, m, d = axes("b", "n", "m", "d")

split_two = rearrange(
    ax[b, (n + m), d],
    (ax[b, n, d], ax[b, m, d]),
).with_sizes(n=1, m=2)
```

`(n + m)` says "this axis has size `n + m`". The tuple output tells
`einf` to split it into two tensors of the corresponding sizes.

## Packs — variadic axes

A `pack` matches zero or more axes. Use packs when the signature
should stay valid across different tensor ranks.

```python
from einf import ax, axes, packs, rearrange

(b,) = axes("b")
(tail,) = packs("tail")

move_b_to_last = rearrange(ax[b, tail], ax[tail, b])

import numpy as np
move_b_to_last(np.zeros((2, 3, 4, 5), dtype=np.float32)).shape  # (3, 4, 5, 2)
move_b_to_last(np.zeros((2,), dtype=np.float32)).shape          # (2,)
```

The same op works on rank-1 and rank-4 inputs because `tail` is
variadic.

## What the signature buys

- **Intent checked at definition time.** Wrong roles (for example, an
  axis on the right that is not kept, introduced, or derivable) fail
  at construction, not inside the backend.
- **Call-site shape inference.** Given the input tensor shape and any
  `with_sizes` bindings, `einf` solves the remaining sizes
  automatically.
- **Reuse across shapes.** The same op definition is valid for any
  input shape consistent with the signature — see
  [TensorOp as a value](tensorop-as-value.md) for the reuse story.
