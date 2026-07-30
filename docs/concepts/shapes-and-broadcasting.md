# Shapes and broadcasting

`einf` defers concrete sizes from op construction to call time. This
page covers the three ways sizes become known, what "shape-free"
execution means for reuse, and how broadcasting appears in a named-axis
world.

## Where sizes come from

An op's signature says *which* axes are there. It does not say how big
each one is. At call time, `einf` resolves each axis size from one of
three sources.

### 1. Solved from input shape

For any axis that appears on at least one input, the solver reads its
size from the corresponding dim.

```python
import numpy as np
from einf import ax, axes, reduce

b, n, d = axes("b", "n", "d")
sum_n = reduce(ax[b, n, d], ax[b, d])

sum_n(np.ones((2, 3, 4), dtype=np.float32)).shape
# (2, 4) — b, n, d all solved from input shape
```

### 2. Explicit via `with_sizes`

Some axes are not directly recoverable from the input shape:

- **Introduced axes** — only appear on the rhs, so there is no input
  dim to read them from.
- **Fused axes with ambiguous factors** — `(h * w)` has many valid
  `(h, w)` pairs for a given product; at least one needs to be fixed.

```python
from einf import repeat, view

# introduced axis r: must be bound
broadcast = repeat(ax[b, n], ax[b, n, r]).with_sizes(r=4)

# ambiguous product: bind one factor, the other is solved
split = view(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=3)
```

You do not need to bind every axis — only the ones that are not
otherwise determined. `einf` tells you exactly which axis is missing
if you forget.

### 3. Derived from other bindings

If `h` is bound and the input dim is `h * w`, then `w` is fixed. The
solver does this automatically, so you only hand-bind what's
structurally underdetermined.

## What happens when sizes are missing or conflict

`einf` fails at call time (before reaching the backend) with a clear
structured error:

- missing introduced axis → `"missing size binding: <axis>"`
- ambiguous fused axis → `ValidationError` with the reason,
- bound size inconsistent with input shape → `ValidationError`.

Size errors are diagnostics, not backend exceptions. You can catch them
and inspect `error.code` against `ErrorCode`.

## Shape-free execution

Many plans do not actually depend on the concrete sizes of kept axes.
For those plans, `einf` executes the same compiled chain across calls
with different sizes — no re-specialization.

```python
import numpy as np
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")
matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

matmul(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32)).shape
# (2, 3, 5)
matmul(np.zeros((7, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32)).shape
# (7, 3, 5) — different batch, same plan
```

Plans that *do* depend on concrete sizes (for example, a `view` that
requires specific stride alignment) specialize per-call-site and cache
the specialization. You rarely have to think about which regime a plan
is in — the reuse just happens.

## Cross-backend execution

The same op can transfer across Array API-compatible implementations without
redefinition. The runtime picks the namespace from the input tensor's type and
checks the capabilities required by the operation. Portable `rearrange`,
`repeat`, named `reduce`, and non-contraction `einop` behavior is tested against
`array-api-strict`; NumPy and PyTorch additionally have full named-backend CI
coverage. See the
[runtime backend matrix](../getting-started/install.md#runtime-backend-support).

```python
import numpy as np
import torch
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")
matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

matmul(np.zeros((2, 3, 4), dtype=np.float32),
       np.zeros((4, 5),    dtype=np.float32)).shape       # numpy
matmul(torch.zeros(2, 3, 4),
       torch.zeros(4, 5)).shape                           # torch
```

## Broadcasting, in a named-axis world

NumPy-style broadcasting works on aligned trailing dims and inserts
size-1 axes implicitly. In `einf`, broadcasting is explicit: it happens
through named axes that are introduced on the rhs.

```python
from einf import ax, axes, repeat

b, c, n = axes("b", "c", "n")

# Broadcast (b, c) to (b, c, n) by introducing n.
broadcast = repeat(ax[b, c], ax[b, c, n]).with_sizes(n=5)

import numpy as np
broadcast(np.ones((2, 3), dtype=np.float32)).shape  # (2, 3, 5)
```

`einop` handles cross-input broadcasting through the same named-axis
mechanism:

```python
from einf import ax, axes, einop

b, c, n = axes("b", "c", "n")

# A per-batch, per-channel scale multiplied into every n step.
apply = einop((ax[b, c, n], ax[b, c]), ax[b, c, n])

x  = np.ones((2, 3, 4), dtype=np.float32)
sc = np.full((2, 3), 2.0, dtype=np.float32)
apply(x, sc).shape  # (2, 3, 4); each element doubled
```

The axes that appear on one input but not another are the ones that
broadcast. There is no silent rank inference — the signature is the
contract.

## Pitfalls

- **Mixing explicit and implicit sizes.** If you pass
  `with_sizes(h=3)` but the input shape already fixes `h` to a
  different value, you get a `ValidationError`. This is preferable to
  silent reshape bugs — just remove the redundant binding.
- **Introduced axis without a binding.** This is the single most
  common beginner mistake on `repeat` and multi-output `rearrange`.
  The error message names the axis.
- **Assuming NumPy broadcasting.** `einop` will *not* silently
  broadcast size-1 dims into named axes it does not know about. Add
  the axis to the signature and `einf` will handle it; don't rely on
  implicit rank promotion.
