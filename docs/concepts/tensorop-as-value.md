# TensorOp as a value

The thing a constructor like `rearrange(...)` returns is not a syntactic
pattern to be re-parsed on every call. It is a `TensorOp` — a first-class
value that carries a compiled execution plan and can be reused across
calls with varying input shapes.

This page is the user-facing view of that choice. The internal story
lives in [internals/design-rationale](../internals/design-rationale.md).

## What you get back

```python
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")

matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])
# matmul is a TensorOp. You can call it, introspect its plan, and
# bind sizes on a copy. It is not a string.
```

A `TensorOp` carries:

1. a normalized signature (lhs / rhs axis terms),
2. a compiled plan describing how the op is actually executed,
3. any user-supplied policy — `.with_sizes(...)`, `.reduce_by(...)`,
   backend hints.

## Calls reuse the plan

Planning and lowering happen when the op is built. Subsequent calls
go through the cached plan; they do not re-parse, re-lower, or
re-search. That is the warm-path contract.

```python
import numpy as np
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")

matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

y1 = matmul(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
y2 = matmul(np.zeros((7, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
# Same op, two different batch sizes, one planning pass.
```

This is how `einf` can compete on warm-path cost with libraries that
compile per call: the compilation happens once, then stays out of
your way.

## Contrast with string-pattern libraries

Libraries like `einops` and `einx` express ops as format strings
that are parsed on every call (caches help, but the unit of
compilation is the string plus the specialization to this call's
shapes).

`einf` flips the unit:

| Concern | string-pattern libraries | `einf` |
| --- | --- | --- |
| Unit of compilation | call site | op value |
| Shape variance | parsed or keyed per call | absorbed by one plan |
| Hoisting out of loops | convention | idiomatic — the type encourages it |

The convention in a string library is to cache the parsed pattern
behind the call. The type in `einf` *is* the cached plan. You do not
opt in; you just keep a reference.

## Idiomatic pattern: hoist the op

```python
# good: one-time construction
REDUCE_BD = reduce(ax[b, n, d], ax[b, d])

def forward(x):
    return REDUCE_BD(x)

# avoid: cold construction per call
def slow_forward(x):
    return reduce(ax[b, n, d], ax[b, d])(x)
```

`einf` also maintains bounded constructor caches for identical op
specs, including unchanged `with_sizes(...)` and `reduce_by(...)`
configurations. That makes the slow form less expensive than it looks,
but a cache miss still does cold-path work. Hoist the op when you can.

## Configuring an op

`TensorOp` methods return new ops rather than mutating in place. This
makes variants cheap and safe to share.

```python
import numpy as np
from einf import ax, axes, reduce

b, n, d = axes("b", "n", "d")

base = reduce(ax[b, n, d], ax[b, d])       # sum by default
mean = base.reduce_by("mean")              # a different TensorOp
mx   = base.reduce_by(np.max)              # another one
```

`with_sizes` and `reduce_by` compose; each returns a fully-formed op
with its own compiled plan.

## What it unlocks

- **Pass ops as arguments.** A transformer block can take its
  attention op as a parameter and get different attention variants
  without string templating.
- **Plan introspection.** Because the op owns the plan, you can ask
  it what it will do (`op.plan()`, `op.plan_dict()`) without running
  anything. Useful for debugging, static analysis, and benchmarking.
- **Structural testing.** Unit tests can assert on plan shape
  ("there is exactly one einsum step"), not just output values.
- **Shape-free execution.** Plans that do not depend on concrete
  sizes can execute without re-specialization across calls — see
  [Shapes and broadcasting](shapes-and-broadcasting.md).
