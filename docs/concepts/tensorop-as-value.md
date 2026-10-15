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

Structural planning and lowering happen when the op is built. Calls
reuse that plan, but may still need runtime specialization for their
input shapes and backend.

Runtime specialization uses bounded caches. A retained entry is reused;
an evicted entry must be built again when needed. A shape stream larger
than the cache can therefore cause repeated specialization. Concurrent
first calls may also build the same missing entry independently: cold
construction is not guaranteed to happen exactly once per key.

```python
import numpy as np
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")

matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

y1 = matmul(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
y2 = matmul(np.zeros((7, 3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
# Same op, two different batch sizes, one planning pass.
```

Keeping the op alive avoids repeated structural planning. Runtime cache
hits also avoid rebuilding specialized steps and runners.

## Contrast with string-pattern libraries

Libraries like `einops` and `einx` express ops as format strings
that are parsed on every call (caches help, but the unit of
compilation is the string plus the specialization to this call's
shapes).

`einf` flips the unit:

| Concern | string-pattern libraries | `einf` |
| --- | --- | --- |
| Unit of compilation | call site | op value |
| Shape variance | parsed or keyed per call | runtime specializations under one plan |
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

## Annotating an op

Use `TensorOp[[T], R]` when a function or variable needs a specific
call signature. The list describes positional tensor inputs; the result
type describes one tensor or a tuple of outputs.

```python
from typing import TypeVar

from einf import ReducibleTensorOp, TensorLike, TensorOp

T = TypeVar("T", bound=TensorLike)

def apply(op: TensorOp[[T], T], tensor: T) -> T:
    return op(tensor)

def apply_binary(op: TensorOp[[T, T], T], left: T, right: T) -> T:
    return op(left, right)

def split(op: TensorOp[[T], tuple[T, T]], tensor: T) -> tuple[T, T]:
    return op(tensor)

def apply_mean(op: ReducibleTensorOp[[T], T], tensor: T) -> T:
    return op.reduce_by("mean")(tensor)
```

Both annotations retain read-only operation metadata, `plan()`,
`plan_dict()`, and `with_sizes()`. `ReducibleTensorOp` also retains
`reduce_by()`. Configuration preserves the annotated call signature.
An op can still be passed where a matching `Callable` is required.

Leave constructor results unannotated when inference is enough. A bare
`TensorOp` remains useful for metadata-only code, but its call result is
`object`, not a specific tensor type.

These are static annotations, not runtime wrappers or additional tensor
checks. `ReducibleTensorOp` is an alias of `TensorOp` at runtime, so
`isinstance(op, ReducibleTensorOp)` does not test reducer support. Use
`op.supports_reducer` for that query.

Pyrefly is the primary checker for explicit operation annotations; the
installed-wheel checks also verify them with mypy. Pyright/BasedPyright
can reject assignment of an inferred, generic constructor result to a
parameterized `TensorOp` protocol. That matching boundary is not supported
by those checkers here; their constructor inference, direct-call, and bare
`TensorOp` coverage remain in place.

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
