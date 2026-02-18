# einf

`einf` is a tensor DSL with axis signatures and first-class `TensorOp` objects.

You define an operation once (for example `rearrange`, `reduce`, `contract`, `einop`) and reuse it across calls.

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

Python: `>=3.10`

## Quick Start

```python
import numpy as np
from einf import ax, axes, rearrange, reduce, einop

b, n, d, m = axes("b", "n", "d", "m")
h, w, r, j = axes("h", "w", "r", "j")

# 1) Rearrange
transpose = rearrange(ax[b, n, d], ax[b, d, n])
y = transpose(np.zeros((2, 3, 4), dtype=np.float32))

# 2) Split with explicit sizes
split_dim1 = rearrange(
    ax[b, (n + m), d],
    (ax[b, n, d], ax[b, m, d]),
).with_sizes(n=1, m=2)
splitted_1, splitted_2 = split_dim1(np.zeros((2, 3, 4), dtype=np.float32))

# 3) Reduce (default reducer is sum)
reduce_dim1 = reduce(ax[b, n, d], ax[b, d])
r = reduce_dim1(np.zeros((2, 3, 4), dtype=np.float32))

# 4) Generic einop
matmul_like = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])
out = matmul_like(
    np.zeros((2, 3, 4), dtype=np.float32),
    np.zeros((4, 5), dtype=np.float32),
)

# 5) Generic einop: contract + split outputs (2 -> 2)
contract_split_factorized = einop(
    (ax[b, ((h + w) * r), n], ax[n, d]),
    (ax[b, (h * r), d], ax[b, (w * r), d]),
).with_sizes(h=2, r=3)
left_f, right_f = contract_split_factorized(
    np.zeros((2, 9, 4), dtype=np.float32),
    np.zeros((4, 5), dtype=np.float32),
)
# left_f: (2, 6, 5), right_f: (2, 3, 5)
```

## More Examples

```python
import numpy as np
from einf import ax, axes, rearrange, reduce

b, h, w, d = axes("b", "h", "w", "d")

# 1) Callable reducer (non-string reducer)
reduce_with_callable = reduce(ax[b, h, d], ax[b]).reduce_by(np.max)
result = reduce_with_callable(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

# 2) Partial explicit sizes:
#    you do not need to provide every dim if remaining dims can be solved from input shape.
split_hw = rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=2)
y = split_hw(np.zeros((3, 10, 4), dtype=np.float32))  # shape: (3, 2, 5, 4)
```

## Pack (Variadic Axes)

```python
import numpy as np
from einf import ax, axes, packs, rearrange

(b,) = axes("b")
(tail,) = packs("tail")

# tail matches zero or more axes
move_b_to_last = rearrange(ax[b, tail], ax[tail, b])

y1 = move_b_to_last(np.zeros((2, 3, 4), dtype=np.float32))  # (3, 4, 2)
y2 = move_b_to_last(np.zeros((5,), dtype=np.float32))       # (5,)  (tail == empty)
```

## Performance Note (Cold vs Warm)

`TensorOp` construction includes planning/lowering. Reuse the same op instance for warm-path performance.

`einf` also has bounded constructor caches for identical operation specs (including configured variants such as `with_sizes(...)` and `reduce_by(...)`), so repeated identical construction often reuses existing objects.

Still, cache misses are cold-path work, so hoisting/reusing one op instance remains the recommended pattern.

```python
# good: one-time construction
REDUCE_BD = reduce(ax[b, n, d], ax[b, d])

def forward(x):
    return REDUCE_BD(x)

# avoid: cold construction per call
def slow_forward(x):
    return reduce(ax[b, n, d], ax[b, d])(x)
```

## Docs

- Spec index: `docs/spec/README.md`
- Core spec: `docs/spec/SPEC.md`
- Conformance: `docs/spec/CONFORMANCE.md`
- Error model: `docs/spec/ERRORS.md`
- Benchmarks: `docs/benchmarks/README.md`

## Development

```bash
pytest -q
basedpyright src tests
ruff check --extend-select I,UP .
ruff format .
```
