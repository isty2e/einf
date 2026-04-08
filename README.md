# einf

`einf` is a tensor DSL with axis signatures and first-class `TensorOp` objects.

You define an operation once (for example `rearrange`, `reduce`, `contract`, `einop`) and reuse it across calls.

## Install

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[analysis]"
pip install -e ".[lsp]"
```

Python: `>=3.10`

`analysis` installs optional parser dependencies used by the validator/LSP analysis stack, including `LibCstParserBackend`.
`lsp` installs the optional LSP sidecar dependencies (`pygls`, `lsprotocol`).

## LSP Sidecar

`einf` ships a minimal external language server sidecar for editor integration.

```bash
einf-lsp
```

The LSP server uses `initialize` options as its configuration source of truth.

```json
{
  "parser": "ast",
  "checkers": ["basedpyright", "pyrefly"]
}
```

Current minimal scope:

1. document sync,
2. `publishDiagnostics` from `einf` semantic analysis,
3. saved-file checker diagnostics from configured external checkers,
4. semantic tokens derived from `axis_tokens`.

Current richer editor affordances on top of the minimal sidecar:

1. hover metadata for axis-group relationships and role summaries,
2. inlay hints for selected non-trivial axis roles (`contracted`, `reduced`, `introduced`, `pack`).

External checker diagnostics refresh on save boundaries. Unsaved document changes continue to receive fresh `einf` semantic diagnostics and semantic tokens, but stale checker diagnostics are not retained as if they were current.

## Validator CLI

`einf` ships a checker-agnostic validator CLI for static DSL analysis.

```bash
einf-validate path/to/module.py
einf-validate src/ --parser ast
einf-validate src/ --parser libcst
einf-validate src/ --checker basedpyright --checker pyrefly
```

The command writes stable JSON to stdout and returns:

1. `0` when no diagnostics or validator failures are present,
2. `1` when any file contains semantic diagnostics, parse failures, or read failures.

Output contract:

1. `schema_version`
2. `parser_backend`
3. `checker_failures[]`
3. `files[]`
   `path`
   `diagnostics[]`
   `checker_diagnostics[]`
   `axis_tokens[]`
   `failures[]`

`diagnostics` contains `einf` semantic diagnostics. `checker_diagnostics` contains normalized external type-checker diagnostics. `failures` contains validator ingress failures such as unreadable files or parse errors. `checker_failures` contains checker invocation failures such as unavailable executables or malformed checker output.

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
