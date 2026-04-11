# einf

`einf` is a tensor DSL with axis signatures and first-class `TensorOp`
objects. Define an operation once — `rearrange`, `reduce`, `contract`,
`einop`, `view`, `repeat` — and reuse it across calls, shapes, and
backends.

## Install

```bash
pip install git+https://github.com/isty2e/einf.git
```

Python `>=3.10`. Extras (`analysis`, `lsp`, `dev`, `docs`) and the
contributor editable install are documented in
[docs/getting-started/install.md](docs/getting-started/install.md).

## A taste

```python
import numpy as np
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")

matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

x = np.zeros((2, 3, 4), dtype=np.float32)
w = np.zeros((4, 5), dtype=np.float32)
matmul(x, w).shape
# -> (2, 3, 5)
```

## Documentation

The [docs site](docs/index.md) is the canonical source of truth.

- [First ops](docs/getting-started/first-ops.md) and
  [attention, progressively](docs/getting-started/attention-progressive.md)
  for the tutorial path.
- [Axis signatures](docs/concepts/axis-signatures.md),
  [TensorOp as a value](docs/concepts/tensorop-as-value.md), and
  [shapes and broadcasting](docs/concepts/shapes-and-broadcasting.md)
  for the mental model.
- Recipes in [cookbook/](docs/cookbook/): attention, layernorm,
  conv/pool, batched gather.
- [Validator CLI](docs/guides/validator-cli.md) and
  [LSP sidecar](docs/guides/lsp-sidecar.md) for tooling.
- [API reference](docs/reference/api/index.md) and
  [internals](docs/internals/architecture.md).

