# Install

`einf` targets Python `>=3.10`.

`einf` is not on PyPI. Install directly from the Git repository:

```bash
pip install git+https://github.com/isty2e/einf.git
```

The base install pulls in the runtime dependencies needed to use the
core ops (`rearrange`, `reduce`, `contract`, `einop`, `view`, `repeat`)
on any backend that speaks the [Array API standard](https://data-apis.org/array-api/)
through `array-api-compat` — NumPy, PyTorch, JAX, and others.

## Optional extras

`einf` splits non-runtime dependencies into named extras so you only install
what your workflow needs.

```bash
pip install "einf[analysis] @ git+https://github.com/isty2e/einf.git"
pip install "einf[lsp]      @ git+https://github.com/isty2e/einf.git"
```

Multiple extras can be combined:

```bash
pip install "einf[analysis,lsp] @ git+https://github.com/isty2e/einf.git"
```

| Extra | Purpose | Consumers |
| --- | --- | --- |
| `analysis` | CST-based parser for richer static analysis | [Validator CLI](../guides/validator-cli.md) |
| `lsp` | Language-server protocol implementation | [LSP sidecar](../guides/lsp-sidecar.md) |
| `dev` | Run the test suite and library comparisons | Contributors |
| `docs` | Build the documentation site | Contributors |

## Contributor install (editable)

If you are working on `einf` itself, clone and install editable with
the developer extras:

```bash
git clone https://github.com/isty2e/einf.git
cd einf
pip install -e ".[dev,analysis,lsp,docs]"
```

## Verify

After installing, the following should print shapes without error:

```python
import numpy as np
from einf import ax, axes, rearrange

b, n, d = axes("b", "n", "d")
transpose = rearrange(ax[b, n, d], ax[b, d, n])
print(transpose(np.zeros((2, 3, 4), dtype=np.float32)).shape)
# -> (2, 4, 3)
```

Next: [first ops](first-ops.md).
