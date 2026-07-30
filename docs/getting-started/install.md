# Install

`einf` targets Python `>=3.10`.

`einf` is not on PyPI. Install directly from the Git repository:

```bash
pip install git+https://github.com/isty2e/einf.git
```

The base install provides the `einf` runtime and its backend-dispatch
dependencies, but it does not install a tensor backend. Install either NumPy
or PyTorch separately.

```bash
pip install numpy
# or install PyTorch for your platform
```

`array-api-compat` identifies the namespace of input tensors. It does not make
every Array API namespace a supported and tested `einf` runtime backend.

## Runtime backend support

The following matrix is the supported and CI-tested runtime contract:

| Operation | NumPy | PyTorch | Other Array API namespaces |
| --- | --- | --- | --- |
| `view` | Supported | Supported | Unsupported |
| `rearrange` | Supported | Supported | Capability-gated |
| `repeat` | Supported | Supported | Capability-gated |
| `reduce` | Supported | Supported | Capability-gated |
| `contract` | Supported | Supported | Capability-gated |
| `einop` | Supported | Supported | Capability-gated |

NumPy and PyTorch are tested in CI across all six operations.
`Capability-gated` is not a support guarantee: the runtime checks that the
namespace provides every primitive selected by the operation's lowered plan,
such as `reshape`, `permute_dims`, `expand_dims`, `broadcast_to`, `concat`, or
the selected reducer. `contract` and contraction paths in `einop` also require
an einsum implementation recognized by `opt_einsum`.

`view` is stricter than the other operations: it succeeds only when `einf` can
prove that the result shares storage with the input. NumPy and PyTorch have
explicit implementations for that proof; other backends fail rather than
silently copy.

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

After installing `einf` and NumPy, the following should print shapes without
error:

```python
import numpy as np
from einf import ax, axes, rearrange

b, n, d = axes("b", "n", "d")
transpose = rearrange(ax[b, n, d], ax[b, d, n])
print(transpose(np.zeros((2, 3, 4), dtype=np.float32)).shape)
# -> (2, 4, 3)
```

Next: [first ops](first-ops.md).
