# Install

`einf` targets Python `>=3.10`.

`einf` is not on PyPI. Install directly from the Git repository:

```bash
pip install git+https://github.com/isty2e/einf.git
```

The base install provides the `einf` runtime and its backend-dispatch
dependencies, but it does not install an array implementation. Install a
compatible implementation separately. NumPy and PyTorch have full named
backend coverage in CI.

```bash
pip install numpy
# or install PyTorch for your platform
```

[`array-api-compat`](https://data-apis.org/array-api-compat/supported-array-libraries.html)
recognizes NumPy, CuPy, PyTorch, Dask, JAX, ndonnx, and Sparse, and accepts
other implementations that expose `__array_namespace__`. This admits the
namespace to `einf`; each operation still checks the capabilities it needs.

## Runtime backend support

Support follows operation capabilities rather than a closed list of backend
names:

| Operation | Portable Array API baseline | Additional requirement | CI coverage |
| --- | --- | --- | --- |
| `view` | No | Backend-specific storage-sharing proof | NumPy, PyTorch |
| `rearrange` | Yes | Methods selected by the layout plan | NumPy, PyTorch, `array-api-strict` |
| `repeat` | Yes | `reshape`, `expand_dims`, `broadcast_to` | NumPy, PyTorch, `array-api-strict` |
| `reduce` | Yes for named reducers | `asarray` plus either the selected named reducer or a backend-compatible callable | NumPy, PyTorch, `array-api-strict` |
| `contract` | No | Namespace `einsum`, or an implementation recognized by `opt_einsum` | NumPy, PyTorch |
| `einop` | Plan-dependent | Every capability required by its selected steps | NumPy, PyTorch, `array-api-strict` for non-contraction plans |

`array-api-strict` is a test-only minimal implementation of the standard. Its
CI coverage verifies that `rearrange`, `repeat`, named `reduce`, and
non-contraction `einop` plans do not accidentally depend on NumPy- or
PyTorch-only behavior. It is not a runtime dependency or an end-user backend.

Other Array API namespaces take the same protocol path. A namespace with the
methods selected by a plan can run that plan without implementing unrelated
operations. `contract` and contraction-bearing `einop` plans require either a
callable `einsum` on the namespace or a backend implementation recognized by
`opt_einsum`. This lets implementations such as JAX, Dask, or MLX work without
a dedicated `einf` adapter, but only NumPy and PyTorch receive full
named-backend CI coverage.

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
