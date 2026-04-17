# einf

`einf` is a tensor DSL with axis signatures and first-class `TensorOp`
objects. Define an operation once — `rearrange`, `reduce`, `contract`,
`einop`, `view`, `repeat` — and reuse it across calls.

```python
import numpy as np
from einf import ax, axes, einop

b, n, d, m = axes("b", "n", "d", "m")

matmul = einop((ax[b, n, d], ax[d, m]), ax[b, n, m])

x = np.zeros((2, 3, 4), dtype=np.float32)
w = np.zeros((4, 5), dtype=np.float32)
matmul(x, w)  # -> (2, 3, 5)
```

## Start here

<div class="grid cards" markdown>

- __[Install](getting-started/install.md)__

    Base install, optional extras, and a one-line smoke test.

- __[First ops](getting-started/first-ops.md)__

    Walk the four core operations on small concrete examples.

- __[Attention, progressively](getting-started/attention-progressive.md)__

    Build attention step by step, then reuse the same op across shapes.

</div>

## Understand the model

- [Axis signatures](concepts/axis-signatures.md) — why named axes, how lhs/rhs roles work.
- [TensorOp as a value](concepts/tensorop-as-value.md) — what first-class ops unlock vs string-recompile libraries.
- [Shapes and broadcasting](concepts/shapes-and-broadcasting.md) — size binding, dynamic vs static axes.

## Recipes

- [Attention](cookbook/attention.md) — single-head, multi-head, causal mask.
- [LayerNorm](cookbook/layernorm.md)
- [Conv / pool](cookbook/conv-pool.md)
- [Batched gather](cookbook/batched-gather.md)

## Tooling

- [Validator CLI](guides/validator-cli.md) — `einf-validate` for batch/CI analysis.
- [LSP sidecar](guides/lsp-sidecar.md) — `einf-lsp` for editor integration.
- [Editors](guides/editors/index.md) — Helix, Zed, VS Code setup notes.

## Support matrix

| Surface | Status |
| --- | --- |
| Core DSL runtime (`rearrange`, `reduce`, `contract`, `einop`) | Stable |
| Validator CLI (`einf-validate`) | Stable |
| `einf-lsp` sidecar | Supported |
| Helix integration | Supported |
| Zed integration | Adapter needed |
| VS Code integration | Not first-class yet |
| Benchmark tooling | Supported for dev and release gates |

## Reference and internals

- [API reference](reference/api/index.md)
- [Architecture](internals/architecture.md), [design rationale](internals/design-rationale.md), [module map](internals/module-map.md)
