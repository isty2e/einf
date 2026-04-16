# einf

`einf` is a tensor DSL with axis signatures and first-class `TensorOp` objects.

Define an operation once (`rearrange`, `reduce`, `contract`, `einop`) and reuse it across calls.

!!! note "Draft"
    This landing page is a placeholder. Content is tracked in ticket `td-w4od`.

## Start here

- [Install](getting-started/install.md)
- [First ops](getting-started/first-ops.md)
- [Attention, progressively](getting-started/attention-progressive.md)

## Understand the model

- [Axis signatures](concepts/axis-signatures.md)
- [TensorOp as a value](concepts/tensorop-as-value.md)
- [Shapes and broadcasting](concepts/shapes-and-broadcasting.md)

## Recipes

- [Attention](cookbook/attention.md)
- [LayerNorm](cookbook/layernorm.md)
- [Conv / pool](cookbook/conv-pool.md)
- [Batched gather](cookbook/batched-gather.md)
