# API reference

This page enumerates the intentional public surface of `einf`. Symbols
are imported directly from the top-level `einf` namespace.

```python
from einf import ax, axes, packs, symbols
from einf import rearrange, reduce, contract, einop, repeat, view
from einf import TensorOp, Signature, Axis, AxisPack, AxisExpr, AxisInt
from einf import TensorLike, Reducer, ReducerCallable, ReducerName
from einf import ValidationError, ExecutionError, ErrorCode
```

## Core operations

Each constructor below returns a [`TensorOp`][einf.TensorOp] that can be
called with concrete tensors and reused across calls.

::: einf.rearrange
    options:
      show_root_heading: true

::: einf.reduce
    options:
      show_root_heading: true

::: einf.contract
    options:
      show_root_heading: true

::: einf.einop
    options:
      show_root_heading: true

::: einf.view
    options:
      show_root_heading: true

::: einf.repeat
    options:
      show_root_heading: true

## TensorOp

::: einf.TensorOp
    options:
      show_root_heading: true
      members:
        - with_sizes
        - reduce_by
        - __call__

## Axis construction

::: einf.ax
    options:
      show_root_heading: true

::: einf.axes
    options:
      show_root_heading: true

::: einf.packs
    options:
      show_root_heading: true

::: einf.symbols
    options:
      show_root_heading: true

## Axis and signature types

::: einf.Axis
    options:
      show_root_heading: true

::: einf.AxisPack
    options:
      show_root_heading: true

::: einf.AxisExpr
    options:
      show_root_heading: true

::: einf.AxisInt
    options:
      show_root_heading: true

::: einf.Signature
    options:
      show_root_heading: true

## Reducer surface

::: einf.Reducer
    options:
      show_root_heading: true

::: einf.ReducerCallable
    options:
      show_root_heading: true

::: einf.ReducerName
    options:
      show_root_heading: true

## Error types

::: einf.ValidationError
    options:
      show_root_heading: true

::: einf.ExecutionError
    options:
      show_root_heading: true

::: einf.ErrorCode
    options:
      show_root_heading: true

## Tensor protocol

::: einf.TensorLike
    options:
      show_root_heading: true
