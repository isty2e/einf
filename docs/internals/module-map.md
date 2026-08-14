# Module map

`src/einf` is organized around a one-way runtime pipeline plus an
independent static-analysis tree. This page walks each top-level
directory and sketches the dependency order.

## Top-level files

| Module | Role |
| --- | --- |
| `__init__.py` | Curated public re-exports (ops, axis primitives, error types, reducer surface). |
| `tensor_types.py` | `TensorLike` protocol and index-key type aliases. The only runtime ingress contract for backends. |
| `diagnostics.py` | `ErrorCode`, `ValidationError`, `ExecutionError`. Used everywhere for structured failures. |
| `signature.py` | `Signature` — frozen pair of `AxisSide` tuples for inputs and outputs. The normalized form consumed by lowering. |
| `output_normalization.py` | Normalizes runtime step outputs to the arity-shaped form expected by callers. |

## Axis and shape foundation

- **`axis/`** — axis primitives and their structural split.
  `AxisTermBase` defines behavior shared by scalar terms and variadic packs;
  `ScalarAxisTermBase` adds scalar evaluation for `Axis`, `AxisExpr`, and
  `AxisInt`. The package also owns `AxisPack`, `AxisSide`, the `ax[...]`
  factory, canonical scalar algebra, and variadic-pack matching. No other einf
  module is a prerequisite. Higher layers use these types to define
  operations.

  `AxisTermBase` and `AxisPack` are structural-only. The `evaluate()`,
  `max_literal()`, and `evaluate_bounds()` methods belong to
  `ScalarAxisTermBase`. Code that receives an `AxisTermBase` must import
  `ScalarAxisTermBase` from `einf.axis` and narrow the term with
  `isinstance(term, ScalarAxisTermBase)` before using scalar algebra.
- **`shape/`** — compiled shape nodes and fast-path shape evaluators.
  Used by solver and plan specialization to go from symbolic axis
  expressions to concrete sizes at call time.
- **`solver/`** — dimension solver. Given an `AxisSide` and an input
  tensor shape, recover sizes for named axes. Consumes `axis/` and
  `shape/` only.

## Reducer and backend

- **`reduction/`** — reducer schema and plan parser. Owns the
  `Reducer` / `ReducerCallable` surface and the normalized reducer plan
  consumed by `reduce` and `einop`.
- **`backend/`** — backend dispatch built on `array-api-compat`.
  `BackendResolver` and `BackendProfile` establish namespace identity;
  `BackendPolicy` and runtime steps check only the primitives selected by the
  active plan. NumPy and PyTorch use explicit runtime primitive adapters; other
  namespaces use the Array API protocol fallback. The portable fallback is
  tested with `array-api-strict`. Memory-alias policy for strict `view` supports
  only NumPy and PyTorch.

## Operations surface

- **`operations/`** — the user-facing API. `api.py` exports the six op
  constructors (`rearrange`, `reduce`, `contract`, `einop`, `view`,
  `repeat`) plus `TensorOp`. `tensor_op.py` carries the op value object
  with planning state. `execution.py` owns call-time tensor execution
  glue. `validation.py` owns constructor-time operation contract checks.
  `policy.py` holds op-level policy knobs. `api.pyi` provides overload
  stubs that ship with the wheel.

## Planning and lowering pipeline

Planning is the internal execution model described in
[Architecture](architecture.md). The data flow is one-way:

```
TensorOp ingress
   │  (normalize and record call-site policy)
   ▼
AbstractPlan       ← operation definition + plan-owned lowering protocol
   │  (lowering expands to symbolic candidates)
   ▼
SymbolicPlan       ← ordered SymbolicStep tuple, scoreable, cacheable
   │  (specialization binds call-site shapes and backend profile)
   ▼
RuntimeStep chain  ← executable program
   │
   ▼
execution
```

`operations/` is the ingress and call boundary: it constructs the canonical
`AbstractPlan` with a concrete lowering program, then delegates repeated calls to
the plan runtime. Lowering still owns IR-to-symbolic candidate generation, and
steps still own primitive specialization/execution.

- **`ir/`** — canonical lowering sources and traces plus pure route solving and
  static routing tables. `LoweringSignature` owns the operation name, normalized
  axis signature, and explicit sizes. `IRProgram.source` is authoritative;
  `IRProgram.trace` is producer-owned observability metadata. The selected
  `SymbolicPlan.steps` remain the executable description. Call-time route
  resolution lives in `plans/`.
- **`lowering/`** — `LoweringProgram` implementations and the
  IR → symbolic-candidates compiler. Chain search, candidate pruning,
  and feasibility enforcement live here, not in runtime specialization.
- **`plans/`** — plan contracts and runtime integration.
  `abstract.py` defines `AbstractPlan`, `symbolic.py` the
  `SymbolicPlan`, `lowering_protocol.py` the `LoweringProgram` seam
  consumed by `AbstractPlan`, and `scoring.py` the plan-level scoring
  record. `cache.py`, `fusion/`, `routing.py`, `runners.py`, and
  `render.py` own the plan runtime support surfaces.
- **`steps/`** — primitive symbolic and runtime step modules
  (`permute`, `reshape`, `reduce`, `expand`, `einsum`, `concat`,
  `axis_slice`, plus shared `base.py`, `context.py`, `scoring.py`, and
  `runtime.py`). Each step owns its own specialization and arity
  contract. Step-consumed runtime context and primitive scoring helpers
  live here, not in `plans/`.

### Constructing lowering programs

The low-level lowering API uses one source value throughout the planning pipeline:

```python
from einf.ir import IRProgram, LoweringSignature
from einf.plans.abstract import AbstractPlan
from einf.plans.symbolic import SymbolicPlan
from einf.signature import Signature

source = LoweringSignature(
    op_name="rearrange",
    signature=Signature(inputs=lhs, outputs=rhs),
    explicit_sizes_items=explicit_sizes_items,
)
ir_program = IRProgram(source=source, trace=trace)
symbolic_plan = SymbolicPlan(source=source, kind="permute", steps=steps)
abstract_plan = AbstractPlan(source=source, lowering=lowering)
```

Call the lowering protocol with these forms:

```python
ir_program = lowering.ir_program(source)
symbolic_candidates = lowering.symbolic_candidates(ir_program=ir_program)
```

`LoweringProgram.ir_program()` takes `source` positionally. Its
`symbolic_candidates()` method takes only the resulting `IRProgram` as a keyword;
explicit sizes are available through `ir_program.source`. The old separate
`op_name`, `lhs`, `rhs`, and `explicit_sizes_items` arguments are not retained.

The exported functions in `einf.lowering.builders` accept the IR and reducer plan
positionally:

```python
symbolic_plan = build_rearrange_symbolic_plan(ir_program, reducer_plan)
```

The compiler entry point keeps keyword-only arguments:

```python
symbolic_candidates = build_symbolic_candidates_from_ir(
    ir_program=ir_program,
    reducer_plan=reducer_plan,
)
```

`IRProgram` and `SymbolicPlan` likewise no longer accept independent structural
fields. Read `IRProgram.op_name`, `lhs`, `rhs`, and arity through their derived
properties when needed. `SymbolicPlan.input_arity` and `output_arity` are also
derived from its source. A trace may change without changing source compatibility
or compiled candidates.

`einf.ir.build_default_ir_program` has been removed. Replace

```python
ir_program = build_default_ir_program(op_name=op_name, lhs=lhs, rhs=rhs)
```

with

```python
source = LoweringSignature(
    op_name=op_name,
    signature=Signature(inputs=lhs, outputs=rhs),
    explicit_sizes_items=explicit_sizes_items,
)
ir_program = IRProgram.from_source(source)
```

Construct the owning plan with `AbstractPlan(source=source, lowering=lowering)`.

### Constructing einsum programs

`einf.steps` exposes low-level pipeline types for integrations that construct
steps directly. The symbolic variants keep each construction form separate:

- `DirectEinsumSymbolicProgram` owns one equation per output. Every equation
  must consume the same number of inputs.
- `ChainEinsumSymbolicProgram` owns an ordered contraction chain and its
  carrier input. Each chain edge is a binary equation.
- `SideEinsumSymbolicProgram` owns axis sides and resolves its equation from
  runtime input shapes when necessary.
- All three variants inherit from `EinsumSymbolicProgram` and record whether
  lowering may use native `matmul`.

Each variant derives its input and output arity from its canonical fields.
`EinsumSymbolicStep` therefore accepts only the program and an optional name;
it does not accept separate arity values.

`EinsumSymbolicProgram` is abstract. Code that previously instantiated it
directly must construct the matching concrete variant or use
`build_einsum_symbolic_program_from_equations()` or
`build_einsum_symbolic_program_from_sides()`. The equation builder returns a
direct or chain variant after checking the caller's declared arities. The side
builder returns a side variant.

Import the concrete variants from `einf.steps.einsum` and map the former field
combinations to their matching constructor:

```python
from einf.steps.einsum import (
    ChainEinsumSymbolicProgram,
    DirectEinsumSymbolicProgram,
    SideEinsumSymbolicProgram,
)

direct_program = DirectEinsumSymbolicProgram(
    equations=("ab,bc->ac",),
    allow_native_matmul=True,
)
chain_program = ChainEinsumSymbolicProgram(
    equations=("ab,bc->ac", "ac,cd->ad"),
    chain_order=(1, 2),
    carrier_index=0,
    allow_native_matmul=True,
)
side_program = SideEinsumSymbolicProgram(
    lhs=lhs,
    rhs=rhs,
    explicit_sizes_items=explicit_sizes_items,
    allow_native_matmul=True,
)
```

Use the direct variant for equations without chain metadata, the chain variant
for equations with `chain_order` and `carrier_index`, and the side variant for
`lhs`, `rhs`, and explicit sizes. The variants derive their arity, so the old
`input_arity` and `output_arity` constructor arguments are omitted.

`EinsumSymbolicProgram` itself is no longer a dataclass. Calls such as
`dataclasses.fields(EinsumSymbolicProgram)` and positional pattern matching
against the base class therefore stop working. The concrete variants remain
dataclasses, but each has its own field layout. Serializers and introspection
code that assumed the former shared layout must dispatch on the concrete
variant.

### Constructing einop lowering plans

`einf.lowering.einop` exposes the canonical plans produced before symbolic-step
construction. The variants keep primitive routing, direct equations, layout
normalization, and composite carrier strategies separate:

- `PrimitiveEinopLoweringPlan` carries one `EinopPrimitiveRoute`.
- `DirectEinsumEinopLoweringPlan` carries one or more independent equations.
- `LayoutNormalizedEinopLoweringPlan` carries a required logical-layout
  normalization.
- `CarrierEinopLoweringPlan` carries one carrier equation, its intermediate
  axes, and the canonical unary tail plan.
- `ChainEinopLoweringPlan` carries the carrier input, ordered binary equations,
  final intermediate axes, and the canonical unary tail plan.
- `EinopChainSearchRequest` is a non-executable result from base planning. The
  complete planner resolves it before returning an `EinopLoweringPlan`.

`EinopLeafLoweringPlan` is the union of the primitive, direct-einsum, and
layout-normalized variants. Carrier and chain plans accept only a leaf selected
for their unary terminal signature; another carrier or chain cannot be nested
as a tail.

`EinopLoweringPlan` is now a closed union alias. Replace direct construction of
its former `kind` and nullable fields with the matching concrete variant:

```python
from einf import ax, axes
from einf.lowering.einop import (
    CarrierEinopLoweringPlan,
    DirectEinsumEinopLoweringPlan,
    EinopPrimitiveRoute,
    PrimitiveEinopLoweringPlan,
)

a, b, c, d = axes("a", "b", "c", "d")
direct_plan = DirectEinsumEinopLoweringPlan(
    equations=("ab,bc->ac",),
)
tail_plan = PrimitiveEinopLoweringPlan(
    route=EinopPrimitiveRoute.REARRANGE,
)
carrier_plan = CarrierEinopLoweringPlan(
    equation="abc,cd->abd",
    intermediate=ax[a, b, d],
    tail=tail_plan,
)
```

The former primitive `kind` strings map to `EinopPrimitiveRoute`, and
`kind="einsum"` maps to the direct variant. Layout metadata maps to the
layout-normalized variant. Carrier and chain plans now store an executable
`tail` instead of a `tail_kind`; symbolic construction consumes that plan
without selecting the tail again.

`build_einop_execution_plan_base()` may return either an executable plan or an
`EinopChainSearchRequest`. `build_einop_execution_plan()` always returns an
executable plan. The union alias is not a dataclass; each concrete plan variant
remains one.

#### Migrating einop planner helpers

The lower-level planner helpers remain available from their defining modules,
but their signatures and result types follow the same variant model:

```python
from einf import ax, axes
from einf.lowering.einop import (
    EinopLeafLoweringPlan,
    build_einop_execution_plan,
)
from einf.lowering.einop.carrier_plan import try_build_carrier_then_unary_plan
from einf.lowering.einop.search_plan import build_symbolic_einsum_chain_plan
from einf.signature import Signature

b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
signature = Signature(
    inputs=(ax[b, h + w, d], ax[d, j], ax[j, k]),
    outputs=(ax[b, h, k], ax[b, w, k]),
)


def build_terminal_tail(
    terminal_signature: Signature,
) -> EinopLeafLoweringPlan:
    plan = build_einop_execution_plan(
        analysis_signature=terminal_signature,
        has_reducer_plan=False,
    )
    if not isinstance(plan, EinopLeafLoweringPlan):
        raise TypeError("terminal planner returned a composite plan")
    return plan

carrier_plan = try_build_carrier_then_unary_plan(
    analysis_signature=signature,
)
chain_plan = build_symbolic_einsum_chain_plan(
    analysis_signature=signature,
    tail_builder=build_terminal_tail,
)
```

`try_build_carrier_then_unary_plan()` now returns
`CarrierEinopLoweringPlan | None`. Use `equation` instead of the former
single-item `equations` tuple; `intermediate` is unchanged, and `tail` contains
the unary leaf plan that symbolic construction will consume.

`build_symbolic_einsum_chain_plan()` now requires a `tail_builder` that returns
`EinopLeafLoweringPlan`, and the helper returns `ChainEinopLoweringPlan | None`.
Its `equations`, `intermediate`,
`carrier_index`, and `chain_order` fields retain their meanings. The executable
`tail` replaces `tail_kind`, so callers no longer select the terminal plan a
second time.

The runtime projection carries different facts.
`EinsumRuntimeProgram.native_matmul_equations` contains only resolved equations
already proven equivalent to native `matmul`.

The supported proof is intentionally narrow. An admitted equation uses only
explicit ASCII letter labels, contains no repeated label within either input or
the output, has a rank-2 right operand, contracts the left operand's final label
with the right operand's first label, and emits the left prefix followed by the
right operand's final label.

Code migrating from the former runtime `allow_native_matmul=True` field must not
admit every equation automatically. Use an empty set when the caller has no
independent proof:

```python
from einf.steps.einsum import EinsumRuntimeProgram

runtime_program = EinsumRuntimeProgram(
    equations=equations,
    chain_order=chain_order,
    carrier_index=carrier_index,
    native_matmul_equations=frozenset(),
)
```

If the caller can prove that an equation matches the native route, it may
include that exact resolved equation:

```python
runtime_program = EinsumRuntimeProgram(
    equations=("ij,jk->ik",),
    chain_order=(),
    carrier_index=None,
    native_matmul_equations=frozenset({"ij,jk->ik"}),
)
```

Construction rejects admissions that are absent from `equations` or do not
match the supported native-matmul grammar.

## Static analysis

- **`analysis/`** — an independent tree for static DSL analysis.
  - `parser/` — AST and LibCST parser backends.
  - `engine.py` + `model.py` + `passes/` — semantic analysis engine,
    diagnostic model, and package-split `einf_calls` pass
    (`syntax`, `semantics`, `reducers`, `diagnostics`, `tokens`,
    `entrypoint`).
  - `checkers/` — adapters for external type checkers (`pyright`,
    `basedpyright`, `zuban`, `ty`, `pyrefly`).
  - `validator/` — batch validator CLI (`einf-validate`) and its
    report model.
  - `lsp/` — `einf-lsp` language server sidecar.
  - `source.py` — source text handling shared across parsers.

  The analysis tree depends on runtime boundary/canonical packages
  read-only (`axis`, `diagnostics`, `operations`, `reduction`,
  `signature`, `tensor_types`), but not on concrete `steps`. Nothing in
  the runtime pipeline depends on analysis.

## Layering sketch

```
tensor_types  diagnostics
      │            │
      ▼            ▼
    axis  ──►  signature
      │            │
      ▼            ▼
    shape       reduction    backend
      │            │            │
      └────────────┼────────────┘
                   ▼
              operations ──► plans ──► steps
                   │           ▲        ▲
                   │           │        │
                   └──────► lowering ───┘
                               ▲
                               │
                              ir

              analysis ──► runtime boundary/canonical packages
```

Use this as a mental map — if a change crosses two boxes, it is worth
checking whether the direction respects the one-way flow. See
[Design rationale](design-rationale.md) for why the layers are split
this way.
