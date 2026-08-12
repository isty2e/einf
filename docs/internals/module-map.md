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

- **`ir/`** — canonical `IRProgram` operation signatures plus pure route
  solving and static routing tables. `IRProgram.op_name/lhs/rhs` are the
  authoritative lowering inputs; its `LoweringTraceStage` sequence is
  producer-owned conceptual metadata only. The selected `SymbolicPlan.steps`
  remain the executable description. Call-time route resolution lives in
  `plans/`.
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

### Constructing einsum runtime programs

`einf.steps` exposes low-level pipeline types for integrations that construct
steps directly. Symbolic and runtime programs carry different facts:

- `EinsumSymbolicProgram.allow_native_matmul` records whether lowering may use
  the optimization.
- `EinsumRuntimeProgram.native_matmul_equations` contains only resolved
  equations already proven equivalent to native `matmul`.

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
