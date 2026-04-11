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

- **`axis/`** — axis primitives. `Axis`, `AxisPack`, `AxisTerm`,
  `AxisSide`, the `ax[...]` factory, canonical scalar algebra, and the
  variadic-pack matcher. No other einf module is a prerequisite. All
  higher layers use these as the vocabulary for op definitions.
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
  `BackendPolicy`, `BackendResolver`, and `BackendProfile` detect the
  active namespace (NumPy / PyTorch / JAX / …) and expose it to steps.
  Also owns memory-alias policy.

## Operations surface

- **`operations/`** — the user-facing API. `api.py` exports the six op
  constructors (`rearrange`, `reduce`, `contract`, `einop`, `view`,
  `repeat`) plus `TensorOp`. `tensor_op.py` carries the op value object
  with planning and call behavior. `policy.py` holds op-level policy
  knobs. `api.pyi` provides overload stubs that ship with the wheel.

## Planning and lowering pipeline

Planning is the internal execution model described in
[Architecture](architecture.md). The data flow is one-way:

```
TensorOp ingress
   │  (normalize and record call-site policy)
   ▼
AbstractPlan       ← operation definition + lowering policy
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

- **`ir/`** — shared IR node model (`AssembleIR`, `GatherIR`,
  `RouteIR`, `TransformIR`) and routing tables. Acts as the
  intermediate representation between op ingress and lowering.
- **`lowering/`** — `LoweringProgram` implementations and the
  IR → symbolic-candidates compiler. Chain search, candidate pruning,
  and feasibility enforcement live here, not in runtime specialization.
- **`plans/`** — plan contracts and runtime integration.
  `abstract.py` defines `AbstractPlan`, `symbolic.py` the
  `SymbolicPlan`, `context.py` the `SpecializationContext`,
  `scoring.py` the deterministic scoring policy, `entrypoint.py` the
  `TensorOp` call-site glue, and `cache.py` / `fusion/` / `runners.py`
  / `render.py` the runtime support surfaces.
- **`steps/`** — primitive symbolic and runtime step modules
  (`permute`, `reshape`, `reduce`, `expand`, `einsum`, `concat`,
  `axis_slice`, plus shared `base.py` and `runtime.py`). Each step
  owns its own specialization and arity contract.

## Static analysis

- **`analysis/`** — an independent tree for static DSL analysis.
  - `parser/` — AST and LibCST parser backends.
  - `engine.py` + `model.py` + `passes/` — semantic analysis engine
    and diagnostic model.
  - `checkers/` — adapters for external type checkers (`pyright`,
    `basedpyright`, `zuban`, `ty`, `pyrefly`).
  - `validator/` — batch validator CLI (`einf-validate`) and its
    report model.
  - `lsp/` — `einf-lsp` language server sidecar.
  - `source.py` — source text handling shared across parsers.

  The analysis tree depends on the runtime packages read-only
  (`axis`, `diagnostics`, `operations`, `reduction`, `signature`,
  `steps.einsum`, `tensor_types`), but nothing in the runtime pipeline
  depends on analysis.

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
                  ir ──►  lowering ──► plans ──► steps
                                          │
                                          ▼
                                      operations ──►  public API
                                          ▲
                                          │ (read-only consumer)
                                      analysis
```

Use this as a mental map — if a change crosses two boxes, it is worth
checking whether the direction respects the one-way flow. See
[Design rationale](design-rationale.md) for why the layers are split
this way.
