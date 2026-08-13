# TensorOp Execution Architecture

Status: Non-normative architecture note

## Goal

Define the internal execution hierarchy for `TensorOp`:

1. `TensorOp`: public operation value and call boundary.
2. `AbstractPlan`: one structural lowering source plus a lowering protocol.
3. `SymbolicPlan`: ordered symbolic primitive steps with deterministic scoring.
4. `RuntimeStep` chain: executable primitive steps specialized for one call shape/backend.

The design keeps flexibility at API ingress and keeps the internal pipeline canonical:

```text
TensorOp ingress -> AbstractPlan -> SymbolicPlan -> RuntimeStep chain -> execution
```

## Design Principles

1. Accept flexible inputs only at API boundaries.
2. Normalize once, then keep internal state canonical and strongly typed.
3. Keep lowering/search in `lowering`, plan selection/caching in `plans`, and primitive specialization in `steps`.
4. Treat optimizations as plan forms, not ad hoc call-site branches.
5. Keep symbolic steps primitive with explicit arity contracts.
6. Keep static analysis read-only and outside the runtime dependency graph.

## Core Entities

### TensorOp

`TensorOp` lives in `operations/tensor_op.py` and is the public operation value.
It owns the user-facing operation state and delegates call-time execution to
`operations/execution.py`.

Constructor-time validation that is not owned by a concrete primitive step lives in
`operations/validation.py`.

### AbstractPlan

`AbstractPlan` lives in `plans/abstract.py`. It contains one structural source and
the plan-owned lowering protocol seam:

- `source: LoweringSignature`
- `lowering: LoweringProgram`

`LoweringSignature` groups the operation name, normalized axis `Signature`, and
explicit size bindings. `AbstractPlan` accepts an `IRProgram` or `SymbolicPlan`
only when its source matches exactly. This check runs during plan construction,
before candidates enter runtime caches.

During initialization it lowers once into:

- `ir_program: IRProgram`
- `symbolic_candidates: tuple[SymbolicPlan, ...]`

`AbstractPlan` owns plan selection, candidate caches, route-runner caches, backend
profile caches, and runner fusion integration. It does not own concrete tensor
execution glue; that remains in `operations/execution.py`.

### LoweringProgram

`LoweringProgram` lives in `plans/lowering_protocol.py` because `AbstractPlan`
consumes the protocol. Concrete implementations live in `lowering/`:

- `DefaultLoweringProgram`
- `StaticLoweringProgram`
- `EmptyLoweringProgram`

`einf.lowering.LoweringProgram` remains a re-export of the plan-owned protocol for
convenient imports, but the canonical owner is `plans/lowering_protocol.py`.
The protocol receives a `LoweringSignature` and carries that source through the IR
and every symbolic candidate. Observability traces are not part of source identity.

### SymbolicPlan

`SymbolicPlan` lives in `plans/symbolic.py`. It is an ordered tuple of symbolic
steps:

- `kind: str`
- `source: LoweringSignature`
- `steps: tuple[SymbolicStep, ...]`

Input and output arity come from `source.signature`; callers cannot configure them
independently. `SymbolicPlan` validates step continuity, specializes symbolic steps
into runtime steps, executes the runtime chain, and computes deterministic scores.

### SymbolicStep

`SymbolicStep` lives in `steps/base.py`. It is one primitive symbolic instruction:

- `name: str`
- `input_arity: int`
- `output_arity: int`
- `score(context: PlanSelectionContext) -> SymbolicStepScore`
- `specialize(context: RuntimeSpecializationContext) -> RuntimeStep`

Step-consumed context and scoring helpers live under `steps/`:

- `steps/context.py`: `PlanSelectionContext`, `RuntimeExecutionContext`, pack expansion, runtime context normalization.
- `steps/scoring.py`: primitive shape/scoring helpers used by concrete steps.
- `plans/scoring.py`: plan-level `SymbolicPlanScore` only.

### RuntimeStep

`RuntimeStep` lives in `steps/base.py` and concrete step packages. It is one
executable primitive instruction:

- `name: str`
- `input_arity: int`
- `output_arity: int`
- `run(tensors) -> tuple[TensorLike, ...]`

Concrete primitive owners:

- `steps/einsum/`
- `steps/reduce/`
- `steps/reshape/`
- `steps/expand/`
- `steps/axis_slice/`
- `steps/concat.py`
- `steps/permute.py`

Each primitive owns its symbolic program model, compile helpers, specialization,
runtime execution, and backend-specific fast paths where applicable.

## Module Boundaries

Current package ownership:

- `operations/`: public op construction, `TensorOp`, call execution glue, constructor validation.
- `ir/`: lowering source/trace models, pure route solving, and static route tables.
- `lowering/`: concrete lowering implementations and IR-to-symbolic-candidate builders.
- `plans/`: abstract/symbolic plan contracts, plan-owned lowering protocol, selection/cache/fusion/route runtime/runners/rendering.
- `steps/`: primitive symbolic/runtime step contracts, models, compilation, specialization, and execution.
- `analysis/`: parser/checker/LSP/validator sidecar with no runtime importers.

Forbidden package dependency directions are enforced by
`tests/package/test_package_import_boundaries.py`. The current exception list is
empty.

## Static Analysis Boundary

`analysis/passes/einf_calls/` is split by responsibility:

- `entrypoint.py`: `ParsedModule` traversal and public `analyze_einf_calls`.
- `model.py`: internal parse/evaluation records and snippet span mapping.
- `syntax.py`: AST axis and side parsing.
- `semantics.py`: base `TensorOp` replay and method-chain dispatch.
- `reducers.py`: `reduce_by` reducer and phase parsing.
- `diagnostics.py`: analysis diagnostic codes and validation-error projection.
- `tokens.py`: axis role and missing-rhs-axis projection.

The analysis tree may read stable runtime boundary/canonical packages such as
`axis`, `diagnostics`, `operations`, `reduction`, `signature`, and `tensor_types`.
It must not import concrete `steps`.

## Test Taxonomy

Tests mirror production ownership:

- `tests/analysis/`
- `tests/axis/`
- `tests/backend/`
- `tests/benchmarks/`
- `tests/integration/`
- `tests/ir/`
- `tests/operations/`
- `tests/package/`
- `tests/plans/`
- `tests/shape/`

Multi-package public flows remain in `tests/integration/`. Package guardrails,
exports, scaffold, typing surface, and pipeline contracts live in `tests/package/`.

## Scoring Policy

`SymbolicPlan.score(context)` is lexicographic and deterministic:

1. `peak_einsum_numel`
2. `pre_einsum_materialize_numel`
3. `post_einsum_materialize_numel`
4. `allocation_count`
5. `kernel_count`
6. `step_count`

Rules:

1. score uses only computable values from plan structure and call context,
2. no traversal-order tie-breaking,
3. ties are resolved deterministically by candidate order.

## Current Lowering Policy

Current concrete lowering provides:

1. default lowering for `view`, `rearrange`, `repeat`, `reduce`, `contract`, and `einop`,
2. `einop` decomposition into primitive symbolic steps,
3. single-einsum carrier plans when representable,
4. bounded deterministic chain search fallback for contraction-bearing `einop`,
5. deterministic validation errors when no staged lowering exists.
