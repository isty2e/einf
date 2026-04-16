# TensorOp Execution Architecture

Status: Non-normative architecture note

## Goal

Define the internal execution hierarchy for `TensorOp`:

1. `AbstractPlan`: ingress-normalized operation definition.
2. `SymbolicPlan`: symbolic execution program as ordered symbolic steps.
3. `RuntimeStep` chain: specialized executable steps assembled per call.

This keeps `TensorOp` ergonomic while moving execution complexity into a strict internal model.

## Design Principles

1. Accept flexibility only at API boundaries.
2. Keep internal state canonical and strongly typed.
3. Keep data flow one-way:
   `TensorOp ingress -> AbstractPlan -> SymbolicPlan -> RuntimeStep chain -> execution`.
4. Treat optimizations as plan forms, not ad hoc fastpath conditionals.
5. Keep chain/einsum search in lowering, never in symbolic step specialization.
6. Keep symbolic steps primitive with explicit arity contracts.

## Core Entities

### AbstractPlan

`AbstractPlan` contains only operation definition and lowering policy.

- `op_name: str`
- `lhs: AxisSide`
- `rhs: AxisSide`
- `lowering: LoweringProgram`

`AbstractPlan` does not own runtime information:

- no `explicit_sizes`
- no `backend_profile`
- no `supports_reducer`
- no duplicated `Signature` field

Those are call-site concerns handled during symbolic selection or specialization.

### SymbolicPlan

`SymbolicPlan` is an ordered tuple of symbolic steps.

- `kind: str`
- `input_arity: int`
- `output_arity: int`
- `steps: tuple[SymbolicStep, ...]`

`SymbolicPlan` does not reference `AbstractPlan`.

### SymbolicStep

`SymbolicStep` is one symbolic instruction and owns specialization.

- `name: str`
- `input_arity: int`
- `output_arity: int`
- `specialize(context) -> RuntimeStep`
- `score(context) -> StepScore`

Primitive arity contracts:

- `einsum`: `N -> 1`
- `contract`: `N -> 1`
- `concat`: `N -> 1` (`N >= 2`)
- `split`: `1 -> N` (`N >= 2`)
- `reduce`: `1 -> 1`
- `expand`: `1 -> 1`
- `view`: `1 -> 1`
- `permute`: `1 -> 1`
- `reshape`: `1 -> 1`
- `reindex` (generic structural fallback): `N -> M` (`N >= 1`, `M >= 1`)

### RuntimeStep

`RuntimeStep` is one executable instruction.

- `name: str`
- `input_arity: int`
- `output_arity: int`
- `run(tensors) -> tuple[TensorLike, ...]`

## Context Object

Specialization requires call-time context:

- input shapes
- backend profile

This is represented as `SpecializationContext` and passed to each `SymbolicStep.specialize`.

## Module Boundaries

Create `src/einf/plans/` with:

- `types.py`: protocols and specialization/selection contexts.
- `abstract.py`: `AbstractPlan` and `LoweringProgram`.
- `symbolic.py`: `SymbolicPlan`.
- `steps/`: step modules split by operation kind (`view`, `rearrange`,
  `expand`, `reduce`, `einsum`) plus shared
  runtime placeholders.
- `lowering.py`: concrete lowering policy helpers.

No execution kernels live in this package; this package models planning contracts only.

## Current Scaffolding Status

Current implementation provides:

1. concrete default lowering (`DefaultLoweringProgram`) for:
   - `view`, `rearrange`, `repeat` (lowered to `expand`), `reduce`, `contract`,
     `einop`
2. `einop` lowering decomposed into non-`einop` symbolic primitives
   (`split`, `concat`, `permute`, `reshape`, `expand`, `reduce`, `contract`, `einsum`).
3. chain search runs in lowering/planning, not inside symbolic runtime specialization.
4. `SymbolicPlan.score()` aggregates deterministic `SymbolicStep.score()` results.

## Feasibility and Candidate Narrowing

Lowering policy:

1. Same `AbstractPlan` MAY yield multiple symbolic candidates.
2. Lowering MUST prune aggressively before runtime:
   - infeasible candidates are removed,
   - equivalent candidates are deduplicated,
   - ordering constraints are enforced (for example: `concat` before `einsum`; no `split` before `einsum`).
3. Runtime selection only chooses among already-feasible candidates.

Current `einop` feasibility path:

1. direct primitive lowering,
2. single-einsum carrier plan (`einsum_carrier_then_unary`) when representable,
3. deterministic exhaustive chain search (`einsum_chain_then_unary`) as fallback,
4. deterministic validation error when no staged non-view lowering exists.

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

## Implementation Progress

1. Add plan ADTs and protocol tests.
2. Build `AbstractPlan` from `TensorOp` configuration.
3. Replace ad hoc `TensorOp.__call__` branching with plan pipeline.
4. Re-express current fastpaths as symbolic/runtime plan variants.
