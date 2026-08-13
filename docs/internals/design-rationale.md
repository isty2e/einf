# Design rationale

[Architecture](architecture.md) specifies *what* the execution layers
are; this page is the companion that explains *why* they exist and what
a contributor stands to lose by collapsing them. Read it before
simplifying boundaries between `TensorOp`, `AbstractPlan`,
`SymbolicPlan`, and the `RuntimeStep` chain.

## Problems the structure solves

1. **Ergonomic API, canonical internals.**
   A user is allowed to hand in many forms — `AxisTerms`, `AxisSide`,
   tuples-of-tuples, `with_sizes(...)`, `reduce_by(...)`. Accepting that
   flexibility inside the op pipeline would produce a tangle of
   per-call special cases. The fix is to normalize at ingress and keep
   everything downstream in a single strongly-typed shape.

2. **Plan reuse across calls.**
   `TensorOp` construction includes planning and lowering. That is
   cold-path work by intent: the expensive decisions happen once, and
   every subsequent call reuses the result. This is the concrete
   payoff of the "first-class TensorOp" design — the cost model only
   works if planning is lifted out of the hot path.

3. **Backend neutrality.**
   A plan describes structure in axis terms, not in backend arrays.
   Binding to a supported runtime backend happens during specialization,
   not during planning. The same `SymbolicPlan` can be specialized against
   different Array API namespaces without replanning. NumPy and PyTorch use
   explicit primitive adapters; other namespaces use a capability-checked
   protocol fallback. `array-api-strict` guards the standard-only subset
   against accidental backend-specific dependencies.

4. **Deterministic optimization.**
   Plan selection is scored lexicographically on computable quantities
   (peak einsum size, pre/post materialization, allocation count,
   kernel count, step count). There is no traversal-order
   tie-breaking. Two builds with the same inputs produce the same
   plan, which keeps benchmarking, debugging, and CI reproducibility
   tractable.

5. **Optimization as plan forms, not as branches.**
   Fast-paths can be expressed as alternative symbolic candidates that
   lose or win under scoring. New optimizations graduate from ad-hoc
   conditionals (which rot silently) to first-class plan variants
   (which are visible, scored, and testable).

## Why each boundary exists

### `TensorOp` → `AbstractPlan`

`TensorOp` carries what the user asked for, plus call-site policy
(sizes, reducer, backend hints). `AbstractPlan` strips everything that
is not part of the operation definition and its lowering policy: no
`explicit_sizes`, no `backend_profile`, no duplicated `Signature`
field. Those are call-time concerns. Keeping them out of
`AbstractPlan` is what lets the same op definition plan once and
specialize many times.

### `AbstractPlan` → `SymbolicPlan`

Lowering is the layer that turns an operation definition into a
program of primitive steps. It is where chain search, candidate
pruning, and feasibility enforcement happen, and it runs at plan
build time. Keeping this work in lowering instead of in runtime
specialization is how we guarantee that per-call overhead does not
depend on the search space size.

An `AbstractPlan` may yield multiple `SymbolicPlan` candidates.
Lowering must prune aggressively before runtime:

- infeasible candidates are removed,
- equivalent candidates are deduplicated,
- ordering constraints are enforced (for example, `concat` before
  `einsum`; no `split` before `einsum`).

Runtime selection then only chooses among already-feasible
candidates, using deterministic scoring.

### `SymbolicPlan` → `RuntimeStep` chain

`SymbolicPlan` is backend-agnostic and shape-agnostic. `RuntimeStep`
is both bound. The `specialize(context)` step is the late-binding
line: it is where input shapes, backend profile, and reducer
bindings attach. Separating the two layers is what makes plan
caching safe — a cached `SymbolicPlan` can be specialized against a
new call-site context without risk of cross-call contamination.

### Primitive arity contracts on `SymbolicStep`

Each symbolic step declares a fixed `input_arity` and `output_arity`:
`einsum` is `N → 1`, `split` is `1 → N`, `reduce` is `1 → 1`, and so
on. The arity contracts make composition checkable mechanically and
prevent steps from secretly growing variadic behavior that would
defeat the primitive vocabulary.

## Concrete wins this paid for

- **`einop` chain search lives in lowering, not in runtime
  specialization.** A bad alternative would be to thread search into
  each call; that would push exponential work into the hot path. One
  candidate budget covers the complete chain search, and each subset
  space reserves its candidates before materialization. Signatures that
  exceed the limit fail during planning without allocating the oversized
  candidate table. Candidate scoring projects target and remaining-axis
  membership onto the subset basis once, so unrelated axes are not
  rescanned for every candidate.

- **Fast-paths re-expressed as plan variants.** For example, a
  single-einsum carrier plan (`einsum_carrier_then_unary`) competes
  with a deterministic exhaustive chain (`einsum_chain_then_unary`)
  via scoring. The "fast" path is not a hidden conditional — it is
  another candidate that wins when it wins.

- **Plan caching.** Bounded caches on identical operation specs (and
  on configured variants via `with_sizes` / `reduce_by`) let repeated
  identical construction reuse existing plan objects, which is what
  makes hoisting-and-reusing the canonical usage pattern idiomatically
  fast rather than just optimal under ideal conditions.

## What to avoid

- **Do not push call-specific specialization state into `AbstractPlan`.**
  Input shapes and backend profiles belong in
  `RuntimeSpecializationContext`, and shape-only candidate selection
  belongs in `PlanSelectionContext`. Leaking them onto the plan breaks
  cacheability and multi-backend reuse.

- **Do not place chain search inside `SymbolicStep.specialize`.**
  Search is a planning concern. Specialization must be cheap and
  deterministic.

- **Do not introduce variadic or optional-arity symbolic steps.** The
  arity contracts are load-bearing. If a new primitive genuinely
  needs variadic arity, give it a dedicated step rather than
  generalizing existing ones.

- **Do not add traversal-order tie-breakers to scoring.** Ties are
  resolved by candidate order, which is stable. Anything else makes
  output depend on insertion order.

If a proposed refactor collapses one of these boundaries, it is
almost certainly trading a long-term invariant for a short-term tidy.
