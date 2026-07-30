# Benchmarking

`einf` ships benchmark tooling for four different jobs:

- `benchmarks/compare/`: library-to-library performance comparison
- `benchmarks/audit/`: diagnostic audits such as layout and stride inspection
- `benchmarks/profile/`: internal overhead and warm call-tree profiling
- `benchmarks/guardrail/`: regression checks against stored raw benchmark output

Shared infrastructure lives under:

- `benchmarks/shared/`: helper utilities used by compare/audit/profile entrypoints
- `benchmarks/harness/`: benchmark scheduling, materialization, rendering, and result models

Assumption:

- you are running from the repository root,
- `einf` and optional benchmark dependencies are already installed in your environment.

## Compare Methodology

The fixed and dynamic compare scripts use the same high-level fairness model:

1. all competitors see the same logical workload,
2. each competitor receives independently materialized tensors,
3. per-batch execution order rotates to spread first-executor bias,
4. eager CPU timing excludes harness-side output observation and materialization.

More concretely:

- fixed cold timing covers operation construction plus the first library call,
- fixed warm benchmarks use paired per-call execution on the same logical input,
- each fixed warm observation is the arithmetic mean of the configured timed
  calls in one repeat,
- dynamic benchmarks use paired same-batch execution on the same logical batch stream inside each round,
- tuple traversal, shape access, indexing, `.item()`, and parity validation happen outside timed regions,
- dynamic rounds still use different batch streams from one round to the next, so round summaries matter for heavy cases.

The shared compare harness supports synchronous CPU outputs. It rejects
asynchronous device outputs instead of reporting incomplete eager-launch
latency. Supporting an asynchronous backend requires explicit synchronization
both before the timer starts, to drain queued work, and after the library call,
before the timer stops. Reports must label that interval as synchronized
latency.

Diagnostic scripts under `benchmarks/audit/` and `benchmarks/profile/` may
define broader task-specific timed regions; their own methodology is
authoritative.

## Taxonomy Refactor Performance Gate

Use this gate before closing any non-LSP taxonomy migration ticket that touches
`operations`, `plans`, `lowering`, `steps`, `backend`, or benchmark harness code.
Reference this section, the exact commands, and the artifact paths in the ticket
close note or the commit body for that migration.

The gate has three separate evidence classes:

1. focused tests prove behavior and import-boundary correctness,
2. compare benchmarks show library-facing timing against `einops` and `einx`,
3. overhead profiles plus guardrail checks prove `einf` hot-path regression status.

Do not substitute one class for another. A passing pytest run is not performance
evidence, and a benchmark report is not a correctness proof.

### Focused Correctness Gate

Run this from the repository root after the migration:

```bash
python -m pytest \
  tests/package/test_package_import_boundaries.py \
  tests/integration/test_array_api_contract_einop_execution.py \
  tests/integration/test_array_api_rearrange_execution.py \
  tests/integration/test_array_api_repeat_reduce_execution.py \
  tests/integration/test_array_api_view_execution.py \
  tests/operations/test_tensorop_contract.py \
  tests/plans/test_plan_hierarchy.py \
  tests/package/test_pipeline_contracts.py \
  tests/benchmarks/test_benchmark_harness.py \
  tests/benchmarks/test_benchmark_guardrails.py
```

Add narrower tests for the touched package when the ticket changes a specific
contract. Keep the import-boundary guard in the command so a migration cannot
close while adding a new forbidden package edge.

### Baseline Reference

Record the baseline git ref before applying the non-LSP migration. Use the same
machine, dependency versions, and shell environment for baseline and candidate.
If the machine was busy, slept, thermally throttled, or dependency versions
changed, discard the artifacts and rerun both sides.

```bash
BASE_REF=$(git rev-parse HEAD)
BENCH_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_DIR="artifacts/bench/taxonomy-${BENCH_STAMP}"
BASE_WORKTREE="/tmp/einf-overhead-baseline-${BENCH_STAMP}"
REPO_ROOT=$(pwd)
mkdir -p "$BENCH_DIR/raw"
git worktree add --detach "$BASE_WORKTREE" "$BASE_REF"
```

When the migration is complete, keep using the same `BENCH_DIR`,
`BASE_WORKTREE`, and `REPO_ROOT` values for the commands below.

### Debiased Overhead Capture

Run the overhead profile in both execution orders. This keeps the current
single-report guardrail available for diagnostics, but the taxonomy migration
gate uses repeated-trial comparison so one order-biased report does not decide
the ticket.

Order A, baseline first:

```bash
(
  cd "$BASE_WORKTREE"
  python -m benchmarks.profile.overhead_breakdown \
    --backend torch \
    --output "$REPO_ROOT/$BENCH_DIR/baseline-overhead-order-a-torch.md" \
    --raw-output "$REPO_ROOT/$BENCH_DIR/raw/baseline-overhead-order-a-torch.json"
)

python -m benchmarks.profile.overhead_breakdown \
  --backend torch \
  --output "$BENCH_DIR/candidate-overhead-order-a-torch.md" \
  --raw-output "$BENCH_DIR/raw/candidate-overhead-order-a-torch.json"
```

Order B, candidate first:

```bash
python -m benchmarks.profile.overhead_breakdown \
  --backend torch \
  --output "$BENCH_DIR/candidate-overhead-order-b-torch.md" \
  --raw-output "$BENCH_DIR/raw/candidate-overhead-order-b-torch.json"

(
  cd "$BASE_WORKTREE"
  python -m benchmarks.profile.overhead_breakdown \
    --backend torch \
    --output "$REPO_ROOT/$BENCH_DIR/baseline-overhead-order-b-torch.md" \
    --raw-output "$REPO_ROOT/$BENCH_DIR/raw/baseline-overhead-order-b-torch.json"
)
```

Expected overhead artifacts:

- `$BENCH_DIR/baseline-overhead-order-a-torch.md`
- `$BENCH_DIR/raw/baseline-overhead-order-a-torch.json`
- `$BENCH_DIR/candidate-overhead-order-a-torch.md`
- `$BENCH_DIR/raw/candidate-overhead-order-a-torch.json`
- `$BENCH_DIR/candidate-overhead-order-b-torch.md`
- `$BENCH_DIR/raw/candidate-overhead-order-b-torch.json`
- `$BENCH_DIR/baseline-overhead-order-b-torch.md`
- `$BENCH_DIR/raw/baseline-overhead-order-b-torch.json`

### Library Compare Capture

Capture baseline and candidate library comparison reports. These reports are
library-facing timing evidence, not the primary current-vs-baseline overhead
guardrail.

```bash
(
  cd "$BASE_WORKTREE"
  python -m benchmarks.compare.einf_einops_einx \
    --backend torch \
    --scale large \
    --rounds 6 \
    --cold-repeats 3 \
    --warmup 4 \
    --warm-repeats 5 \
    --warm-iterations 60 \
    --output "$REPO_ROOT/$BENCH_DIR/baseline-fixed-large-torch.md"
)

python -m benchmarks.compare.einf_einops_einx \
  --backend torch \
  --scale large \
  --rounds 6 \
  --cold-repeats 3 \
  --warmup 4 \
  --warm-repeats 5 \
  --warm-iterations 60 \
  --output "$BENCH_DIR/candidate-fixed-large-torch.md"

(
  cd "$BASE_WORKTREE"
  python -m benchmarks.compare.einf_einops_einx_dynamic \
    --backend torch \
    --scale large \
    --batches 64 \
    --warmup-batches 8 \
    --repeats 6 \
    --rounds 3 \
    --parity-checks 8 \
    --output "$REPO_ROOT/$BENCH_DIR/baseline-dynamic-large-torch.md"
)

python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --parity-checks 8 \
  --output "$BENCH_DIR/candidate-dynamic-large-torch.md"
```

Expected compare artifacts:

- `$BENCH_DIR/baseline-fixed-large-torch.md`
- `$BENCH_DIR/baseline-dynamic-large-torch.md`
- `$BENCH_DIR/candidate-fixed-large-torch.md`
- `$BENCH_DIR/candidate-dynamic-large-torch.md`

### Guardrail Checks

The overhead profile covers the core TensorOp hot path in fixed and dynamic,
medium and large scenarios. Run repeated-trial checks for both metrics:

```bash
python -m benchmarks.guardrail.check_overhead_trials \
  --pair \
    "$BENCH_DIR/raw/baseline-overhead-order-a-torch.json" \
    "$BENCH_DIR/raw/candidate-overhead-order-a-torch.json" \
  --pair \
    "$BENCH_DIR/raw/baseline-overhead-order-b-torch.json" \
    "$BENCH_DIR/raw/candidate-overhead-order-b-torch.json" \
  --metric instrumented_call_ms \
  --max-regression-ratio 0.10

python -m benchmarks.guardrail.check_overhead_trials \
  --pair \
    "$BENCH_DIR/raw/baseline-overhead-order-a-torch.json" \
    "$BENCH_DIR/raw/candidate-overhead-order-a-torch.json" \
  --pair \
    "$BENCH_DIR/raw/baseline-overhead-order-b-torch.json" \
    "$BENCH_DIR/raw/candidate-overhead-order-b-torch.json" \
  --metric unpatched_call_ms \
  --max-regression-ratio 0.10
```

Do not use `--allow-missing-cases` for this gate. Missing baseline cases are a
gate failure because they make regression status unknowable.

By default, `check_overhead_trials` requires the same case to exceed the
threshold in every supplied pair. For a two-order gate, a one-order regression
is evidence to inspect or rerun, not a ticket-closing blocker by itself.

### Interpretation Rules

Close a non-LSP taxonomy migration only when all of these are true:

- focused correctness tests pass,
- both repeated-trial overhead guardrail commands exit 0,
- candidate compare reports do not show a new `einf` slowdown pattern that
  contradicts the repeated-trial overhead result,
- dynamic compare aggregate medians are read together with round median summaries,
- any repeated-trial overhead slowdown in the 5-10 % range is rerun or explained
  with case-level evidence,
- any repeated-trial overhead slowdown above 10 % blocks the ticket unless the
  user explicitly accepts the regression.

Single-run compare deltas are diagnostic when comparing one git ref to another.
They should trigger inspection or reruns when they contradict the overhead gate,
but they do not override a clean repeated-trial overhead result by themselves.

For dynamic compare reports, compare libraries within the same round before
reading across rounds. The harness pairs libraries on the same logical batch
stream inside a round; different rounds may legitimately use different workload
streams.

If a guardrail fails, collect a call tree before changing benchmark policy:

```bash
python -m benchmarks.profile.warm_calltree \
  --backend torch \
  --mode dynamic \
  --scale large \
  --case einop_contract_split \
  --warmup 32 \
  --loops 256 \
  --sort cumtime \
  --top 40 \
  --output "$BENCH_DIR/candidate-warm-calltree-dynamic-large-einop-contract-split.md" \
  --raw-output "$BENCH_DIR/raw/candidate-warm-calltree-dynamic-large-einop-contract-split.json"
```

Use the call tree to explain the regression source. Do not relax the guardrail
threshold to make a migration pass.

## Fixed-Shape Compare

Use `benchmarks/compare/einf_einops_einx.py` for cold and warm fixed-shape comparisons.

Example:

```bash
python -m benchmarks.compare.einf_einops_einx \
  --backend torch \
  --scale large \
  --rounds 6 \
  --cold-repeats 3 \
  --warmup 4 \
  --warm-repeats 3 \
  --warm-iterations 60 \
  --output artifacts/bench/2026-02-15-einf-vs-einops-einx-large-torch.md
```

What the script reports:

- cold construction + first-call timing,
- warm steady-state observations, each an arithmetic mean over
  `--warm-iterations` timed calls,
- round-level warm summaries,
- per-case library order for each round.

## Dynamic-Shape Compare

Use `benchmarks/compare/einf_einops_einx_dynamic.py` when batch shapes vary.

Example:

```bash
python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --parity-checks 8 \
  --output artifacts/bench/2026-02-15-einf-vs-einops-einx-dynamic-large-target-r3-torch.md
```

What matters here:

- each report derives sampled and fixed dimensions, inclusive sampling ranges,
  base tensor shapes, and exact scale ratios from the case's executable shape
  workload,
- dynamic heavy cases can still have meaningful round-to-round spread,
- marginal call-observation summaries should be read together with round
  summaries and paired comparisons,
- independently materialized inputs avoid the older shared-input locality
  artifact, but paired scheduling does not eliminate genuine workload-stream
  variance.

### Dynamic evidence contract

The dynamic benchmark preserves the workload definition and two levels of
timing evidence:

1. **Workload metadata** records sampled and fixed dimensions, inclusive
   integer sampling ranges, base input and output shapes, aggregate element
   counts across those tensors, and exact ratios against the medium profile.
2. **Call observations** retain
   `(case, round, measured batch, repeat, library, order position, latency)`.
   Marginal count/median/IQR tables summarize these calls descriptively.
3. **Paired batch units** identify one measured batch inside one round. Repeated
   calls for the same unit are technical replications, not independent samples,
   and are averaged before comparison.

For each competitor, the reported point effect is:

```text
mean competitor latency across paired batch units
-------------------------------------------------
mean einf latency across paired batch units
```

The 95% interval is a deterministic percentile bootstrap. It resamples paired
batch units jointly across libraries within each observed round, preserving the
round strata and library pairing. The interval is conditional on the observed
run and rounds. It is not a p-value, does not turn marginal IQR overlap into a
significance test, and does not establish cross-machine or long-run temporal
generalization.

At least two measured batches per round are required; otherwise the script
rejects the comparison rather than emitting a degenerate interval.

### Dynamic raw receipt

When `--output report.md` is provided, the dynamic script also writes
`report.json` unless `--raw-output` selects another path. Schema v2 of the
versioned JSON receipt includes:

- environment and benchmark configuration,
- case identities and execution forms,
- case-specific workload dimensions, shapes, element counts, and exact scale
  ratios,
- per-library marginal and round summaries,
- round execution orders,
- every call observation with its pairing and order identity,
- paired effects, interval bounds, bootstrap seed, and resample count.

Workload ratios use integer `numerator` and `denominator` fields so downstream
analysis does not have to recover exact values from rounded decimals.

The raw receipt is written before the Markdown file. Keep it when a comparison
may need re-analysis; Markdown alone intentionally does not contain enough
information to reconstruct every pair.

The dynamic reports currently stored under `artifacts/bench/` predate this
receipt schema, and the archived fixed reports likewise retain only marginal
summaries. Their numeric tables remain historical descriptive snapshots, but
per-observation pairing and paired uncertainty cannot be reconstructed from the
archived Markdown.

## Expression Parity

Use `benchmarks/compare/expression_parity.py` to decide whether a gap comes from
the expression itself or from `einf` runtime overhead. It compares the target
with output-equivalent library and plain `torch` strategies on the same logical
batches. A contraction-only lower bound is reported separately because it does
not produce the final split outputs.

Example:

```bash
python -m benchmarks.compare.expression_parity \
  --scale large \
  --case einop_contract_split_dynamic \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --parity-checks 8 \
  --output artifacts/bench/2026-04-10-gap-expression-parity.md \
  --raw-output artifacts/bench/raw/2026-04-10-gap-expression-parity.json
```

Current built-in strategies include:

- `einf`
- `einops`
- `einx`
- `torch_matmul_only`
- `torch_matmul_split`
- `torch_matmul_slice`

For each measured batch and repeat, every available strategy receives an
independent clone of the same input. The logical batch is regenerated from the
same seed for each repeat instead of retaining every large tensor in memory.
This keeps peak memory near one batch while preserving the paired input. The
strategies run back-to-back, with their order rotated from one deterministic
shuffle across all measured coordinates. Warmup uses a separate continuous
rotation. Batch generation, cloning, and output observation are outside the
timed interval.

Parity checks run after timing on disposable strategy instances. A mismatch
aborts report generation, while `--parity-checks` cannot warm instance-local or
process-global caches before measurement.

The report estimates each competitor's mean batch latency relative to the
`einf` target. Repeats are averaged within each round and measured batch before
the ratio is computed. The 95% interval resamples batches within each round, so
repeated calls are not counted as independent inputs.

Keep the JSON receipt when the result may need re-analysis. It records the
schedule and estimand, per-round summaries, every call's round, batch, repeat,
strategy, order, and latency identity, and the paired estimates. The Markdown
report is meant for interpretation; it does not replace the raw evidence.

Read the two comparison sections differently:

- **Equivalent-output comparisons** show whether another complete expression
  performs the same work faster or slower than the target.
- **Lower-bound diagnostic** estimates how much time remains when the final
  split work is removed. It is a floor, not an interchangeable implementation.

If a plain `torch` split or slice strategy shows a material gap, inspect the
expression and tensor layout before profiling `einf` internals. If the
equivalent plain `torch` strategies stay close to the target while the lower
bound does not, the next useful step is internal overhead profiling.

## Layout Audit

Use `benchmarks/audit/expression_layout.py` when parity results alone do not explain a gap.

Example:

```bash
python -m benchmarks.audit.expression_layout \
  --scale large \
  --case einop_contract_split_dynamic \
  --seed 20260215 \
  --batches 32 \
  --top-k 3 \
  --output artifacts/bench/2026-04-10-gap-expression-layout.md \
  --raw-output artifacts/bench/raw/2026-04-10-gap-expression-layout.json
```

This audit records:

- input layout,
- intermediate contracted layout,
- output layout,
- fastest/slowest sample excerpts.

## Overhead Breakdown

Use `benchmarks/profile/overhead_breakdown.py` to decompose internal `einf` overhead into coarse runtime stages.

Example:

```bash
python -m benchmarks.profile.overhead_breakdown \
  --backend torch \
  --output artifacts/bench/2026-02-15-overhead-breakdown-torch.md \
  --raw-output artifacts/bench/raw/2026-02-15-overhead-breakdown-torch.json
```

This is an internal profiler, not a library-to-library fairness benchmark.

## Warm Call-Tree Profiling

Use `benchmarks/profile/warm_calltree.py` when you need a Python call-tree for a residual hot path after warmup.

Example:

```bash
python -m benchmarks.profile.warm_calltree \
  --backend torch \
  --mode dynamic \
  --scale large \
  --case einop_contract_split \
  --warmup 32 \
  --loops 128 \
  --sort cumtime \
  --top 40 \
  --output artifacts/bench/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.md \
  --raw-output artifacts/bench/raw/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.json
```

## LSP Latency Smoke

Use `benchmarks.profile.lsp_latency` to measure the editor-facing semantic
analysis path separately from cached hover, inlay-hint, and semantic-token
serving. Run it as a module from the repository root:

```bash
python -m benchmarks.profile.lsp_latency \
  --repeats 30 \
  --output artifacts/bench/lsp-latency.md \
  --json-output artifacts/bench/raw/lsp-latency.json
```

By default this does not invoke external checkers. Add `--checker` only when
you explicitly want to measure fallback save-boundary checker overhead:

```bash
python -m benchmarks.profile.lsp_latency \
  --repeats 3 \
  --checker basedpyright \
  --output artifacts/bench/lsp-latency-checker.md
```

## Guardrail Checks

Use `benchmarks/guardrail/check_overhead.py` to compare stored raw profiler outputs and fail on unacceptable regressions.

Example:

```bash
python -m benchmarks.guardrail.check_overhead \
  --baseline artifacts/bench/raw/2026-02-15-overhead-breakdown-post-6pv-10-torch.json \
  --candidate artifacts/bench/raw/2026-02-16-overhead-breakdown-torch.json \
  --metric instrumented_call_ms \
  --max-regression-ratio 0.10
```

## Reading the Reports

Use these interpretations consistently:

1. compare benchmarks answer “which library/runtime path is faster on the same logical workload?”
2. profile benchmarks answer “where is `einf` spending Python-side time internally?”
3. parity and layout audits answer “is the remaining gap expression-driven, layout-driven, or runtime-driven?”

Do not mix those contracts when reading the reports.
