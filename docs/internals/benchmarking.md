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

The receipt-producing compare, audit, and profile commands described here write
Markdown to stdout. Use `--receipt PATH` to publish their canonical JSON
receipt. Publication uses a same-directory temporary file and atomic
replacement, so a failed write does not leave a partial receipt at the
destination. Redirect stdout when you also want to retain the Markdown
projection; create the projection's parent directory before running the
command because the shell opens that file first.

## Compare Methodology

Fixed and dynamic comparisons use the same measurement contract. Their only
substantive difference is how they construct the input stream: fixed cases reuse
one prepared batch, while dynamic cases draw deterministic batches whose shapes
can vary by coordinate.

At each measured coordinate, the harness:

1. selects one already-prepared backend batch,
2. passes the same tuple and tensor objects to every available library,
3. synchronizes the target before starting the host clock,
4. calls one library,
5. waits for that call's target work to complete before stopping the clock.

Runner construction, warmup, input generation, device transfer, reference
evaluation, numerical comparison, device-to-host transfer, summary calculation,
and serialization are outside the timed interval. The result is per-call
completion latency after inputs and runners are ready, not input-pipeline latency
or asynchronous launch latency.

Reference functions receive private NumPy snapshots and are expected to be
deterministic and side-effect free. The harness never passes a prepared backend
batch to reference code, so accidental mutation cannot change the measured
workload or another strategy's reference input.

Fixed cases prepare one batch and reuse it across measured coordinates. Dynamic
cases materialize one target batch per coordinate before the first library runs.
After each timed call has completed and its timer has stopped, the harness checks
that call's returned output against the reference. It then releases the output
before moving on. This covers every repeat without retaining the input stream or
more than one call's output.

Library order rotates at each paired coordinate to spread first-executor and
position effects. Fixed calls use repeats as paired units and iterations as
technical replications. Dynamic calls use measured batches as paired units and
repeats as technical replications. Dynamic rounds draw different deterministic
batch streams, so heavy cases should still be read with their round summaries.

NumPy comparisons run on CPU. Torch comparisons accept backend-native device
names through `--device`, such as `cpu`, `mps`, or `cuda:0`. The harness resolves
that target once and fails before measurement if it cannot allocate,
synchronize, or keep outputs on the requested device.

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
    --receipt "$REPO_ROOT/$BENCH_DIR/raw/baseline-overhead-order-a-torch.json" \
    > "$REPO_ROOT/$BENCH_DIR/baseline-overhead-order-a-torch.md"
)

python -m benchmarks.profile.overhead_breakdown \
  --backend torch \
  --receipt "$BENCH_DIR/raw/candidate-overhead-order-a-torch.json" \
  > "$BENCH_DIR/candidate-overhead-order-a-torch.md"
```

Order B, candidate first:

```bash
python -m benchmarks.profile.overhead_breakdown \
  --backend torch \
  --receipt "$BENCH_DIR/raw/candidate-overhead-order-b-torch.json" \
  > "$BENCH_DIR/candidate-overhead-order-b-torch.md"

(
  cd "$BASE_WORKTREE"
  python -m benchmarks.profile.overhead_breakdown \
    --backend torch \
    --receipt "$REPO_ROOT/$BENCH_DIR/raw/baseline-overhead-order-b-torch.json" \
    > "$REPO_ROOT/$BENCH_DIR/baseline-overhead-order-b-torch.md"
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
    --device cpu \
    --scale large \
    --rounds 6 \
    --warmup 4 \
    --repeats 5 \
    --iterations 60 \
    --receipt "$REPO_ROOT/$BENCH_DIR/raw/baseline-fixed-large-torch.json" \
    > "$REPO_ROOT/$BENCH_DIR/baseline-fixed-large-torch.md"
)

python -m benchmarks.compare.einf_einops_einx \
  --backend torch \
  --device cpu \
  --scale large \
  --rounds 6 \
  --warmup 4 \
  --repeats 5 \
  --iterations 60 \
  --receipt "$BENCH_DIR/raw/candidate-fixed-large-torch.json" \
  > "$BENCH_DIR/candidate-fixed-large-torch.md"

(
  cd "$BASE_WORKTREE"
  python -m benchmarks.compare.einf_einops_einx_dynamic \
    --backend torch \
    --device cpu \
    --scale large \
    --batches 64 \
    --warmup-batches 8 \
    --repeats 6 \
    --rounds 3 \
    --receipt "$REPO_ROOT/$BENCH_DIR/raw/baseline-dynamic-large-torch.json" \
    > "$REPO_ROOT/$BENCH_DIR/baseline-dynamic-large-torch.md"
)

python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch \
  --device cpu \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --receipt "$BENCH_DIR/raw/candidate-dynamic-large-torch.json" \
  > "$BENCH_DIR/candidate-dynamic-large-torch.md"
```

Expected compare artifacts:

- `$BENCH_DIR/baseline-fixed-large-torch.md`
- `$BENCH_DIR/raw/baseline-fixed-large-torch.json`
- `$BENCH_DIR/baseline-dynamic-large-torch.md`
- `$BENCH_DIR/raw/baseline-dynamic-large-torch.json`
- `$BENCH_DIR/candidate-fixed-large-torch.md`
- `$BENCH_DIR/raw/candidate-fixed-large-torch.json`
- `$BENCH_DIR/candidate-dynamic-large-torch.md`
- `$BENCH_DIR/raw/candidate-dynamic-large-torch.json`

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
Each path passed to a repeated-trial check must identify a distinct existing
receipt. Reusing a receipt within or across pairs would count the same evidence
more than once, so the command rejects aliases before loading any report.

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
  --receipt "$BENCH_DIR/raw/candidate-warm-calltree-dynamic-large-einop-contract-split.json" \
  > "$BENCH_DIR/candidate-warm-calltree-dynamic-large-einop-contract-split.md"
```

Use the call tree to explain the regression source. Do not relax the guardrail
threshold to make a migration pass.

## Fixed-Shape Compare

Use `benchmarks/compare/einf_einops_einx.py` for fixed-shape steady-state
comparisons.

Example:

```bash
mkdir -p artifacts/bench/current/raw

python -m benchmarks.compare.einf_einops_einx \
  --backend torch \
  --device cpu \
  --scale large \
  --rounds 6 \
  --warmup 4 \
  --repeats 3 \
  --iterations 60 \
  --receipt artifacts/bench/current/raw/fixed-large-torch.json \
  > artifacts/bench/current/fixed-large-torch.md
```

What the script reports:

- individual steady-state call observations,
- paired latency ratios and round-stratified intervals,
- round-level summaries,
- per-case library order for each round,
- requested and resolved execution devices,
- a versioned JSON receipt with the measured source revision.

## Dynamic-Shape Compare

Use `benchmarks/compare/einf_einops_einx_dynamic.py` when batch shapes vary.

Example:

```bash
python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch \
  --device cpu \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --receipt artifacts/bench/current/raw/dynamic-large-torch.json \
  > artifacts/bench/current/dynamic-large-torch.md
```

What matters here:

- each report derives sampled and fixed dimensions, inclusive sampling ranges,
  base tensor shapes, and exact scale ratios from the case's executable shape
  workload,
- dynamic heavy cases can still have meaningful round-to-round spread,
- marginal call-observation summaries should be read together with round
  summaries and paired comparisons,
- one target batch is materialized per paired coordinate and shared by all
  libraries,
- host and target inputs are released before the next coordinate,
- every distinct measured coordinate is numerically checked after timing and
  before report generation,
- repeats regenerate the same logical unit from its recorded seed, and each
  receipt includes the realized input shapes.

### Paired evidence contract

Fixed and dynamic comparison runs use the same observation and comparison
contract. Each timed call retains:

`(phase, case, round, unit, repeat, library, order position, latency)`.

The meaning of `unit` follows the measurement schedule:

- a fixed unit is one repeat block; its timed iterations are technical
  replications,
- a dynamic unit is one measured batch; its repeats are technical
  replications.

Replications are averaged within each `(round, unit)` before comparison.

For each competitor, the reported point effect is:

```text
mean competitor latency across paired units
---------------------------------------------
mean einf latency across paired units
```

The 95% interval is a deterministic percentile bootstrap. It resamples paired
units jointly across libraries within each observed round. The interval is
conditional on the observed workload and run. It is not a p-value or evidence
of cross-machine or long-run generalization.

Each measured phase requires at least two paired units per round. The scripts
reject smaller configurations rather than emit a degenerate interval.

### Canonical receipts

Pass `--receipt report.json` to the comparison, layout-audit, overhead, and
warm-call-tree commands described on this page. The JSON receipt is the
canonical artifact; Markdown is a stdout projection for reading or redirection.
Fixed and dynamic receipts use schema v6 and keep the same single `steady`
measurement phase. Each receipt includes:

- environment and benchmark configuration; `environment.einf` identifies the
  imported source as a Git checkout or installed distribution,
- requested and resolved execution devices,
- the synchronized-completion timed-region contract,
- case identities and execution forms,
- per-library marginal and round summaries,
- round execution orders,
- every call observation with its pairing and order identity,
- paired effects, interval bounds, bootstrap seed, and resample count.

Dynamic receipts also record case-specific workload dimensions, shapes, element
counts, and exact scale ratios. Each `realized_input_units` entry records the
round, unit, stream, derived seed, and input shapes used for that paired unit.
The case-level `validation` object identifies the measured coordinate source,
participating libraries, executions per coordinate, and the fact that the timed
call's own return value was checked immediately after its timer stopped.
Ratios use integer `numerator` and `denominator` fields rather than rounded
decimals.

Expression-parity receipts use schema v5. They share the source-provenance
contract used by the fixed and dynamic scripts, but keep their
strategy-specific result shape.

Receipt publication serializes the complete document before atomically
replacing the destination. Concurrent writers are last-writer-wins, but a
reader sees one complete receipt rather than a partially overwritten file.
If `--receipt` points to a symbolic link, the link remains in place and
publication atomically replaces its resolved target.
New receipts use `0666` access permissions filtered by the process umask.
Replacing an existing regular receipt preserves its access permission bits.
The temporary file containing receipt data remains private during serialization.
Atomic replacement creates a new inode, so ownership, hard-link identity, ACLs,
and extended attributes are not preservation guarantees.
Keep the receipt when a comparison may need re-analysis; Markdown alone does
not contain enough information to reconstruct every pair.

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
  --device cpu \
  --scale large \
  --case einop_contract_split_dynamic \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --receipt artifacts/bench/current/raw/expression-parity-large.json \
  > artifacts/bench/current/expression-parity-large.md
```

Current built-in strategies include:

- `einf`
- `einops`
- `einx`
- `torch_matmul_only`
- `torch_matmul_split`
- `torch_matmul_slice`

For each measured batch and repeat, every available strategy receives the same
prepared tuple and tensor objects. The logical batch is regenerated from the
same seed for each repeat, materialized on the target once, and released before
the next coordinate. The strategies run back-to-back, with their order rotated
from one deterministic shuffle across all measured coordinates. Warmup uses a
separate continuous rotation.

Every timed strategy output is checked against its reference after that call's
timer stops. A mismatch aborts report generation. The check covers every repeat,
not a sample of shapes or a separately generated output.

Expression timing uses the same synchronized-completion boundary as the fixed
and dynamic comparisons. Generation, device transfer, and output validation are
outside the timed interval.

The report estimates each competitor's mean batch latency relative to the
`einf` target. Repeats are averaged within each round and measured batch before
the ratio is computed. The 95% interval resamples batches within each round, so
repeated calls are not counted as independent inputs.

Keep the JSON receipt when the result may need re-analysis. It records the
schedule and estimand, per-round summaries, every call's round, batch, repeat,
strategy, order, and latency identity, and the paired estimates. The Markdown
report is meant for interpretation; it does not replace the receipt evidence.
Expression receipt schema v5 also records structured environment metadata, the
requested and resolved device, the shared synchronized-completion contract, and
the exact measured executions covered by validation.

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
mkdir -p artifacts/bench/raw

python -m benchmarks.audit.expression_layout \
  --scale large \
  --case einop_contract_split_dynamic \
  --seed 20260215 \
  --batches 32 \
  --top-k 3 \
  --receipt artifacts/bench/raw/2026-04-10-gap-expression-layout.json \
  > artifacts/bench/2026-04-10-gap-expression-layout.md
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
  --receipt artifacts/bench/raw/2026-02-15-overhead-breakdown-torch.json \
  > artifacts/bench/2026-02-15-overhead-breakdown-torch.md
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
  --receipt artifacts/bench/raw/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.json \
  > artifacts/bench/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.md
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
