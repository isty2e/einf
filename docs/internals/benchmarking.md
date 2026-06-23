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
4. measured time covers only the library call itself, not harness-side materialization.

More concretely:

- fixed warm benchmarks use paired per-call execution on the same logical input,
- dynamic benchmarks use paired same-batch execution on the same logical batch stream inside each round,
- dynamic rounds still use different batch streams from one round to the next, so round summaries matter for heavy cases.

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
  tests/test_package_import_boundaries.py \
  tests/test_array_api_execution.py \
  tests/test_tensorop_contract.py \
  tests/test_plan_hierarchy.py \
  tests/test_pipeline_contracts.py \
  tests/test_benchmark_harness.py \
  tests/test_benchmark_guardrails.py
```

Add narrower tests for the touched package when the ticket changes a specific
contract. Keep the import-boundary guard in the command so a migration cannot
close while adding a new forbidden package edge.

### Baseline Capture

Capture the baseline before applying the non-LSP migration. Use the same machine,
same dependency versions, and same shell environment for baseline and candidate.
If the machine was busy, slept, thermally throttled, or dependency versions
changed, discard the artifacts and rerun both sides.

```bash
BENCH_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_DIR="artifacts/bench/taxonomy-${BENCH_STAMP}"
mkdir -p "$BENCH_DIR/raw"

python benchmarks/profile/overhead_breakdown.py \
  --backend torch \
  --output "$BENCH_DIR/baseline-overhead-torch.md" \
  --raw-output "$BENCH_DIR/raw/baseline-overhead-torch.json"

python benchmarks/compare/einf_einops_einx.py \
  --backend torch \
  --scale large \
  --rounds 6 \
  --cold-repeats 3 \
  --warmup 4 \
  --warm-repeats 5 \
  --warm-iterations 60 \
  --output "$BENCH_DIR/baseline-fixed-large-torch.md"

python benchmarks/compare/einf_einops_einx_dynamic.py \
  --backend torch \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --parity-checks 8 \
  --output "$BENCH_DIR/baseline-dynamic-large-torch.md"
```

Expected baseline artifacts:

- `$BENCH_DIR/baseline-overhead-torch.md`
- `$BENCH_DIR/raw/baseline-overhead-torch.json`
- `$BENCH_DIR/baseline-fixed-large-torch.md`
- `$BENCH_DIR/baseline-dynamic-large-torch.md`

### Candidate Capture

Run the same commands after the migration, writing candidate artifacts into the
same `BENCH_DIR`:

```bash
python benchmarks/profile/overhead_breakdown.py \
  --backend torch \
  --output "$BENCH_DIR/candidate-overhead-torch.md" \
  --raw-output "$BENCH_DIR/raw/candidate-overhead-torch.json"

python benchmarks/compare/einf_einops_einx.py \
  --backend torch \
  --scale large \
  --rounds 6 \
  --cold-repeats 3 \
  --warmup 4 \
  --warm-repeats 5 \
  --warm-iterations 60 \
  --output "$BENCH_DIR/candidate-fixed-large-torch.md"

python benchmarks/compare/einf_einops_einx_dynamic.py \
  --backend torch \
  --scale large \
  --batches 64 \
  --warmup-batches 8 \
  --repeats 6 \
  --rounds 3 \
  --parity-checks 8 \
  --output "$BENCH_DIR/candidate-dynamic-large-torch.md"
```

Expected candidate artifacts:

- `$BENCH_DIR/candidate-overhead-torch.md`
- `$BENCH_DIR/raw/candidate-overhead-torch.json`
- `$BENCH_DIR/candidate-fixed-large-torch.md`
- `$BENCH_DIR/candidate-dynamic-large-torch.md`

### Guardrail Checks

The overhead profile covers the core TensorOp hot path in fixed and dynamic,
medium and large scenarios. Run both metrics:

```bash
python benchmarks/guardrail/check_overhead.py \
  --baseline "$BENCH_DIR/raw/baseline-overhead-torch.json" \
  --candidate "$BENCH_DIR/raw/candidate-overhead-torch.json" \
  --metric instrumented_call_ms \
  --max-regression-ratio 0.10

python benchmarks/guardrail/check_overhead.py \
  --baseline "$BENCH_DIR/raw/baseline-overhead-torch.json" \
  --candidate "$BENCH_DIR/raw/candidate-overhead-torch.json" \
  --metric unpatched_call_ms \
  --max-regression-ratio 0.10
```

Do not use `--allow-missing-cases` for this gate. Missing baseline cases are a
gate failure because they make regression status unknowable.

### Interpretation Rules

Close a non-LSP taxonomy migration only when all of these are true:

- focused correctness tests pass,
- both overhead guardrail commands exit 0,
- candidate compare reports do not show a new `einf` slowdown that contradicts
  the overhead guardrail,
- dynamic compare aggregate medians are read together with round median summaries,
- any observed 5-10 % slowdown is rerun or explained with case-level evidence,
- any slowdown above 10 % blocks the ticket unless the user explicitly accepts
  the regression.

For dynamic compare reports, compare libraries within the same round before
reading across rounds. The harness pairs libraries on the same logical batch
stream inside a round; different rounds may legitimately use different workload
streams.

If a guardrail fails, collect a call tree before changing benchmark policy:

```bash
python benchmarks/profile/warm_calltree.py \
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
python benchmarks/compare/einf_einops_einx.py \
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
- warm steady-state per-call timing,
- round-level warm summaries,
- per-case library order for each round.

## Dynamic-Shape Compare

Use `benchmarks/compare/einf_einops_einx_dynamic.py` when batch shapes vary.

Example:

```bash
python benchmarks/compare/einf_einops_einx_dynamic.py \
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

- dynamic heavy cases can still have meaningful round-to-round spread,
- aggregate medians should be read together with round summaries,
- same-batch paired scheduling removes the older shared-input locality artifact, but it does not eliminate genuine workload-stream variance.

## Expression Parity

Use `benchmarks/compare/expression_parity.py` when a gap case needs a more focused comparison against plain `torch` expression strategies.

Example:

```bash
python benchmarks/compare/expression_parity.py \
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

## Layout Audit

Use `benchmarks/audit/expression_layout.py` when parity results alone do not explain a gap.

Example:

```bash
python benchmarks/audit/expression_layout.py \
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
python benchmarks/profile/overhead_breakdown.py \
  --backend torch \
  --output artifacts/bench/2026-02-15-overhead-breakdown-torch.md \
  --raw-output artifacts/bench/raw/2026-02-15-overhead-breakdown-torch.json
```

This is an internal profiler, not a library-to-library fairness benchmark.

## Warm Call-Tree Profiling

Use `benchmarks/profile/warm_calltree.py` when you need a Python call-tree for a residual hot path after warmup.

Example:

```bash
python benchmarks/profile/warm_calltree.py \
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
python benchmarks/guardrail/check_overhead.py \
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
