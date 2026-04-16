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
  --output docs/benchmarks/2026-02-15-einf-vs-einops-einx-large-torch.md
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
  --output docs/benchmarks/2026-02-15-einf-vs-einops-einx-dynamic-large-target-r3-torch.md
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
  --output docs/benchmarks/2026-04-10-gap-expression-parity.md \
  --raw-output docs/benchmarks/raw/2026-04-10-gap-expression-parity.json
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
  --output docs/benchmarks/2026-04-10-gap-expression-layout.md \
  --raw-output docs/benchmarks/raw/2026-04-10-gap-expression-layout.json
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
  --output docs/benchmarks/2026-02-15-overhead-breakdown-torch.md \
  --raw-output docs/benchmarks/raw/2026-02-15-overhead-breakdown-torch.json
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
  --output docs/benchmarks/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.md \
  --raw-output docs/benchmarks/raw/2026-04-09-warm-calltree-dynamic-large-einop-contract-split.json
```

## Guardrail Checks

Use `benchmarks/guardrail/check_overhead.py` to compare stored raw profiler outputs and fail on unacceptable regressions.

Example:

```bash
python benchmarks/guardrail/check_overhead.py \
  --baseline docs/benchmarks/raw/2026-02-15-overhead-breakdown-post-6pv-10-torch.json \
  --candidate docs/benchmarks/raw/2026-02-16-overhead-breakdown-torch.json \
  --metric instrumented_call_ms \
  --max-regression-ratio 0.10
```

## Reading the Reports

Use these interpretations consistently:

1. compare benchmarks answer “which library/runtime path is faster on the same logical workload?”
2. profile benchmarks answer “where is `einf` spending Python-side time internally?”
3. parity and layout audits answer “is the remaining gap expression-driven, layout-driven, or runtime-driven?”

Do not mix those contracts when reading the reports.
