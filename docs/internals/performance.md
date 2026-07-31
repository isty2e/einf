# Performance

This page records a historical torch benchmark snapshot against `einops` and
`einx` and discusses plausible mechanisms behind the gaps. Treat the results as
archived measurements, not current performance claims. The operation-level
"Why" sections propose explanations consistent with the measurements; this
benchmark does not decompose latency causally.

The archived reports use a retired CPU-only harness that materialized separate
inputs for each library. The current harness shares one prepared input per
paired coordinate and measures synchronized completion on the resolved device.
Do not compare these tables directly with receipts produced by the current
harness.

## Snapshot scope

- Date captured: 2026-04-17
- Backend: `torch` (CPU path)
- Hardware: macOS ARM64 (Apple Silicon), Python 3.11.10
- Libraries: `torch 2.11.0`, `numpy 2.4.4`, `einops 0.8.2`, `einx 0.4.3`,
  `einf` from `src/einf`
- Archived Markdown reports:
  - `artifacts/bench/bench-torch-medium.md` — static, medium shapes
  - `artifacts/bench/bench-torch-large.md` — static, large shapes
  - `artifacts/bench/bench-torch-medium-dynamic.md` — dynamic, medium
  - `artifacts/bench/bench-torch-large-dynamic.md` — dynamic, large
  - `artifacts/bench/SUMMARY.md` — aggregated tables

Absolute milliseconds are hardware-dependent. Relative gaps are usually
more portable than raw wall-clock numbers, but they still depend on
this machine, these library versions, and this benchmark harness.

For the current measurement contract, see [Benchmarking](benchmarking.md). All
numbers below are archived warm-state medians with `[p25 – p75]`
inter-quartile ranges from the retired harness.

The archived dynamic reports predate the current versioned raw observation
receipt, and the fixed reports likewise retain only marginal summaries. They
do not contain the identities needed to reconstruct paired effects or
bootstrap intervals. The discussion below is therefore descriptive: marginal
median and IQR separation is not a statistical significance or noise test.

## How to read these numbers

- **Static** cases use fixed-shape inputs and report steady state after
  warmup. Each of the 30 warm observations per library/case is the
  arithmetic mean of 60 timed calls. Their marginal IQRs describe
  observation spread; overlap or separation does not establish significance.
- **Dynamic** cases resample the dimensions declared dynamic for that case;
  the remaining dimensions stay fixed at their profile values. Each archived
  case reports 420 call observations per library/case: 84 round/batch workload
  units, each measured five times. Their marginal IQRs mix workload-shape
  variation and timing variation. Repeats are not independent workload units.
- In both suites every library sees the same logical input at each
  round position with independently materialised tensors and rotated
  order. That design spreads cache-state, first-executor, and order effects,
  but it does not prove that those effects or system jitter are absent.

Two scale families are reported, `M` (medium) and `L` (large). The medium
dimension bases are `b=16, n=192, d=96, h=32, w=24, r=16, j=128`; the large
bases are `b=24, n=384, d=128, h=48, w=32, r=24, j=192`. The exact
large-to-medium ratios are `3/2` for `b`, `h`, `r`, and `j`; `2` for `n`; and
`4/3` for `d` and `w`. Input and output volume therefore scale differently by
case.
Current generated reports list these ratios and each case's sampling ranges
from the executable workload definition.

## Per-op breakdown

### `rearrange_transpose` — `(b, n, d) → (b, d, n)`

```python
# einf
rearrange(ax[b, n, d], ax[b, d, n])(x)
# einops
einops.rearrange(x, "b n d -> b d n")
# einx
einx.rearrange("b n d -> b d n", x)
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0045 [0.0045 – 0.0046] | 0.0054 [0.0053 – 0.0055] | 0.0183 [0.0180 – 0.0192] |
| Static L | 0.0045 [0.0045 – 0.0047] | 0.0053 [0.0053 – 0.0056] | 0.0180 [0.0178 – 0.0191] |
| Dynamic M | 0.0110 [0.0097 – 0.0125] | 0.0133 [0.0115 – 0.0166] | 0.0400 [0.0325 – 0.0534] |
| Dynamic L | 0.0152 [0.0125 – 0.0185] | 0.0186 [0.0155 – 0.0243] | 0.0544 [0.0461 – 0.0720] |

The archived medians place einf roughly 15 – 20 % below einops on the
static scales and at about one-third to one-quarter of einx on all four
cells. The static marginal IQRs do not overlap.

**Why.** The best current explanation is that this operation reduces to
a single `torch.permute`, so per-call dispatch dominates. einf's hot
path caches the compiled runner by runtime input types, making a repeat
call close to one dict lookup plus the permute dispatch. einops still
walks a small recipe interpreter per call, and einx appears to pay
more per-call spec-processing overhead on this CPU path.

### `rearrange_flatten_hw` — `(b, h, w, d) → (b, h*w, d)`

```python
# einf
rearrange(ax[b, h, w, d], ax[b, (h * w), d])(x)
# einops
einops.rearrange(x, "b h w d -> b (h w) d")
# einx
einx.rearrange("b h w d -> b (h w) d", x)
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0049 [0.0049 – 0.0050] | 0.0052 [0.0051 – 0.0053] | 0.0176 [0.0173 – 0.0179] |
| Static L | 0.0049 [0.0049 – 0.0050] | 0.0052 [0.0052 – 0.0053] | 0.0178 [0.0175 – 0.0181] |
| Dynamic M | 0.0163 [0.0137 – 0.0204] | 0.0180 [0.0147 – 0.0242] | 0.0562 [0.0463 – 0.0742] |
| Dynamic L | 0.0371 [0.0288 – 0.0455] | 0.0395 [0.0305 – 0.0491] | 0.0961 [0.0835 – 0.1196] |

The archived einf medians are about 6 – 9 % below einops, with some
marginal IQRs touching at the edges. The einx medians are roughly
2.5 – 3.5× the einf medians.

**Why.** When the input is already contiguous, the operation is a
single `reshape` (or `view` when stride-compatible). All three
libraries emit a one-call torch op, so most of the remaining
difference is plausibly dispatch cost. The gap is smaller than
`rearrange_transpose` because reshape is slightly cheaper than permute,
so the same absolute dispatch saving is a smaller percentage of the
total.

### `rearrange_split_hw` — `(b, h*w, d) → (b, h, w, d)`

```python
# einf
rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=h, w=w)(x)
# einops
einops.rearrange(x, "b (h w) d -> b h w d", h=h, w=w)
# einx
einx.rearrange("b (h w) d -> b h w d", x, h=h, w=w)
```

This case runs only in the static suite — the split factors `h, w` are
fixed at plan time, so a varying-shape input stream would change the
problem rather than stress the same op.

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0055 [0.0055 – 0.0056] | 0.0060 [0.0060 – 0.0061] | 0.0189 [0.0188 – 0.0191] |
| Static L | 0.0058 [0.0057 – 0.0059] | 0.0064 [0.0063 – 0.0065] | 0.0198 [0.0194 – 0.0206] |

The archived einf median is about 8 – 9 % below einops at both scales;
their marginal IQRs do not overlap.

**Why.** This is symmetric to the flatten case: one reshape call with
the output sizes supplied via `with_sizes`. The current explanation is
that einf resolves the split factor once at plan-build time, while
einops still has some per-call kwarg validation and recipe handling.

### `repeat_expand_axis` — `(b, d) → (b, d, r)`

```python
# einf
repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)
# einops
einops.repeat(x, "b d -> b d r", r=r)
# einx (einx has no first-class repeat; rearrange handles the broadcast)
einx.rearrange("b d -> b d r", x, r=r)
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0056 [0.0054 – 0.0058] | 0.0063 [0.0062 – 0.0066] | 0.0194 [0.0189 – 0.0206] |
| Static L | 0.0057 [0.0056 – 0.0058] | 0.0065 [0.0064 – 0.0066] | 0.0190 [0.0188 – 0.0194] |
| Dynamic M | 0.0060 [0.0059 – 0.0065] | 0.0072 [0.0069 – 0.0079] | 0.0210 [0.0202 – 0.0238] |
| Dynamic L | 0.0058 [0.0057 – 0.0063] | 0.0068 [0.0067 – 0.0076] | 0.0202 [0.0199 – 0.0227] |

The archived einf medians are about 12 – 17 % below einops, with
non-overlapping marginal IQRs in all four cells. The einx medians are
roughly 3.3 – 3.5× the einf medians.

**Why.** This is pure broadcast — `torch.expand` (or equivalent
unsqueeze + expand). There is no data movement, so the benchmark is
mostly exposing dispatch cost. This is where the compiled-plan
advantage is most visible: the hot path is very small, so any extra
per-call interpretation shows up directly.

### `reduce_sum_axes` — `(b, h, w, d) → (b, d)`

```python
# einf
reduce(ax[b, h, w, d], ax[b, d])(x)
# einops
einops.reduce(x, "b h w d -> b d", "sum")
# einx
einx.sum("b h w d -> b d", x)
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0895 [0.0884 – 0.0932] | 0.0952 [0.0928 – 0.0981] | 0.1217 [0.1187 – 0.1257] |
| Static L | 0.3176 [0.3119 – 0.3262] | 0.3221 [0.3146 – 0.3363] | 0.3573 [0.3497 – 0.3795] |
| Dynamic M | 0.1178 [0.0966 – 0.1453] | 0.1289 [0.1079 – 0.1563] | 0.1774 [0.1439 – 0.2384] |
| Dynamic L | 0.3294 [0.2430 – 0.4170] | 0.3448 [0.2575 – 0.4437] | 0.4248 [0.3261 – 0.6985] |

The archived einf median is about 6 % below einops at static M, where
the marginal IQRs touch, and the static-L medians are close. Dynamic
einf medians are 4 – 9 % lower, with overlapping marginal IQRs.

**Why.** `torch.sum` does the actual work and that cost scales with the
reduced volume, so at large shapes any dispatch saving becomes a small
fraction of the wall clock. The table is consistent with a dispatch
advantage that gets diluted as real compute starts to dominate.

### `contract_matmul` — `(b, n, d) × (d, j) → (b, n, j)`

```python
# einf
contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)
# einops
einops.einsum(lhs, rhs, "b n d, d j -> b n j")
# einx
einx.dot("b n d, d j -> b n j", lhs, rhs)
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.0819 [0.0811 – 0.0842] | 0.0918 [0.0909 – 0.0940] | 0.1093 [0.1076 – 0.1139] |
| Static L | 0.6688 [0.6634 – 0.6746] | 0.6906 [0.6842 – 0.6941] | 0.7230 [0.7169 – 0.7342] |
| Dynamic M | 0.2996 [0.1616 – 0.4244] | 0.3051 [0.1813 – 0.4392] | 0.3984 [0.2327 – 0.7447] |
| Dynamic L | 0.7401 [0.4795 – 1.0669] | 0.7776 [0.4899 – 1.0866] | 0.9231 [0.5876 – 1.7038] |

The archived einf median is about 11 % below einops at static M and
about 3 % below at static L; their marginal IQRs do not overlap.
Dynamic einf medians are lower, with substantial marginal IQR overlap.

**Why.** The current explanation is that einf's contract path picks the
backend contraction call once at plan time, so less path-planning work
is left for each call. einops and einx still carry more call-time spec
or validation work. At large shapes, BLAS dominates and any dispatch
saving shrinks to a single-digit percentage.

### `einop_contract_split` — contract-then-split, two-output

```python
# einf — one fused TensorOp, two outputs from a single plan
einop(
    (ax[b, ((h + w) * r), n], ax[n, d]),
    (ax[b, (h * r), d], ax[b, (w * r), d]),
).with_sizes(h=h, w=w, r=r)(lhs, rhs)

# einops — explicit contract then slice
tmp = einops.einsum(lhs, rhs, "b t n, n d -> b t d")
out_h, out_w = tmp[:, : h * r, :], tmp[:, h * r :, :]

# einx — same three-call path with einx.dot
tmp = einx.dot("b t n, n d -> b t d", lhs, rhs)
out_h, out_w = tmp[:, : h * r, :], tmp[:, h * r :, :]
```

| Scale | einf | einops | einx |
|---|---|---|---|
| Static M | 0.8109 [0.7991 – 0.8362] | 0.8302 [0.8119 – 0.8449] | 0.8697 [0.8560 – 0.8911] |
| Static L | 4.6868 [4.6594 – 4.7318] | 4.7129 [4.6788 – 4.7667] | 4.7783 [4.7330 – 4.8267] |
| Dynamic M | 0.8580 [0.6217 – 1.1867] | 0.9417 [0.6520 – 1.2222] | 1.0874 [0.7335 – 1.5821] |
| Dynamic L | 5.4291 [4.3414 – 7.3027] | 5.5144 [4.3879 – 7.3904] | 6.2577 [4.7312 – 10.8784] |

The archived einf median ranges from about 1 – 9 % below einops across
the four cells, with overlapping marginal IQRs. The einx median is higher
in every row.

**Why.** This is a fused-signature case: one `einop` call produces two
outputs from a contract-then-split shape. einf expresses that as one
compiled plan, but the underlying backend work is still one einsum plus
two tensor slices. The table is consistent with the view that, once
those backend calls dominate, any compiled-plan dispatch saving is a
small fraction of the total. The fused signature here looks more like a
*modelling* win than a runtime win.

## Where the gap comes from, overall

The descriptive median pattern across all seven cases is:

1. **Dispatch-bound ops** (rearrange, repeat) — einf medians are
   5 – 20 % lower than einops. The data is consistent with the idea that
   caching the full runner removes enough per-call interpretation to matter.
2. **Light compute** (reduce, contract at medium shape) — einf medians
   are lower, while backend compute is a larger share of wall-clock.
3. **Heavy compute** (reduce L, contract L, fused contract+split at
   either scale) — the einf/einops median gap is generally smaller. That
   pattern is consistent with backend compute taking a larger share of
   wall-clock time.

In this snapshot, einx has the highest median on every case. On
rearrange-family calls its median is roughly 2.5 – 4× the einf median, which is
consistent with higher per-call overhead on dispatch-bound paths. On
reduce and contraction paths the median gap is generally smaller.

## What this does not prove

- **No "N× faster" global headline.** The roughly 2.5 – 4× range versus einx is
  specific to dispatch-tier rearrange and repeat cases. Averaging it
  across contraction cases or large shapes would overstate the result.
- **No statistical significance claim.** Marginal IQR overlap or separation
  is descriptive, not an inferential test. The 30 static observations each
  aggregate 60 calls; the 420 dynamic call observations represent 84
  round/batch workload units repeated five times. The archived Markdown does
  not retain the raw identities needed for the current paired estimator.
- **No causal decomposition.** The operation-level explanations are consistent
  with the observed scaling and execution forms, but this snapshot does not
  separately measure dispatch, backend compute, or other latency components.
- **No GPU data.** Everything above is CPU torch. The dispatch tier is
  where Python overhead is most visible; on GPU, CUDA launch time
  changes the shape of the comparison and nothing on this page should
  be assumed to transfer.
- **No long-run drift guarantee.** einops and einx continue to
  improve. These tables are a snapshot of 2026-04-17 versions. Re-run
  the suite before quoting numbers in a new context.

## Running the current harness

From the repo root, in an environment with `einf`, `einops`, `einx`,
and `torch` installed:

```bash
python -m benchmarks.compare.einf_einops_einx \
  --backend torch --device cpu --scale medium \
  --rounds 6 --warmup 4 --repeats 5 --iterations 60 \
  --output artifacts/bench/current/bench-torch-medium.md

python -m benchmarks.compare.einf_einops_einx \
  --backend torch --device cpu --scale large \
  --rounds 6 --warmup 4 --repeats 5 --iterations 60 \
  --output artifacts/bench/current/bench-torch-large.md

python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch --device cpu --scale medium \
  --batches 32 --warmup-batches 4 --repeats 5 --rounds 3 \
  --output artifacts/bench/current/bench-torch-medium-dynamic.md \
  --raw-output artifacts/bench/current/raw/bench-torch-medium-dynamic.json

python -m benchmarks.compare.einf_einops_einx_dynamic \
  --backend torch --device cpu --scale large \
  --batches 32 --warmup-batches 4 --repeats 5 --rounds 3 \
  --output artifacts/bench/current/bench-torch-large-dynamic.md \
  --raw-output artifacts/bench/current/raw/bench-torch-large-dynamic.json
```

Running these commands now produces current-harness Markdown and raw JSON
receipts. It does not recreate the archived 2026-04-17 environment, measurement
contract, or missing pairing identities. Use new output paths rather than
overwriting the historical artifacts listed above.

See [Benchmarking](benchmarking.md) for the full methodology, other
backends, and audit tools.
