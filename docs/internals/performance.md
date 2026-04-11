# Performance

This page reports current benchmark results against `einops` and `einx`
on the torch backend and explains where each gap comes from. It is
meant to be read as a snapshot with receipts, not a marketing line.

## Snapshot scope

- Date captured: 2026-04-17
- Backend: `torch` (CPU path)
- Hardware: macOS ARM64 (Apple Silicon), Python 3.11.10
- Libraries: `torch 2.11.0`, `numpy 2.4.4`, `einops 0.8.2`, `einx 0.4.3`,
  `einf` from `src/einf`
- Raw reports (re-runnable):
  - `artifacts/bench/bench-torch-medium.md` — static, medium shapes
  - `artifacts/bench/bench-torch-large.md` — static, large shapes
  - `artifacts/bench/bench-torch-medium-dynamic.md` — dynamic, medium
  - `artifacts/bench/bench-torch-large-dynamic.md` — dynamic, large
  - `artifacts/bench/SUMMARY.md` — aggregated tables

Absolute milliseconds are hardware-dependent. Relative gaps are usually
more portable than raw wall-clock numbers, but they still depend on
this machine, these library versions, and this benchmark harness.

For how the measurements are produced, see
[Benchmarking](benchmarking.md). All numbers below are warm-state
medians with `[p25 – p75]` inter-quartile ranges, from the same paired
scheduling harness described there.

## How to read these numbers

- **Static** cases use fixed-shape inputs and report per-call steady
  state after warmup. 30 warm samples per library per case. IQRs are
  narrow (tight measurement), so non-overlapping IQRs between two
  libraries constitute a real gap at this shape.
- **Dynamic** cases resample every batch dimension in `[0.6×, 1.4×]`
  of the base size and report per-batch timing. 420 batches per
  library per case. IQRs are wide because the workload itself varies
  shape from call to call; the median is well-settled but single-shot
  gap calls are weaker than on static.
- In both suites every library sees the same logical input at each
  round position with independently materialised tensors and rotated
  order. That strongly reduces cache-state, first-executor, and
  order-bias effects, so the remaining differences are more likely to
  come from dispatch overhead plus the cost of the library's chosen
  backend call.

Two scale families are reported, `M` (medium) and `L` (large).
Dimension bases: `b=16, n=192, d=96, h=32, w=24, r=16, j=128` at medium;
4× larger per applicable axis at large. Large shapes push per-call
compute up; dispatch overhead becomes a smaller fraction of the total.

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

einf is roughly 15 – 20 % ahead of einops with non-overlapping IQRs on
both static scales, and 3 – 4× ahead of einx on all four cells.

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

Gap to einops is real but small (~6 %, IQRs touch at the edges). einx
remains 3 – 4× behind.

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

einf is ~8 – 9 % ahead of einops with non-overlapping IQRs at both
scales.

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

einf ~12 – 17 % ahead of einops, non-overlapping IQRs on all four
cells. einx 3× behind.

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

~6 % ahead of einops at static M (IQRs touch), effectively tied at
static L. Dynamic medians favour einf by 4 – 9 % but IQRs overlap.

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

Static M: ~11 % ahead of einops with non-overlapping IQRs — the one
contraction case where the gap is clearly more than noise. Static L:
~3 % ahead, IQRs just non-overlapping. Dynamic: medians favour einf
but IQR overlap is heavy, so the ordering is weaker.

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

einf and einops are tied within noise at all four scales (IQRs
overlap). einx is consistently 5 – 15 % behind.

**Why.** This is a fused-signature case: one `einop` call produces two
outputs from a contract-then-split shape. einf expresses that as one
compiled plan, but the underlying backend work is still one einsum plus
two tensor slices. The table is consistent with the view that, once
those backend calls dominate, any compiled-plan dispatch saving is a
small fraction of the total. The fused signature here looks more like a
*modelling* win than a runtime win.

## Where the gap comes from, overall

The current pattern across all seven cases is:

1. **Dispatch-bound ops** (rearrange, repeat) — einf wins 10 – 20 %
   over einops. The data is consistent with the idea that caching the
   full runner removes enough per-call interpretation to matter here.
2. **Light compute** (reduce, contract at medium shape) — einf wins
   by a smaller single-digit margin; BLAS is starting to dominate
   wall-clock but dispatch is still visible.
3. **Heavy compute** (reduce L, contract L, fused contract+split at
   either scale) — parity with einops. The backend call dominates.
   There is nothing to win on the dispatch tier when you are already
   4.7 ms into a single einsum.

In this snapshot, einx is consistently the slowest on every case. On
rearrange-family calls the gap is large (3 – 4×), which is consistent
with higher per-call overhead on dispatch-bound paths. On contraction
the gap narrows to 5 – 15 %.

## What this does not prove

- **No "N× faster" global headline.** The 3 – 4× figure versus einx is
  specific to dispatch-tier rearrange and repeat cases. Averaging it
  across contraction cases or large shapes is dishonest.
- **No significance claim beyond IQR overlap.** 30 static samples and
  420 dynamic samples are enough to pin medians. A proper paired
  Wilcoxon test on per-sample deltas would need the raw paired data
  and is not reported here. Where the table says "non-overlapping IQR"
  the gap is visible; where it says "IQR overlap" the call is
  median-only and weaker.
- **No GPU data.** Everything above is CPU torch. The dispatch tier is
  where Python overhead is most visible; on GPU, CUDA launch time
  changes the shape of the comparison and nothing on this page should
  be assumed to transfer.
- **No long-run drift guarantee.** einops and einx continue to
  improve. These tables are a snapshot of 2026-04-17 versions. Re-run
  the suite before quoting numbers in a new context.

## Reproducing

From the repo root, in an environment with `einf`, `einops`, `einx`,
and `torch` installed:

```bash
python benchmarks/compare/einf_einops_einx.py \
  --backend torch --scale medium \
  --rounds 6 --cold-repeats 3 --warmup 4 --warm-repeats 5 --warm-iterations 60 \
  --output artifacts/bench/bench-torch-medium.md

python benchmarks/compare/einf_einops_einx.py \
  --backend torch --scale large \
  --rounds 6 --cold-repeats 3 --warmup 4 --warm-repeats 5 --warm-iterations 60 \
  --output artifacts/bench/bench-torch-large.md

python benchmarks/compare/einf_einops_einx_dynamic.py \
  --backend torch --scale medium \
  --batches 32 --warmup-batches 4 --repeats 5 --rounds 3 \
  --output artifacts/bench/bench-torch-medium-dynamic.md

python benchmarks/compare/einf_einops_einx_dynamic.py \
  --backend torch --scale large \
  --batches 32 --warmup-batches 4 --repeats 5 --rounds 3 \
  --output artifacts/bench/bench-torch-large-dynamic.md
```

See [Benchmarking](benchmarking.md) for the full methodology, other
backends, and audit tools.
