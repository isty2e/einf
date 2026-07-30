# Benchmark summary — `einf` vs `einops` vs `einx`

Torch backend. Fresh venv (`torch 2.11.0`, `numpy 2.4.4`,
`einops 0.8.2`, `einx 0.4.3`), macOS ARM64, Python 3.11.10.

> Historical descriptive snapshot. The dynamic Markdown reports predate the
> current raw observation receipt, and the fixed reports likewise retain only
> marginal summaries. Neither format retains the observation identities needed
> to reconstruct paired effects or uncertainty intervals. Numeric tables are
> preserved as captured; marginal IQR overlap or separation is not a
> significance test.

Archived reports:

- `artifacts/bench/bench-torch-medium.md`
- `artifacts/bench/bench-torch-large.md`
- `artifacts/bench/bench-torch-medium-dynamic.md`
- `artifacts/bench/bench-torch-large-dynamic.md`

Configs:

- Static: `--rounds 6 --cold-repeats 3 --warmup 4 --warm-repeats 5
  --warm-iterations 60` → 30 warm observations per library per case,
  each the arithmetic mean of 60 timed calls.
- Dynamic: `--batches 32 --warmup-batches 4 --repeats 5 --rounds 3`
  → 420 call observations per library per case: 84 round/batch workload
  units, each measured five times.

## Static (warm, ms — median [p25 – p75])

| Case | einf M | einops M | einx M |
|---|---|---|---|
| rearrange_transpose    | 0.0045 [0.0045 – 0.0046] | 0.0054 [0.0053 – 0.0055] | 0.0183 [0.0180 – 0.0192] |
| rearrange_flatten_hw   | 0.0049 [0.0049 – 0.0050] | 0.0052 [0.0051 – 0.0053] | 0.0176 [0.0173 – 0.0179] |
| rearrange_split_hw     | 0.0055 [0.0055 – 0.0056] | 0.0060 [0.0060 – 0.0061] | 0.0189 [0.0188 – 0.0191] |
| repeat_expand_axis     | 0.0056 [0.0054 – 0.0058] | 0.0063 [0.0062 – 0.0066] | 0.0194 [0.0189 – 0.0206] |
| reduce_sum_axes        | 0.0895 [0.0884 – 0.0932] | 0.0952 [0.0928 – 0.0981] | 0.1217 [0.1187 – 0.1257] |
| contract_matmul        | 0.0819 [0.0811 – 0.0842] | 0.0918 [0.0909 – 0.0940] | 0.1093 [0.1076 – 0.1139] |
| einop_contract_split   | 0.8109 [0.7991 – 0.8362] | 0.8302 [0.8119 – 0.8449] | 0.8697 [0.8560 – 0.8911] |

| Case | einf L | einops L | einx L |
|---|---|---|---|
| rearrange_transpose    | 0.0045 [0.0045 – 0.0047] | 0.0053 [0.0053 – 0.0056] | 0.0180 [0.0178 – 0.0191] |
| rearrange_flatten_hw   | 0.0049 [0.0049 – 0.0050] | 0.0052 [0.0052 – 0.0053] | 0.0178 [0.0175 – 0.0181] |
| rearrange_split_hw     | 0.0058 [0.0057 – 0.0059] | 0.0064 [0.0063 – 0.0065] | 0.0198 [0.0194 – 0.0206] |
| repeat_expand_axis     | 0.0057 [0.0056 – 0.0058] | 0.0065 [0.0064 – 0.0066] | 0.0190 [0.0188 – 0.0194] |
| reduce_sum_axes        | 0.3176 [0.3119 – 0.3262] | 0.3221 [0.3146 – 0.3363] | 0.3573 [0.3497 – 0.3795] |
| contract_matmul        | 0.6688 [0.6634 – 0.6746] | 0.6906 [0.6842 – 0.6941] | 0.7230 [0.7169 – 0.7342] |
| einop_contract_split   | 4.6868 [4.6594 – 4.7318] | 4.7129 [4.6788 – 4.7667] | 4.7783 [4.7330 – 4.8267] |

## Dynamic (ms — median [p25 – p75], 420 call observations / library / case)

| Case | einf M | einops M | einx M |
|---|---|---|---|
| rearrange_transpose_dynamic    | 0.0110 [0.0097 – 0.0125] | 0.0133 [0.0115 – 0.0166] | 0.0400 [0.0325 – 0.0534] |
| rearrange_flatten_hw_dynamic   | 0.0163 [0.0137 – 0.0204] | 0.0180 [0.0147 – 0.0242] | 0.0562 [0.0463 – 0.0742] |
| repeat_expand_axis_dynamic     | 0.0060 [0.0059 – 0.0065] | 0.0072 [0.0069 – 0.0079] | 0.0210 [0.0202 – 0.0238] |
| reduce_sum_axes_dynamic        | 0.1178 [0.0966 – 0.1453] | 0.1289 [0.1079 – 0.1563] | 0.1774 [0.1439 – 0.2384] |
| contract_matmul_dynamic        | 0.2996 [0.1616 – 0.4244] | 0.3051 [0.1813 – 0.4392] | 0.3984 [0.2327 – 0.7447] |
| einop_contract_split_dynamic   | 0.8580 [0.6217 – 1.1867] | 0.9417 [0.6520 – 1.2222] | 1.0874 [0.7335 – 1.5821] |

| Case | einf L | einops L | einx L |
|---|---|---|---|
| rearrange_transpose_dynamic    | 0.0152 [0.0125 – 0.0185] | 0.0186 [0.0155 – 0.0243] | 0.0544 [0.0461 – 0.0720] |
| rearrange_flatten_hw_dynamic   | 0.0371 [0.0288 – 0.0455] | 0.0395 [0.0305 – 0.0491] | 0.0961 [0.0835 – 0.1196] |
| repeat_expand_axis_dynamic     | 0.0058 [0.0057 – 0.0063] | 0.0068 [0.0067 – 0.0076] | 0.0202 [0.0199 – 0.0227] |
| reduce_sum_axes_dynamic        | 0.3294 [0.2430 – 0.4170] | 0.3448 [0.2575 – 0.4437] | 0.4248 [0.3261 – 0.6985] |
| contract_matmul_dynamic        | 0.7401 [0.4795 – 1.0669] | 0.7776 [0.4899 – 1.0866] | 0.9231 [0.5876 – 1.7038] |
| einop_contract_split_dynamic   | 5.4291 [4.3414 – 7.3027] | 5.5144 [4.3879 – 7.3904] | 6.2577 [4.7312 – 10.8784] |

## How to read these numbers

### What pairing does and doesn't control

Both suites use paired scheduling: every library sees the same logical
input at the same point in the round, with independently materialized
tensors and rotating library order. That spreads order and cache-state
effects but does not prove their absence, and it does not remove random
system jitter.

- **Static** measurements vary little because each observation is the arithmetic
  mean of 60 timed calls. The marginal IQRs describe spread across 30 such
  observations; overlap or separation is not significance.
- **Dynamic** marginal IQRs mix workload-shape and timing variation across
  420 calls. Those calls represent 84 round/batch workload units, not 420
  independent samples. The archived artifact lacks the raw identities needed
  to reconstruct the current paired estimator.

### Gap patterns

| Family | Static M | Static L | Dyn M | Dyn L |
|---|---|---|---|---|
| rearrange / repeat | 6 – 17 % lower einf median | 6 – 15 % lower | 9 – 17 % lower | 6 – 18 % lower |
| reduce             | ~6 % lower einf median | close medians | ~9 % lower | ~5 % lower |
| contract_matmul    | ~11 % lower einf median | ~3 % lower | close medians | ~5 % lower |
| einop_contract_split | ~2 % lower einf median | close medians | ~9 % lower | close medians |

Versus einx, the einf median is about 25 – 40 % as large on
rearrange/repeat cases. The einf median is also lower on reduce and
contraction, with a smaller gap on the fused einop case at large shapes.

### What the snapshot shows

Observed in the archived marginal summaries:

- Dispatch-tier einf medians are roughly 5 – 20 % below einops
  on torch in these runs.
- einx has the highest dispatch-tier medians in these runs. The tables do
  not isolate a causal explanation.
- On heavy compute (`reduce L`, `contract L`, `einop_contract_split`),
  the median gap to einops is generally smaller. That pattern is consistent
  with backend compute taking a larger share of wall-clock time.

Not established:

- Any "N× faster" headline across the whole suite.
- Any significance claim from IQR overlap or separation.
- Paired effects or uncertainty intervals: the required raw identities
  were not retained in these historical Markdown reports.

## Snapshot summary

In this 2026-04-17 torch CPU snapshot, einf has lower marginal medians
than einops on dispatch-heavy operations and similar medians on the
largest compute-bound contractions. These are descriptive historical
values, not current performance guarantees or inferential conclusions.
