# einf vs einops vs einx Benchmark

## Configuration

- Python: `3.11.10`
- backend: `torch`
- NumPy: `2.4.4`
- torch: `2.11.0`
- einops: `0.8.2`
- einx: `0.4.3`
- einf: workspace source (`src/einf`)
- sizes: `b=16, n=192, d=96, h=32, w=24, r=16, j=128`
- seed: `1234`
- rounds: `6`
- cold repeats per round: `3`
- warmup calls: `4`
- warm repeats per round: `5`
- warm iterations per repeat: `60`
- aggregated cold samples per available library: `18`
- aggregated warm samples per available library: `30`
- table units: `ms`
- `cold`: op construction + first execution
- `warm`: post-warmup steady-state per-call latency

## Methodology

- For each case, balanced round orders rotate libraries through first/middle/last positions deterministically.
- Cold validation uses outputs captured during timed cold samples.
- Warm timing runs paired per-call execution on the same logical input while using independently materialized per-library tensors.
- Within each round, per-call library order rotates from the reported base order to spread position bias.
- Per-library summaries aggregate all samples across order rounds.

## Results

### rearrange_transpose

Axis permutation: (b, n, d) -> (b, d, n). einf: rearrange(ax[b, n, d], ax[b, d, n]); einops/einx: 'b n d -> b d n'.

Execution forms:

- `einf`: `rearrange(ax[b, n, d], ax[b, d, n])(x)`
- `einops`: `einops.rearrange(x, "b n d -> b d n")`
- `einx`: `einx.rearrange("b n d -> b d n", x)`

Round base order (paired execution rotates within each round):

- round 1: `einf -> einops -> einx`
- round 2: `einops -> einx -> einf`
- round 3: `einx -> einf -> einops`
- round 4: `einops -> einx -> einf`
- round 5: `einx -> einf -> einops`
- round 6: `einf -> einops -> einx`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.1309 | 0.1509 | 0.1977 | 0.0668 | 1.3405 | 30 | 0.0045 | 0.0045 | 0.0046 | 0.0001 | 0.0052 |
| einops | 18 | 0.0181 | 0.0229 | 0.0354 | 0.0173 | 0.1537 | 30 | 0.0053 | 0.0054 | 0.0055 | 0.0002 | 0.0067 |
| einx | 18 | 0.0475 | 0.0688 | 0.0992 | 0.0517 | 12.8323 | 30 | 0.0180 | 0.0183 | 0.0192 | 0.0012 | 0.0238 |

Round median summaries (ms):

- round 1: `einf=0.0045, einops=0.0053, einx=0.0193`
- round 2: `einf=0.0047, einops=0.0055, einx=0.0187`
- round 3: `einf=0.0045, einops=0.0053, einx=0.0183`
- round 4: `einf=0.0046, einops=0.0054, einx=0.0181`
- round 5: `einf=0.0045, einops=0.0054, einx=0.0181`
- round 6: `einf=0.0046, einops=0.0054, einx=0.0183`

### rearrange_flatten_hw

Concat spatial axes: (b, h, w, d) -> (b, h*w, d). einf: rearrange(ax[b, h, w, d], ax[b, (h * w), d]); einops/einx: 'b h w d -> b (h w) d'.

Execution forms:

- `einf`: `rearrange(ax[b, h, w, d], ax[b, (h * w), d])(x)`
- `einops`: `einops.rearrange(x, "b h w d -> b (h w) d")`
- `einx`: `einx.rearrange("b h w d -> b (h w) d", x)`

Round base order (paired execution rotates within each round):

- round 1: `einops -> einf -> einx`
- round 2: `einf -> einx -> einops`
- round 3: `einx -> einops -> einf`
- round 4: `einx -> einf -> einops`
- round 5: `einf -> einops -> einx`
- round 6: `einops -> einx -> einf`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.2449 | 0.2564 | 0.3402 | 0.0952 | 0.5234 | 30 | 0.0049 | 0.0049 | 0.0050 | 0.0001 | 0.0060 |
| einops | 18 | 0.0927 | 0.1127 | 0.1401 | 0.0474 | 0.2297 | 30 | 0.0051 | 0.0052 | 0.0053 | 0.0002 | 0.0060 |
| einx | 18 | 0.1165 | 0.1387 | 0.1676 | 0.0511 | 1.7413 | 30 | 0.0173 | 0.0176 | 0.0179 | 0.0006 | 0.0215 |

Round median summaries (ms):

- round 1: `einf=0.0050, einops=0.0053, einx=0.0181`
- round 2: `einf=0.0049, einops=0.0051, einx=0.0173`
- round 3: `einf=0.0049, einops=0.0051, einx=0.0174`
- round 4: `einf=0.0049, einops=0.0051, einx=0.0174`
- round 5: `einf=0.0049, einops=0.0051, einx=0.0175`
- round 6: `einf=0.0050, einops=0.0052, einx=0.0177`

### rearrange_split_hw

Split one axis: (b, h*w, d) -> (b, h, w, d). einf: rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(...); einops/einx: 'b (h w) d -> b h w d' with h,w.

Execution forms:

- `einf`: `rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=h, w=w)(x)`
- `einops`: `einops.rearrange(x, "b (h w) d -> b h w d", h=h, w=w)`
- `einx`: `einx.rearrange("b (h w) d -> b h w d", x, h=h, w=w)`

Round base order (paired execution rotates within each round):

- round 1: `einx -> einops -> einf`
- round 2: `einops -> einf -> einx`
- round 3: `einf -> einx -> einops`
- round 4: `einx -> einf -> einops`
- round 5: `einf -> einops -> einx`
- round 6: `einops -> einx -> einf`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.2892 | 0.3002 | 0.3482 | 0.0591 | 0.4832 | 30 | 0.0055 | 0.0055 | 0.0056 | 0.0001 | 0.0058 |
| einops | 18 | 0.0920 | 0.0951 | 0.1004 | 0.0085 | 0.2452 | 30 | 0.0060 | 0.0060 | 0.0061 | 0.0002 | 0.0066 |
| einx | 18 | 0.0995 | 0.1232 | 0.1435 | 0.0440 | 1.5468 | 30 | 0.0188 | 0.0189 | 0.0191 | 0.0003 | 0.0209 |

Round median summaries (ms):

- round 1: `einf=0.0054, einops=0.0060, einx=0.0187`
- round 2: `einf=0.0056, einops=0.0062, einx=0.0192`
- round 3: `einf=0.0055, einops=0.0060, einx=0.0188`
- round 4: `einf=0.0056, einops=0.0061, einx=0.0191`
- round 5: `einf=0.0055, einops=0.0060, einx=0.0188`
- round 6: `einf=0.0055, einops=0.0060, einx=0.0189`

### repeat_expand_axis

Broadcast repeat: (b, d) -> (b, d, r).

Execution forms:

- `einf`: `repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)`
- `einops`: `einops.repeat(x, "b d -> b d r", r=r)`
- `einx`: `einx.rearrange("b d -> b d r", x, r=r)`

Round base order (paired execution rotates within each round):

- round 1: `einf -> einops -> einx`
- round 2: `einops -> einx -> einf`
- round 3: `einx -> einf -> einops`
- round 4: `einops -> einx -> einf`
- round 5: `einx -> einf -> einops`
- round 6: `einf -> einops -> einx`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.0761 | 0.0841 | 0.1307 | 0.0547 | 0.4513 | 30 | 0.0054 | 0.0056 | 0.0058 | 0.0004 | 0.0063 |
| einops | 18 | 0.0084 | 0.0098 | 0.0123 | 0.0039 | 0.0691 | 30 | 0.0062 | 0.0063 | 0.0066 | 0.0004 | 0.0074 |
| einx | 18 | 0.0290 | 0.0302 | 0.0346 | 0.0056 | 1.1588 | 30 | 0.0189 | 0.0194 | 0.0206 | 0.0017 | 0.0211 |

Round median summaries (ms):

- round 1: `einf=0.0058, einops=0.0064, einx=0.0197`
- round 2: `einf=0.0058, einops=0.0067, einx=0.0206`
- round 3: `einf=0.0054, einops=0.0061, einx=0.0185`
- round 4: `einf=0.0055, einops=0.0063, einx=0.0193`
- round 5: `einf=0.0057, einops=0.0064, einx=0.0198`
- round 6: `einf=0.0055, einops=0.0063, einx=0.0191`

### reduce_sum_axes

Reduce sum over spatial axes: (b, h, w, d) -> (b, d).

Execution forms:

- `einf`: `reduce(ax[b, h, w, d], ax[b, d])(x)`
- `einops`: `einops.reduce(x, "b h w d -> b d", "sum")`
- `einx`: `einx.sum("b h w d -> b d", x)`

Round base order (paired execution rotates within each round):

- round 1: `einx -> einops -> einf`
- round 2: `einops -> einf -> einx`
- round 3: `einf -> einx -> einops`
- round 4: `einf -> einops -> einx`
- round 5: `einops -> einx -> einf`
- round 6: `einx -> einf -> einops`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.2315 | 0.2418 | 0.2512 | 0.0197 | 0.2809 | 30 | 0.0884 | 0.0895 | 0.0932 | 0.0049 | 0.0979 |
| einops | 18 | 0.1157 | 0.1227 | 0.1302 | 0.0145 | 0.1779 | 30 | 0.0928 | 0.0952 | 0.0981 | 0.0053 | 0.1046 |
| einx | 18 | 0.1378 | 0.1465 | 0.1539 | 0.0161 | 1.5505 | 30 | 0.1187 | 0.1217 | 0.1257 | 0.0070 | 0.1348 |

Round median summaries (ms):

- round 1: `einf=0.0901, einops=0.0952, einx=0.1226`
- round 2: `einf=0.0927, einops=0.0960, einx=0.1248`
- round 3: `einf=0.0936, einops=0.0986, einx=0.1292`
- round 4: `einf=0.0886, einops=0.0937, einx=0.1198`
- round 5: `einf=0.0887, einops=0.0929, einx=0.1189`
- round 6: `einf=0.0892, einops=0.0951, einx=0.1237`

### contract_matmul

Tensor contraction: (b, n, d) x (d, j) -> (b, n, j).

Execution forms:

- `einf`: `contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)`
- `einops`: `einops.einsum(lhs, rhs, "b n d, d j -> b n j")`
- `einx`: `einx.dot("b n d, d j -> b n j", lhs, rhs)`

Round base order (paired execution rotates within each round):

- round 1: `einf -> einops -> einx`
- round 2: `einops -> einx -> einf`
- round 3: `einx -> einf -> einops`
- round 4: `einops -> einx -> einf`
- round 5: `einx -> einf -> einops`
- round 6: `einf -> einops -> einx`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 0.2796 | 0.3197 | 0.3478 | 0.0681 | 0.7966 | 30 | 0.0811 | 0.0819 | 0.0842 | 0.0030 | 0.0905 |
| einops | 18 | 0.1292 | 0.1895 | 0.2144 | 0.0853 | 0.4367 | 30 | 0.0909 | 0.0918 | 0.0940 | 0.0031 | 0.0986 |
| einx | 18 | 0.2299 | 0.2651 | 0.2764 | 0.0464 | 1.6297 | 30 | 0.1076 | 0.1093 | 0.1139 | 0.0064 | 0.1184 |

Round median summaries (ms):

- round 1: `einf=0.0804, einops=0.0898, einx=0.1068`
- round 2: `einf=0.0814, einops=0.0910, einx=0.1093`
- round 3: `einf=0.0812, einops=0.0917, einx=0.1076`
- round 4: `einf=0.0844, einops=0.0955, einx=0.1142`
- round 5: `einf=0.0876, einops=0.0928, einx=0.1140`
- round 6: `einf=0.0822, einops=0.0929, einx=0.1096`

### einop_contract_split

Two-stage contract+split path. einf: einop((ax[b, ((h + w) * r), n], ax[n, d]), (ax[b, (h * r), d], ax[b, (w * r), d])).with_sizes(...); einops/einx: einsum then deterministic axis split.

Execution forms:

- `einf`: `einop((ax[b, ((h + w) * r), n], ax[n, d]), (ax[b, (h * r), d], ax[b, (w * r), d])).with_sizes(h=h, w=w, r=r)(lhs, rhs)`
- `einops`: `tmp = einops.einsum(lhs, rhs, "b t n, n d -> b t d"); (tmp[:, :h*r, :], tmp[:, h*r:, :])`
- `einx`: `tmp = einx.dot("b t n, n d -> b t d", lhs, rhs); (tmp[:, :h*r, :], tmp[:, h*r:, :])`

Round base order (paired execution rotates within each round):

- round 1: `einops -> einx -> einf`
- round 2: `einx -> einf -> einops`
- round 3: `einf -> einops -> einx`
- round 4: `einops -> einf -> einx`
- round 5: `einf -> einx -> einops`
- round 6: `einx -> einops -> einf`

| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einf | 18 | 1.7276 | 1.7778 | 1.8683 | 0.1406 | 2.8751 | 30 | 0.7991 | 0.8109 | 0.8362 | 0.0370 | 0.9045 |
| einops | 18 | 0.8191 | 0.8286 | 0.8732 | 0.0541 | 1.1399 | 30 | 0.8119 | 0.8302 | 0.8449 | 0.0330 | 0.8918 |
| einx | 18 | 0.8786 | 0.8928 | 0.9424 | 0.0638 | 2.4352 | 30 | 0.8560 | 0.8697 | 0.8911 | 0.0351 | 0.9403 |

Round median summaries (ms):

- round 1: `einf=0.7897, einops=0.8070, einx=0.8554`
- round 2: `einf=0.8461, einops=0.8310, einx=0.8781`
- round 3: `einf=0.8361, einops=0.8380, einx=0.8812`
- round 4: `einf=0.8099, einops=0.8445, einx=0.8669`
- round 5: `einf=0.8040, einops=0.8372, einx=0.8538`
- round 6: `einf=0.8074, einops=0.8294, einx=0.8719`

## Notes

- Cold and warm compare runs use the same logical workload while keeping physical input storage independent per library.
- Comparisons are only meaningful when all libraries are available in one environment.

