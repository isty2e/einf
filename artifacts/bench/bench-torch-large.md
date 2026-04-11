# einf vs einops vs einx Benchmark

## Configuration

- Python: `3.11.10`
- backend: `torch`
- NumPy: `2.4.4`
- torch: `2.11.0`
- einops: `0.8.2`
- einx: `0.4.3`
- einf: workspace source (`src/einf`)
- sizes: `b=24, n=384, d=128, h=48, w=32, r=24, j=192`
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
| einf | 18 | 0.1809 | 0.2239 | 0.2726 | 0.0917 | 1.9941 | 30 | 0.0045 | 0.0045 | 0.0047 | 0.0002 | 0.0053 |
| einops | 18 | 0.0768 | 0.1084 | 0.1894 | 0.1126 | 0.3678 | 30 | 0.0053 | 0.0053 | 0.0056 | 0.0003 | 0.0059 |
| einx | 18 | 0.1127 | 0.1380 | 0.2908 | 0.1782 | 15.5889 | 30 | 0.0178 | 0.0180 | 0.0191 | 0.0013 | 0.0208 |

Round median summaries (ms):

- round 1: `einf=0.0045, einops=0.0054, einx=0.0188`
- round 2: `einf=0.0045, einops=0.0053, einx=0.0180`
- round 3: `einf=0.0045, einops=0.0053, einx=0.0177`
- round 4: `einf=0.0045, einops=0.0053, einx=0.0178`
- round 5: `einf=0.0045, einops=0.0053, einx=0.0179`
- round 6: `einf=0.0047, einops=0.0056, einx=0.0188`

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
| einf | 18 | 0.4373 | 0.4685 | 0.5237 | 0.0864 | 0.8474 | 30 | 0.0049 | 0.0049 | 0.0050 | 0.0001 | 0.0052 |
| einops | 18 | 0.3164 | 0.3280 | 0.3705 | 0.0541 | 0.4366 | 30 | 0.0052 | 0.0052 | 0.0053 | 0.0001 | 0.0054 |
| einx | 18 | 0.3299 | 0.3844 | 0.4135 | 0.0836 | 1.8724 | 30 | 0.0175 | 0.0178 | 0.0181 | 0.0006 | 0.0188 |

Round median summaries (ms):

- round 1: `einf=0.0050, einops=0.0053, einx=0.0182`
- round 2: `einf=0.0048, einops=0.0052, einx=0.0178`
- round 3: `einf=0.0049, einops=0.0052, einx=0.0177`
- round 4: `einf=0.0049, einops=0.0052, einx=0.0176`
- round 5: `einf=0.0050, einops=0.0053, einx=0.0180`
- round 6: `einf=0.0049, einops=0.0052, einx=0.0175`

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
| einf | 18 | 0.5286 | 0.5826 | 0.6458 | 0.1172 | 1.1404 | 30 | 0.0057 | 0.0058 | 0.0059 | 0.0002 | 0.0061 |
| einops | 18 | 0.2688 | 0.2946 | 0.3572 | 0.0884 | 0.3876 | 30 | 0.0063 | 0.0064 | 0.0065 | 0.0002 | 0.0067 |
| einx | 18 | 0.3256 | 0.3525 | 0.3944 | 0.0687 | 1.6840 | 30 | 0.0194 | 0.0198 | 0.0206 | 0.0012 | 0.0212 |

Round median summaries (ms):

- round 1: `einf=0.0057, einops=0.0063, einx=0.0194`
- round 2: `einf=0.0058, einops=0.0065, einx=0.0200`
- round 3: `einf=0.0058, einops=0.0065, einx=0.0198`
- round 4: `einf=0.0057, einops=0.0064, einx=0.0196`
- round 5: `einf=0.0059, einops=0.0065, einx=0.0207`
- round 6: `einf=0.0057, einops=0.0062, einx=0.0193`

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
| einf | 18 | 0.0782 | 0.0796 | 0.1163 | 0.0380 | 0.1679 | 30 | 0.0056 | 0.0057 | 0.0058 | 0.0002 | 0.0060 |
| einops | 18 | 0.0084 | 0.0089 | 0.0106 | 0.0022 | 0.0366 | 30 | 0.0064 | 0.0065 | 0.0066 | 0.0002 | 0.0070 |
| einx | 18 | 0.0263 | 0.0286 | 0.0308 | 0.0045 | 1.1319 | 30 | 0.0188 | 0.0190 | 0.0194 | 0.0006 | 0.0206 |

Round median summaries (ms):

- round 1: `einf=0.0056, einops=0.0064, einx=0.0188`
- round 2: `einf=0.0057, einops=0.0065, einx=0.0189`
- round 3: `einf=0.0057, einops=0.0065, einx=0.0192`
- round 4: `einf=0.0056, einops=0.0063, einx=0.0188`
- round 5: `einf=0.0057, einops=0.0066, einx=0.0194`
- round 6: `einf=0.0058, einops=0.0065, einx=0.0190`

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
| einf | 18 | 0.4761 | 0.5103 | 0.6009 | 0.1248 | 0.6779 | 30 | 0.3119 | 0.3176 | 0.3262 | 0.0143 | 0.3752 |
| einops | 18 | 0.3390 | 0.3478 | 0.3635 | 0.0245 | 0.4281 | 30 | 0.3146 | 0.3221 | 0.3363 | 0.0217 | 0.3927 |
| einx | 18 | 0.3839 | 0.3931 | 0.4712 | 0.0873 | 1.3929 | 30 | 0.3497 | 0.3573 | 0.3795 | 0.0298 | 0.4195 |

Round median summaries (ms):

- round 1: `einf=0.3177, einops=0.3158, einx=0.3573`
- round 2: `einf=0.3252, einops=0.3273, einx=0.3644`
- round 3: `einf=0.3348, einops=0.3592, einx=0.3909`
- round 4: `einf=0.3114, einops=0.3101, einx=0.3442`
- round 5: `einf=0.3069, einops=0.3206, einx=0.3525`
- round 6: `einf=0.3208, einops=0.3368, einx=0.3835`

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
| einf | 18 | 0.8252 | 0.9506 | 1.0237 | 0.1985 | 1.2947 | 30 | 0.6634 | 0.6688 | 0.6746 | 0.0112 | 0.6851 |
| einops | 18 | 0.7337 | 0.7571 | 0.8063 | 0.0726 | 0.9039 | 30 | 0.6842 | 0.6906 | 0.6941 | 0.0100 | 0.7017 |
| einx | 18 | 0.7711 | 0.8396 | 0.9085 | 0.1374 | 2.4143 | 30 | 0.7169 | 0.7230 | 0.7342 | 0.0172 | 0.7493 |

Round median summaries (ms):

- round 1: `einf=0.6686, einops=0.6909, einx=0.7175`
- round 2: `einf=0.6645, einops=0.6868, einx=0.7277`
- round 3: `einf=0.6734, einops=0.6913, einx=0.7330`
- round 4: `einf=0.6701, einops=0.6845, einx=0.7191`
- round 5: `einf=0.6685, einops=0.6904, einx=0.7222`
- round 6: `einf=0.6690, einops=0.6923, einx=0.7279`

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
| einf | 18 | 5.5595 | 5.6132 | 5.8851 | 0.3256 | 6.3497 | 30 | 4.6594 | 4.6868 | 4.7318 | 0.0723 | 4.8037 |
| einops | 18 | 4.7264 | 4.7850 | 5.0330 | 0.3065 | 5.8787 | 30 | 4.6788 | 4.7129 | 4.7667 | 0.0880 | 4.9067 |
| einx | 18 | 4.7624 | 4.8228 | 5.0742 | 0.3118 | 8.8479 | 30 | 4.7330 | 4.7783 | 4.8267 | 0.0937 | 4.9286 |

Round median summaries (ms):

- round 1: `einf=4.7847, einops=4.7832, einx=4.8668`
- round 2: `einf=4.6589, einops=4.6724, einx=4.7327`
- round 3: `einf=4.7453, einops=4.7911, einx=4.8307`
- round 4: `einf=4.6497, einops=4.6666, einx=4.7317`
- round 5: `einf=4.6900, einops=4.6829, einx=4.7390`
- round 6: `einf=4.6807, einops=4.7113, einx=4.7936`

## Notes

- Cold and warm compare runs use the same logical workload while keeping physical input storage independent per library.
- Comparisons are only meaningful when all libraries are available in one environment.

