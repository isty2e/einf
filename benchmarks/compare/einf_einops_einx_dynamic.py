#!/usr/bin/env python3
"""Compare einf/einops/einx on dynamic-shape batches."""

import argparse
import platform
from pathlib import Path

import numpy as np

from benchmarks.harness import (
    BackendName,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    CaseCalls,
    DynamicCaseSpec,
    DynamicTaskConfig,
    MarkdownPrinter,
    Profiler,
    TensorGenerator,
    TestResult,
    dynamic_sizes_for_scale,
)
from benchmarks.shared import as_single_array, available_libraries, version_or_missing
from einf import ax, axes, contract, einop, rearrange, reduce, repeat

try:
    import einops
except Exception:
    einops = None

try:
    import einx
except Exception:
    einx = None


def _build_case_specs(*, sizes) -> list[DynamicCaseSpec]:
    b, n, d, h, w, r, j = axes("b", "n", "d", "h", "w", "r", "j")

    def ref_rearrange_transpose(inputs):
        (x,) = inputs
        return np.transpose(x, (0, 2, 1))

    def batch_rearrange_transpose(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        n_dim = generator.draw_dimension(base=sizes.n)
        d_dim = generator.draw_dimension(base=sizes.d)
        return generator.backend_batch((generator.randn_numpy((b_dim, n_dim, d_dim)),))

    def make_einf_rearrange_transpose():
        op = rearrange(ax[b, n, d], ax[b, d, n])
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_rearrange_transpose():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.rearrange(inputs[0], "b n d -> b d n")

    def make_einx_rearrange_transpose():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.rearrange("b n d -> b d n", inputs[0])
        )

    def ref_rearrange_flatten(inputs):
        (x,) = inputs
        b_dim, h_dim, w_dim, d_dim = x.shape
        return x.reshape((b_dim, h_dim * w_dim, d_dim))

    def batch_rearrange_flatten(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        h_dim = generator.draw_dimension(base=sizes.h)
        w_dim = generator.draw_dimension(base=sizes.w)
        d_dim = generator.draw_dimension(base=sizes.d)
        return generator.backend_batch(
            (generator.randn_numpy((b_dim, h_dim, w_dim, d_dim)),)
        )

    def make_einf_rearrange_flatten():
        op = rearrange(ax[b, h, w, d], ax[b, (h * w), d])
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_rearrange_flatten():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.rearrange(inputs[0], "b h w d -> b (h w) d")

    def make_einx_rearrange_flatten():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.rearrange("b h w d -> b (h w) d", inputs[0])
        )

    def ref_repeat(inputs):
        (x,) = inputs
        b_dim, d_dim = x.shape
        return np.broadcast_to(x[..., None], (b_dim, d_dim, sizes.r))

    def batch_repeat(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        d_dim = generator.draw_dimension(base=sizes.d)
        return generator.backend_batch((generator.randn_numpy((b_dim, d_dim)),))

    def make_einf_repeat():
        op = repeat(ax[b, d], ax[b, d, r]).with_sizes(r=sizes.r)
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_repeat():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.repeat(inputs[0], "b d -> b d r", r=sizes.r)

    def make_einx_repeat():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.rearrange("b d -> b d r", inputs[0], r=sizes.r)
        )

    def ref_reduce(inputs):
        (x,) = inputs
        return np.sum(x, axis=(1, 2))

    def batch_reduce(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        h_dim = generator.draw_dimension(base=sizes.h)
        w_dim = generator.draw_dimension(base=sizes.w)
        d_dim = generator.draw_dimension(base=sizes.d)
        return generator.backend_batch(
            (generator.randn_numpy((b_dim, h_dim, w_dim, d_dim)),)
        )

    def make_einf_reduce():
        op = reduce(ax[b, h, w, d], ax[b, d])
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_reduce():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.reduce(inputs[0], "b h w d -> b d", "sum")

    def make_einx_reduce():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.sum("b h w d -> b d", inputs[0])
        )

    def ref_contract(inputs):
        lhs, rhs = inputs
        return np.einsum("bnd,dj->bnj", lhs, rhs)

    def batch_contract(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        n_dim = generator.draw_dimension(base=sizes.n)
        d_dim = generator.draw_dimension(base=sizes.d)
        j_dim = generator.draw_dimension(base=sizes.j)
        lhs = generator.randn_numpy((b_dim, n_dim, d_dim))
        rhs = generator.randn_numpy((d_dim, j_dim))
        return generator.backend_batch((lhs, rhs))

    def make_einf_contract():
        op = contract((ax[b, n, d], ax[d, j]), ax[b, n, j])
        return lambda inputs: as_single_array(op(inputs[0], inputs[1]))

    def make_einops_contract():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.einsum(
            inputs[0], inputs[1], "b n d, d j -> b n j"
        )

    def make_einx_contract():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.dot("b n d, d j -> b n j", inputs[0], inputs[1])
        )

    def ref_einop_contract_split(inputs):
        lhs, rhs = inputs
        split_index = sizes.h * sizes.r
        contracted = np.einsum("btn,nd->btd", lhs, rhs)
        return (contracted[:, :split_index, :], contracted[:, split_index:, :])

    def batch_einop_contract_split(generator: TensorGenerator):
        b_dim = generator.draw_dimension(base=sizes.b)
        n_dim = generator.draw_dimension(base=sizes.n)
        d_dim = generator.draw_dimension(base=sizes.d)
        lhs = generator.randn_numpy((b_dim, (sizes.h + sizes.w) * sizes.r, n_dim))
        rhs = generator.randn_numpy((n_dim, d_dim))
        return generator.backend_batch((lhs, rhs))

    def make_einf_einop_contract_split():
        op = einop(
            (ax[b, ((h + w) * r), n], ax[n, d]),
            (ax[b, (h * r), d], ax[b, (w * r), d]),
        ).with_sizes(h=sizes.h, w=sizes.w, r=sizes.r)
        return lambda inputs: op(inputs[0], inputs[1])

    def make_einops_two_stage_contract_split():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        split_index = sizes.h * sizes.r

        def run(inputs):
            contracted = einops_module.einsum(
                inputs[0], inputs[1], "b t n, n d -> b t d"
            )
            return (contracted[:, :split_index, :], contracted[:, split_index:, :])

        return run

    def make_einx_two_stage_contract_split():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        split_index = sizes.h * sizes.r

        def run(inputs):
            contracted = as_single_array(
                einx_module.dot("b t n, n d -> b t d", inputs[0], inputs[1])
            )
            return (contracted[:, :split_index, :], contracted[:, split_index:, :])

        return run

    return [
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="rearrange_transpose_dynamic",
                description=(
                    "Transpose with (b, n, d) varying every batch. "
                    "einf: rearrange(ax[b, n, d], ax[b, d, n]); "
                    "einops/einx: 'b n d -> b d n'."
                ),
                calls=CaseCalls(
                    einf="rearrange(ax[b, n, d], ax[b, d, n])(x)",
                    einops='einops.rearrange(x, "b n d -> b d n")',
                    einx='einx.rearrange("b n d -> b d n", x)',
                ),
                reference=ref_rearrange_transpose,
                make_einf_runner=make_einf_rearrange_transpose,
                make_einops_runner=make_einops_rearrange_transpose,
                make_einx_runner=make_einx_rearrange_transpose,
            ),
            batch_factory=batch_rearrange_transpose,
        ),
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="rearrange_flatten_hw_dynamic",
                description=(
                    "Flatten (h, w) where b/h/w/d vary every batch. "
                    "einf: rearrange(ax[b, h, w, d], ax[b, (h * w), d]); "
                    "einops/einx: 'b h w d -> b (h w) d'."
                ),
                calls=CaseCalls(
                    einf="rearrange(ax[b, h, w, d], ax[b, (h * w), d])(x)",
                    einops='einops.rearrange(x, "b h w d -> b (h w) d")',
                    einx='einx.rearrange("b h w d -> b (h w) d", x)',
                ),
                reference=ref_rearrange_flatten,
                make_einf_runner=make_einf_rearrange_flatten,
                make_einops_runner=make_einops_rearrange_flatten,
                make_einx_runner=make_einx_rearrange_flatten,
            ),
            batch_factory=batch_rearrange_flatten,
        ),
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="repeat_expand_axis_dynamic",
                description="Repeat with fixed r but b/d varying every batch.",
                calls=CaseCalls(
                    einf="repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)",
                    einops='einops.repeat(x, "b d -> b d r", r=r)',
                    einx='einx.rearrange("b d -> b d r", x, r=r)',
                ),
                reference=ref_repeat,
                make_einf_runner=make_einf_repeat,
                make_einops_runner=make_einops_repeat,
                make_einx_runner=make_einx_repeat,
            ),
            batch_factory=batch_repeat,
        ),
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="reduce_sum_axes_dynamic",
                description="Reduce sum with b/h/w/d varying every batch.",
                calls=CaseCalls(
                    einf="reduce(ax[b, h, w, d], ax[b, d])(x)",
                    einops='einops.reduce(x, "b h w d -> b d", "sum")',
                    einx='einx.sum("b h w d -> b d", x)',
                ),
                reference=ref_reduce,
                make_einf_runner=make_einf_reduce,
                make_einops_runner=make_einops_reduce,
                make_einx_runner=make_einx_reduce,
            ),
            batch_factory=batch_reduce,
        ),
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="contract_matmul_dynamic",
                description="Contract with b/n/d/j varying every batch.",
                calls=CaseCalls(
                    einf="contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
                    einops='einops.einsum(lhs, rhs, "b n d, d j -> b n j")',
                    einx='einx.dot("b n d, d j -> b n j", lhs, rhs)',
                ),
                reference=ref_contract,
                make_einf_runner=make_einf_contract,
                make_einops_runner=make_einops_contract,
                make_einx_runner=make_einx_contract,
            ),
            batch_factory=batch_contract,
        ),
        DynamicCaseSpec(
            case=BenchmarkCase(
                name="einop_contract_split_dynamic",
                description=(
                    "Two-stage contract+split with dynamic b/n/d and fixed h/w/r "
                    "split boundary."
                ),
                calls=CaseCalls(
                    einf=(
                        "einop((ax[b, ((h + w) * r), n], ax[n, d]), "
                        "(ax[b, (h * r), d], ax[b, (w * r), d]))"
                        ".with_sizes(h=h, w=w, r=r)(lhs, rhs)"
                    ),
                    einops=(
                        'tmp = einops.einsum(lhs, rhs, "b t n, n d -> b t d"); '
                        "(tmp[:, :h*r, :], tmp[:, h*r:, :])"
                    ),
                    einx=(
                        'tmp = einx.dot("b t n, n d -> b t d", lhs, rhs); '
                        "(tmp[:, :h*r, :], tmp[:, h*r:, :])"
                    ),
                ),
                reference=ref_einop_contract_split,
                make_einf_runner=make_einf_einop_contract_split,
                make_einops_runner=make_einops_two_stage_contract_split,
                make_einx_runner=make_einx_two_stage_contract_split,
            ),
            batch_factory=batch_einop_contract_split,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic-shape benchmark for einf vs einops vs einx.",
    )
    parser.add_argument(
        "--scale",
        choices=("medium", "large"),
        default="medium",
        help="Base shape profile.",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="Tensor backend used for dynamic batch tensors.",
    )
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of balanced paired rounds per case.",
    )
    parser.add_argument(
        "--round-order-seed",
        type=int,
        default=None,
        help="Optional seed controlling deterministic per-round library order.",
    )
    parser.add_argument(
        "--parity-checks",
        type=int,
        default=8,
        help="Number of first batches used for parity validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    args = parser.parse_args()

    if args.batches < 1:
        raise ValueError(f"batches must be >= 1, got {args.batches}")
    if args.warmup_batches < 0:
        raise ValueError(f"warmup-batches must be >= 0, got {args.warmup_batches}")
    if args.warmup_batches >= args.batches:
        raise ValueError(
            "warmup-batches must be less than batches to leave measured batches"
        )
    if args.repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {args.repeats}")
    if args.rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {args.rounds}")

    round_order_seed = (
        args.seed if args.round_order_seed is None else args.round_order_seed
    )
    backend_name: BackendName = args.backend
    backend = BackendSpec(name=backend_name)
    backend.validate_available()
    sizes = dynamic_sizes_for_scale(args.scale)

    config = DynamicTaskConfig(
        backend=backend_name,
        scale=args.scale,
        seed=args.seed,
        batches=args.batches,
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
        rounds=args.rounds,
        round_order_seed=round_order_seed,
        parity_checks=args.parity_checks,
    )

    available = available_libraries(
        einops_module=einops,
        einx_module=einx,
    )
    profiler = Profiler(backend=backend)
    runner = BenchmarkRunner(
        backend=backend,
        profiler=profiler,
        available=available,
    )

    case_specs = _build_case_specs(sizes=sizes)
    case_results = [
        runner.run_dynamic_case(case_spec=case_spec, config=config, case_index=index)
        for index, case_spec in enumerate(case_specs)
    ]

    measured_batches_per_repeat = args.batches - args.warmup_batches
    total_samples_per_library = measured_batches_per_repeat * args.repeats * args.rounds

    result = TestResult(
        title="# einf vs einops vs einx Dynamic-Shape Benchmark",
        configuration=[
            f"Python: `{platform.python_version()}`",
            f"NumPy: `{version_or_missing('numpy')}`",
            f"torch: `{version_or_missing('torch')}`",
            f"einops: `{version_or_missing('einops')}`",
            f"einx: `{version_or_missing('einx')}`",
            "einf: workspace source (`src/einf`)",
            f"backend: `{args.backend}`",
            (
                f"sizes(base): `b={sizes.b}, n={sizes.n}, d={sizes.d}, "
                f"h={sizes.h}, w={sizes.w}, r={sizes.r}, j={sizes.j}`"
            ),
            "each dimension is sampled per batch in `[0.6x, 1.4x]` of base size",
            f"seed: `{args.seed}`",
            f"round order seed: `{round_order_seed}`",
            f"total batches per case: `{args.batches}`",
            f"warmup batches: `{args.warmup_batches}`",
            f"repeats per round: `{args.repeats}`",
            f"rounds: `{args.rounds}`",
            f"measured batches per repeat: `{measured_batches_per_repeat}`",
            f"expected samples per library/case: `{total_samples_per_library}`",
            "table units: `ms`",
        ],
        methodology=[
            "Each case uses deterministic dynamic inputs generated by seeded `RandomState`.",
            "In each round, every measured batch is executed in paired cross-library order on the same logical batch stream.",
            "Latency is sampled per post-warmup batch call (no repeat-level averaging).",
            "Each library receives independently materialized backend batches to avoid cross-library input locality artifacts.",
            "Within each round, per-batch library order rotates from the reported base order to spread position bias.",
            "Round summaries expose residual order-sensitive variation; aggregate stats pool all per-batch samples across rounds.",
            "Reported stats: `count`, `p25`, `median`, `p75`, `p95`, `mean`, `min`, `max`.",
        ],
        case_results=case_results,
        notes=[
            "Each runner is constructed once per case and reused across batches.",
            "Dynamic parity checks run before timing using the first round batches.",
            "Dynamic compare uses the same logical batches within each round while keeping physical input storage independent per library.",
        ],
    )

    report = MarkdownPrinter().render_dynamic(result)
    print(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"\nWrote report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
