#!/usr/bin/env python3
"""Compare einf/einops/einx performance on common tensor operations."""

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

from benchmarks.harness import (
    BackendName,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    BenchSizes,
    CaseCalls,
    FixedCaseResult,
    FixedCaseSpec,
    FixedTaskConfig,
    MarkdownPrinter,
    Profiler,
    TensorGenerator,
    TestResult,
    UnavailableRun,
    fixed_sizes_for_scale,
)
from benchmarks.harness.receipt import (
    execution_target_payload,
    paired_evidence_payload,
    synchronized_measurement_contract_payload,
    timing_summary_payload,
)
from benchmarks.shared import (
    as_single_array,
    available_libraries,
    einf_source_metadata,
    version_or_missing,
)
from benchmarks.shared.artifacts import publish_receipt
from einf import ax, axes, contract, einop, rearrange, reduce, repeat

try:
    import einops
except ImportError:
    einops = None

try:
    import einx
except ImportError:
    einx = None


def _build_case_specs(*, sizes, seed: int, backend: BackendSpec) -> list[FixedCaseSpec]:
    generator = TensorGenerator.from_seed(backend=backend, seed=seed)
    x_bnd = generator.randn_backend((sizes.b, sizes.n, sizes.d))
    x_bhwd = generator.randn_backend((sizes.b, sizes.h, sizes.w, sizes.d))
    x_bflatd = x_bhwd.reshape((sizes.b, sizes.h * sizes.w, sizes.d))
    x_bd = generator.randn_backend((sizes.b, sizes.d))
    w_dj = generator.randn_backend((sizes.d, sizes.j))
    x_split_contract = generator.randn_backend(
        (sizes.b, (sizes.h + sizes.w) * sizes.r, sizes.n)
    )
    w_nd = generator.randn_backend((sizes.n, sizes.d))

    b, n, d, h, w, r, j = axes("b", "n", "d", "h", "w", "r", "j")

    def ref_rearrange_transpose(inputs):
        (x,) = inputs
        return np.transpose(x, (0, 2, 1))

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

    def ref_rearrange_split(inputs):
        (x,) = inputs
        b_dim, _, d_dim = x.shape
        return x.reshape((b_dim, sizes.h, sizes.w, d_dim))

    def make_einf_rearrange_split():
        op = rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(
            h=sizes.h, w=sizes.w
        )
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_rearrange_split():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.rearrange(
            inputs[0],
            "b (h w) d -> b h w d",
            h=sizes.h,
            w=sizes.w,
        )

    def make_einx_rearrange_split():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.rearrange(
                "b (h w) d -> b h w d",
                inputs[0],
                h=sizes.h,
                w=sizes.w,
            )
        )

    def ref_repeat(inputs):
        (x,) = inputs
        b_dim, d_dim = x.shape
        return np.broadcast_to(x[..., None], (b_dim, d_dim, sizes.r))

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

    def ref_reduce_sum(inputs):
        (x,) = inputs
        return np.sum(x, axis=(1, 2))

    def make_einf_reduce_sum():
        op = reduce(ax[b, h, w, d], ax[b, d])
        return lambda inputs: as_single_array(op(inputs[0]))

    def make_einops_reduce_sum():
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops
        return lambda inputs: einops_module.reduce(inputs[0], "b h w d -> b d", "sum")

    def make_einx_reduce_sum():
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx
        return lambda inputs: as_single_array(
            einx_module.sum("b h w d -> b d", inputs[0])
        )

    def ref_contract(inputs):
        lhs, rhs = inputs
        return np.einsum("bnd,dj->bnj", lhs, rhs)

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
        FixedCaseSpec(
            case=BenchmarkCase(
                name="rearrange_transpose",
                description=(
                    "Axis permutation: (b, n, d) -> (b, d, n). "
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
            inputs=(x_bnd,),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="rearrange_flatten_hw",
                description=(
                    "Concat spatial axes: (b, h, w, d) -> (b, h*w, d). "
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
            inputs=(x_bhwd,),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="rearrange_split_hw",
                description=(
                    "Split one axis: (b, h*w, d) -> (b, h, w, d). "
                    "einf: rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(...); "
                    "einops/einx: 'b (h w) d -> b h w d' with h,w."
                ),
                calls=CaseCalls(
                    einf=(
                        "rearrange(ax[b, (h * w), d], ax[b, h, w, d])."
                        "with_sizes(h=h, w=w)(x)"
                    ),
                    einops='einops.rearrange(x, "b (h w) d -> b h w d", h=h, w=w)',
                    einx='einx.rearrange("b (h w) d -> b h w d", x, h=h, w=w)',
                ),
                reference=ref_rearrange_split,
                make_einf_runner=make_einf_rearrange_split,
                make_einops_runner=make_einops_rearrange_split,
                make_einx_runner=make_einx_rearrange_split,
            ),
            inputs=(x_bflatd,),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="repeat_expand_axis",
                description="Broadcast repeat: (b, d) -> (b, d, r).",
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
            inputs=(x_bd,),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="reduce_sum_axes",
                description="Reduce sum over spatial axes: (b, h, w, d) -> (b, d).",
                calls=CaseCalls(
                    einf="reduce(ax[b, h, w, d], ax[b, d])(x)",
                    einops='einops.reduce(x, "b h w d -> b d", "sum")',
                    einx='einx.sum("b h w d -> b d", x)',
                ),
                reference=ref_reduce_sum,
                make_einf_runner=make_einf_reduce_sum,
                make_einops_runner=make_einops_reduce_sum,
                make_einx_runner=make_einx_reduce_sum,
            ),
            inputs=(x_bhwd,),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="contract_matmul",
                description="Tensor contraction: (b, n, d) x (d, j) -> (b, n, j).",
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
            inputs=(x_bnd, w_dj),
        ),
        FixedCaseSpec(
            case=BenchmarkCase(
                name="einop_contract_split",
                description=(
                    "Two-stage contract+split path. "
                    "einf: einop((ax[b, ((h + w) * r), n], ax[n, d]), "
                    "(ax[b, (h * r), d], ax[b, (w * r), d])).with_sizes(...); "
                    "einops/einx: einsum then deterministic axis split."
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
            inputs=(x_split_contract, w_nd),
        ),
    ]


def _receipt_payload(
    *,
    config: FixedTaskConfig,
    sizes: BenchSizes,
    case_results: list[FixedCaseResult],
    backend: BackendSpec,
) -> dict[str, object]:
    return {
        "schema_version": 5,
        "benchmark": "einf-vs-einops-einx-fixed",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": version_or_missing("numpy"),
            "torch": version_or_missing("torch"),
            "einops": version_or_missing("einops"),
            "einx": version_or_missing("einx"),
            "einf": einf_source_metadata(),
        },
        "execution_target": execution_target_payload(backend),
        "measurement_contract": synchronized_measurement_contract_payload(),
        "configuration": {
            "backend": backend.name,
            "scale": config.scale,
            "seed": config.seed,
            "rounds": config.rounds,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "iterations": config.iterations,
            "base_sizes": dict(sizes.items()),
        },
        "cases": [
            {
                "case": {
                    "name": case_result.case.name,
                    "description": case_result.case.description,
                    "calls": {
                        "einf": case_result.case.calls.einf,
                        "einops": case_result.case.calls.einops,
                        "einx": case_result.case.calls.einx,
                    },
                },
                "runs": {
                    library: (
                        {
                            "status": "unavailable",
                            "reason": run.reason,
                        }
                        if isinstance(run, UnavailableRun)
                        else {
                            "status": "available",
                            "summary": timing_summary_payload(run.summary),
                            "round_summaries": [
                                timing_summary_payload(round_summary)
                                for round_summary in run.round_summaries
                            ],
                        }
                    )
                    for library, run in case_result.runs.items()
                },
                "round_orders": [
                    list(round_order) for round_order in case_result.round_orders
                ],
                "measurements": [
                    {
                        "phase": "steady",
                        **paired_evidence_payload(case_result.evidence),
                    },
                ],
            }
            for case_result in case_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark einf vs einops vs einx on common tensor operations.",
    )
    parser.add_argument(
        "--scale",
        choices=("small", "medium", "large"),
        default="small",
        help="Input shape profile.",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="Array backend used for data generation and execution.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Backend-native execution device, such as cpu, mps, or cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional canonical JSON receipt path; Markdown is written to stdout.",
    )
    args = parser.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.repeats < 2:
        raise ValueError("--repeats must be >= 2 for paired uncertainty")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    backend_name: BackendName = args.backend
    backend = BackendSpec(name=backend_name, requested_device=args.device)
    sizes = fixed_sizes_for_scale(args.scale)
    config = FixedTaskConfig(
        scale=args.scale,
        seed=args.seed,
        rounds=args.rounds,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
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

    case_specs = _build_case_specs(
        sizes=sizes,
        seed=args.seed,
        backend=backend,
    )
    case_results = [
        runner.run_fixed_case(
            case_spec=case_spec,
            config=config,
            order_seed=args.seed + (case_index * 1009),
        )
        for case_index, case_spec in enumerate(case_specs)
    ]

    result = TestResult(
        title="# einf vs einops vs einx Benchmark",
        configuration=[
            f"Python: `{platform.python_version()}`",
            f"backend: `{backend.name}`",
            f"requested device: `{backend.requested_device}`",
            f"resolved device: `{backend.resolved_device}`",
            f"NumPy: `{version_or_missing('numpy')}`",
            f"torch: `{version_or_missing('torch')}`",
            f"einops: `{version_or_missing('einops')}`",
            f"einx: `{version_or_missing('einx')}`",
            "einf: workspace source (`src/einf`)",
            (
                f"sizes: `b={sizes.b}, n={sizes.n}, d={sizes.d}, h={sizes.h}, "
                f"w={sizes.w}, r={sizes.r}, j={sizes.j}`"
            ),
            f"seed: `{args.seed}`",
            f"rounds: `{args.rounds}`",
            f"warmup calls: `{args.warmup}`",
            f"repeats per round: `{args.repeats}`",
            f"iterations per repeat: `{args.iterations}`",
            (
                "call observations per available library: "
                f"`{args.rounds * args.repeats * args.iterations}`"
            ),
            "table units: `ms`",
            "phase: synchronized steady completion latency",
        ],
        methodology=[
            "The timer starts after target synchronization and stops when the call's submitted work has completed on that target.",
            "For each case, balanced round orders rotate libraries through first/middle/last positions deterministically.",
            "Runner construction, parity validation, warmup, input preparation, and device transfer stay outside the timed interval.",
            "At each paired coordinate, every library receives the same prepared tuple and tensor objects.",
            "Every timed call remains an individual observation.",
            "Within each round, per-call library order rotates from the reported base order to spread position bias.",
            "Repeat blocks are paired across libraries before ratio estimation.",
            "95% intervals use deterministic paired-unit bootstrap resampling stratified by round.",
            "Per-library summaries aggregate all samples across order rounds.",
        ],
        case_results=case_results,
        notes=[
            "Fixed comparisons reuse one prepared input batch across libraries and measured coordinates.",
            "Comparisons are only meaningful when all libraries are available in one environment.",
        ],
    )

    report = MarkdownPrinter().render_fixed(result)
    if args.receipt is not None:
        publish_receipt(
            args.receipt,
            _receipt_payload(
                config=config,
                sizes=sizes,
                case_results=case_results,
                backend=backend,
            ),
        )
    print(report)
    if args.receipt is not None:
        print(f"Wrote receipt: {args.receipt}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
