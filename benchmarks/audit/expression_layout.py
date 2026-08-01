#!/usr/bin/env python3
"""Audit layout and tail sensitivity for one gap expression case."""

import argparse
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.compare.expression_parity import ExpressionParityCase, build_gap_cases
from benchmarks.harness import (
    BackendSpec,
    TensorGenerator,
    dynamic_sizes_for_scale,
    torch,
)
from benchmarks.harness.config import BenchSizes
from benchmarks.harness.types import Array, TorchTensor
from benchmarks.shared import version_or_missing
from benchmarks.shared.artifacts import publish_receipt
from einf import ax, axes, einop

try:
    import einops
except ImportError:
    einops = None

try:
    import einx
except ImportError:
    einx = None


ArrayBatch = tuple[Array, ...]
AuditedRunner = Callable[[ArrayBatch], "AuditedRun"]
AuditedRunnerFactory = Callable[[], AuditedRunner]


@dataclass(frozen=True, slots=True)
class LayoutSnapshot:
    """One named tensor layout snapshot."""

    label: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    is_contiguous: bool | None
    storage_offset: int | None


@dataclass(frozen=True, slots=True)
class AuditedRun:
    """One audited strategy execution with intermediate layout snapshots."""

    snapshots: tuple[LayoutSnapshot, ...]


@dataclass(frozen=True, slots=True)
class LayoutSample:
    """One timed batch sample with captured layout snapshots."""

    batch_index: int
    latency_ms: float
    snapshots: tuple[LayoutSnapshot, ...]


@dataclass(frozen=True, slots=True)
class LayoutSummary:
    """Aggregate layout summary for one strategy."""

    sample_count: int
    fastest_samples: tuple[LayoutSample, ...]
    slowest_samples: tuple[LayoutSample, ...]
    stride_signatures: tuple[tuple[str, int], ...]
    contiguous_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class LayoutStrategyReport:
    """One strategy layout audit result."""

    name: str
    call_repr: str
    summary: LayoutSummary


@dataclass(frozen=True, slots=True)
class LayoutAuditReport:
    """Top-level layout audit report for one gap expression case."""

    title: str
    configuration: list[str]
    methodology: list[str]
    strategies: tuple[LayoutStrategyReport, ...]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class AuditedExpressionRunnerSpec:
    """One expression strategy with layout-aware execution."""

    name: str
    call_repr: str
    make_runner: AuditedRunnerFactory


def _torch_pair(inputs: ArrayBatch) -> tuple[TorchTensor, TorchTensor]:
    lhs, rhs = inputs
    if torch is None:
        raise RuntimeError("torch is not available")
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("layout audit requires torch tensor inputs")
    return lhs, rhs


def _shape_tuple(tensor: TorchTensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _stride_tuple(tensor: TorchTensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.stride())


def _snapshot(label: str, tensor: TorchTensor) -> LayoutSnapshot:
    return LayoutSnapshot(
        label=label,
        shape=_shape_tuple(tensor),
        stride=_stride_tuple(tensor),
        is_contiguous=bool(tensor.is_contiguous()),
        storage_offset=int(tensor.storage_offset()),
    )


def _summarize_samples(
    *,
    samples: list[LayoutSample],
    top_k: int,
) -> LayoutSummary:
    sorted_by_latency = sorted(samples, key=lambda sample: sample.latency_ms)
    fastest = tuple(sorted_by_latency[:top_k])
    slowest = tuple(reversed(sorted_by_latency[-top_k:]))

    stride_signatures: Counter[str] = Counter()
    contiguous_counts: Counter[str] = Counter()
    for sample in samples:
        for snapshot in sample.snapshots:
            stride_signatures[
                f"{snapshot.label}:shape={snapshot.shape}:stride={snapshot.stride}"
            ] += 1
            contiguous_counts[
                f"{snapshot.label}:contiguous={snapshot.is_contiguous}"
            ] += 1

    return LayoutSummary(
        sample_count=len(samples),
        fastest_samples=fastest,
        slowest_samples=slowest,
        stride_signatures=tuple(stride_signatures.most_common(8)),
        contiguous_counts=tuple(contiguous_counts.most_common()),
    )


def _build_audited_runner_specs(
    *,
    sizes: BenchSizes,
    case: ExpressionParityCase,
) -> tuple[AuditedExpressionRunnerSpec, ...]:
    b, n, d, h, w, r = axes("b", "n", "d", "h", "w", "r")
    split_index = sizes.h * sizes.r
    second_split = sizes.w * sizes.r
    call_repr_by_name = {spec.name: spec.call_repr for spec in case.runner_specs}

    def make_einf_runner() -> AuditedRunner:
        op = einop(
            (ax[b, ((h + w) * r), n], ax[n, d]),
            (ax[b, (h * r), d], ax[b, (w * r), d]),
        ).with_sizes(h=sizes.h, w=sizes.w, r=sizes.r)

        def run(inputs: ArrayBatch) -> AuditedRun:
            lhs, rhs = _torch_pair(inputs)
            outputs = op(lhs, rhs)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise TypeError("expected two outputs from audited einf case")
            return AuditedRun(
                snapshots=(
                    _snapshot("lhs", lhs),
                    _snapshot("rhs", rhs),
                    _snapshot("out0", outputs[0]),
                    _snapshot("out1", outputs[1]),
                )
            )

        return run

    def make_einops_runner() -> AuditedRunner:
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops

        def run(inputs: ArrayBatch) -> AuditedRun:
            lhs, rhs = _torch_pair(inputs)
            contracted = einops_module.einsum(lhs, rhs, "b t n, n d -> b t d")
            out0 = contracted[:, :split_index, :]
            out1 = contracted[:, split_index:, :]
            return AuditedRun(
                snapshots=(
                    _snapshot("lhs", lhs),
                    _snapshot("rhs", rhs),
                    _snapshot("contracted", contracted),
                    _snapshot("out0", out0),
                    _snapshot("out1", out1),
                )
            )

        return run

    def make_einx_runner() -> AuditedRunner:
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx

        def run(inputs: ArrayBatch) -> AuditedRun:
            lhs, rhs = _torch_pair(inputs)
            contracted = einx_module.dot("b t n, n d -> b t d", lhs, rhs)
            if not isinstance(contracted, torch.Tensor):
                raise TypeError("expected torch tensor from einx dot")
            out0 = contracted[:, :split_index, :]
            out1 = contracted[:, split_index:, :]
            return AuditedRun(
                snapshots=(
                    _snapshot("lhs", lhs),
                    _snapshot("rhs", rhs),
                    _snapshot("contracted", contracted),
                    _snapshot("out0", out0),
                    _snapshot("out1", out1),
                )
            )

        return run

    def make_torch_matmul_split_runner() -> AuditedRunner:
        def run(inputs: ArrayBatch) -> AuditedRun:
            lhs, rhs = _torch_pair(inputs)
            contracted = torch.matmul(lhs, rhs)
            split_outputs = contracted.split((split_index, second_split), dim=1)
            out0, out1 = split_outputs
            return AuditedRun(
                snapshots=(
                    _snapshot("lhs", lhs),
                    _snapshot("rhs", rhs),
                    _snapshot("contracted", contracted),
                    _snapshot("out0", out0),
                    _snapshot("out1", out1),
                )
            )

        return run

    def make_torch_matmul_slice_runner() -> AuditedRunner:
        def run(inputs: ArrayBatch) -> AuditedRun:
            lhs, rhs = _torch_pair(inputs)
            contracted = torch.matmul(lhs, rhs)
            out0 = contracted[:, :split_index, :]
            out1 = contracted[:, split_index:, :]
            return AuditedRun(
                snapshots=(
                    _snapshot("lhs", lhs),
                    _snapshot("rhs", rhs),
                    _snapshot("contracted", contracted),
                    _snapshot("out0", out0),
                    _snapshot("out1", out1),
                )
            )

        return run

    return (
        AuditedExpressionRunnerSpec(
            name="einf",
            call_repr=call_repr_by_name["einf"],
            make_runner=make_einf_runner,
        ),
        AuditedExpressionRunnerSpec(
            name="einops",
            call_repr=call_repr_by_name["einops"],
            make_runner=make_einops_runner,
        ),
        AuditedExpressionRunnerSpec(
            name="einx",
            call_repr=call_repr_by_name["einx"],
            make_runner=make_einx_runner,
        ),
        AuditedExpressionRunnerSpec(
            name="torch_matmul_split",
            call_repr=call_repr_by_name["torch_matmul_split"],
            make_runner=make_torch_matmul_split_runner,
        ),
        AuditedExpressionRunnerSpec(
            name="torch_matmul_slice",
            call_repr=call_repr_by_name["torch_matmul_slice"],
            make_runner=make_torch_matmul_slice_runner,
        ),
    )


def _find_gap_case(*, case_name: str, sizes: BenchSizes) -> ExpressionParityCase:
    case_names = {case.name for case in build_gap_cases(sizes=sizes)}
    for case in build_gap_cases(sizes=sizes):
        if case.name == case_name:
            return case
    raise ValueError(f"unknown gap case: {case_name!r}; known={sorted(case_names)!r}")


def _make_batches(
    *,
    case: ExpressionParityCase,
    seed: int,
    count: int,
) -> list[ArrayBatch]:
    backend = BackendSpec(name="torch")
    return [
        case.batch_factory(
            TensorGenerator.from_seed(backend=backend, seed=seed + index)
        )
        for index in range(count)
    ]


def _audit_case(
    *,
    case_name: str,
    sizes: BenchSizes,
    seed: int,
    batches: int,
    top_k: int,
) -> LayoutAuditReport:
    case = _find_gap_case(case_name=case_name, sizes=sizes)
    runners = _build_audited_runner_specs(sizes=sizes, case=case)
    batch_data = _make_batches(case=case, seed=seed, count=batches)
    strategy_reports: list[LayoutStrategyReport] = []

    for spec in runners:
        runner = spec.make_runner()
        samples: list[LayoutSample] = []
        for batch_index, batch in enumerate(batch_data):
            started = time.perf_counter()
            audited = runner(batch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples.append(
                LayoutSample(
                    batch_index=batch_index,
                    latency_ms=elapsed_ms,
                    snapshots=audited.snapshots,
                )
            )
        strategy_reports.append(
            LayoutStrategyReport(
                name=spec.name,
                call_repr=spec.call_repr,
                summary=_summarize_samples(samples=samples, top_k=top_k),
            )
        )

    return LayoutAuditReport(
        title="# Gap Expression Layout Audit",
        configuration=[
            f"Python: `{platform.python_version()}`",
            f"NumPy: `{version_or_missing('numpy')}`",
            f"torch: `{version_or_missing('torch')}`",
            f"einops: `{version_or_missing('einops')}`",
            f"einx: `{version_or_missing('einx')}`",
            "backend: `torch`",
            f"case: `{case.name}`",
            (
                f"sizes(base): `b={sizes.b}, n={sizes.n}, d={sizes.d}, "
                f"h={sizes.h}, w={sizes.w}, r={sizes.r}, j={sizes.j}`"
            ),
            f"seed: `{seed}`",
            f"batches: `{batches}`",
            f"top-k per tail: `{top_k}`",
        ],
        methodology=[
            "Each strategy runs over identical deterministic torch batches.",
            "This audit is diagnostic, not a competitive benchmark: it captures per-batch latency with layout snapshots for inputs, intermediates, and outputs.",
            "The report surfaces the fastest and slowest samples plus recurring stride/contiguity signatures.",
        ],
        strategies=tuple(strategy_reports),
        notes=[
            "Use this audit to explain tail behavior and suite-vs-isolated ordering differences after expression-parity benchmarking.",
        ],
    )


def _to_json(report: LayoutAuditReport) -> dict[str, object]:
    return asdict(report)


def _render_markdown(report: LayoutAuditReport) -> str:
    lines = [report.title, "", "## Configuration", ""]
    lines.extend(f"- {entry}" for entry in report.configuration)
    lines.extend(["", "## Methodology", ""])
    lines.extend(f"- {entry}" for entry in report.methodology)
    lines.extend(["", "## Results", ""])
    for strategy in report.strategies:
        lines.extend(
            [
                f"### {strategy.name}",
                "",
                f"- Call: `{strategy.call_repr}`",
                f"- Samples: `{strategy.summary.sample_count}`",
                "",
                "Most common stride signatures:",
                "",
            ]
        )
        for signature, count in strategy.summary.stride_signatures:
            lines.append(f"- `{signature}` x `{count}`")
        lines.extend(["", "Contiguity counts:", ""])
        for signature, count in strategy.summary.contiguous_counts:
            lines.append(f"- `{signature}` x `{count}`")
        lines.extend(["", "Fastest samples:", ""])
        for sample in strategy.summary.fastest_samples:
            lines.append(
                f"- batch `{sample.batch_index}` latency `{sample.latency_ms:.4f} ms`"
            )
            for snapshot in sample.snapshots:
                lines.append(
                    f"  - `{snapshot.label}` shape={snapshot.shape} stride={snapshot.stride} "
                    f"contiguous={snapshot.is_contiguous} storage_offset={snapshot.storage_offset}"
                )
        lines.extend(["", "Slowest samples:", ""])
        for sample in strategy.summary.slowest_samples:
            lines.append(
                f"- batch `{sample.batch_index}` latency `{sample.latency_ms:.4f} ms`"
            )
            for snapshot in sample.snapshots:
                lines.append(
                    f"  - `{snapshot.label}` shape={snapshot.shape} stride={snapshot.stride} "
                    f"contiguous={snapshot.is_contiguous} storage_offset={snapshot.storage_offset}"
                )
        lines.append("")
    lines.extend(["## Notes", ""])
    lines.extend(f"- {entry}" for entry in report.notes)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit layout and tail sensitivity for one gap expression case.",
    )
    parser.add_argument(
        "--scale",
        choices=("medium", "large"),
        default="large",
        help="Base dynamic-shape profile.",
    )
    parser.add_argument(
        "--case",
        default="einop_contract_split_dynamic",
        help="Gap case name to audit.",
    )
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional canonical JSON receipt path; Markdown is written to stdout.",
    )
    args = parser.parse_args()

    if args.batches < 1:
        raise ValueError(f"batches must be >= 1, got {args.batches}")
    if args.top_k < 1:
        raise ValueError(f"top-k must be >= 1, got {args.top_k}")

    sizes = dynamic_sizes_for_scale(args.scale)
    _ = _find_gap_case(case_name=args.case, sizes=sizes)
    report = _audit_case(
        case_name=args.case,
        sizes=sizes,
        seed=args.seed,
        batches=args.batches,
        top_k=args.top_k,
    )
    markdown = _render_markdown(report)
    if args.receipt is not None:
        publish_receipt(args.receipt, _to_json(report))
    print(markdown)
    if args.receipt is not None:
        print(f"Wrote receipt: {args.receipt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
