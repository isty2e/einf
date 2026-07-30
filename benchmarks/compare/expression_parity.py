#!/usr/bin/env python3
"""Benchmark gap-case expression strategies on dynamic torch batches."""

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks.harness import (
    BackendSpec,
    DynamicTaskConfig,
    Profiler,
    TensorGenerator,
    TimingSummary,
    dynamic_sizes_for_scale,
    torch,
)
from benchmarks.harness.config import BenchSizes
from benchmarks.harness.types import Array, NumpyArray, Output, Runner, TorchTensor
from benchmarks.shared import as_single_array, version_or_missing
from einf import ax, axes, einop

try:
    import einops
except ImportError:
    einops = None

try:
    import einx
except ImportError:
    einx = None


ExpressionRole = Literal["equivalent", "baseline"]
Reference = Callable[[tuple[NumpyArray, ...]], Output]
BatchFactory = Callable[[TensorGenerator], tuple[Array, ...]]
RunnerFactory = Callable[[], Runner]

CASE_SEED_STRIDE = 1009
ROUND_BATCH_SEED_STRIDE = 7919


@dataclass(frozen=True, slots=True)
class ExpressionRunnerSpec:
    """One expression strategy benchmarked within one gap case."""

    name: str
    role: ExpressionRole
    description: str
    call_repr: str
    available: bool
    reason: str
    reference: Reference
    make_runner: RunnerFactory


@dataclass(frozen=True, slots=True)
class ExpressionParityCase:
    """Dynamic benchmark case with multiple competing expression strategies."""

    name: str
    description: str
    batch_factory: BatchFactory
    runner_specs: tuple[ExpressionRunnerSpec, ...]


@dataclass(frozen=True, slots=True)
class ExpressionRun:
    """One expression strategy timing result."""

    available: bool
    reason: str
    role: ExpressionRole
    dynamic: TimingSummary | None


@dataclass(frozen=True, slots=True)
class ExpressionCaseResult:
    """One gap case measured across expression strategies."""

    case: ExpressionParityCase
    runs: dict[str, ExpressionRun]
    round_orders: list[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ExpressionParityReport:
    """Top-level report for expression-parity benchmark runs."""

    title: str
    configuration: list[str]
    methodology: list[str]
    case_results: tuple[ExpressionCaseResult, ...]
    notes: list[str]


def _round_orders(
    *,
    strategy_names: list[str],
    rounds: int,
    seed: int,
) -> list[tuple[str, ...]]:
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    random_state = np.random.RandomState(seed)
    orders: list[tuple[str, ...]] = []
    for _ in range(rounds):
        order = list(strategy_names)
        random_state.shuffle(order)
        orders.append(tuple(order))
    return orders


def _numpy_batch(
    *,
    backend: BackendSpec,
    batch: tuple[Array, ...],
) -> tuple[NumpyArray, ...]:
    return tuple(backend.to_numpy_array(item) for item in batch)


def _validate_output(
    *,
    backend: BackendSpec,
    runner_spec: ExpressionRunnerSpec,
    batch: tuple[Array, ...],
    output: Output,
) -> None:
    expected = backend.to_numpy_output(
        runner_spec.reference(_numpy_batch(backend=backend, batch=batch))
    )
    got = backend.to_numpy_output(output)
    if len(expected) != len(got):
        raise ValueError(
            f"{runner_spec.name} output arity mismatch: "
            f"expected {len(expected)}, got {len(got)}"
        )

    uses_torch_backend = any(backend.is_torch_tensor(item) for item in batch)
    atol = 1e-4 if uses_torch_backend else 1e-5
    rtol = 1e-4 if uses_torch_backend else 1e-5

    for index, (expected_array, got_array) in enumerate(
        zip(expected, got, strict=True)
    ):
        if expected_array.shape != got_array.shape:
            raise ValueError(
                f"{runner_spec.name} output[{index}] shape mismatch: "
                f"expected {expected_array.shape}, got {got_array.shape}"
            )
        if not np.allclose(expected_array, got_array, atol=atol, rtol=rtol):
            max_abs = float(np.max(np.abs(expected_array - got_array)))
            raise ValueError(
                f"{runner_spec.name} output[{index}] value mismatch: "
                f"max abs diff={max_abs}, atol={atol}, rtol={rtol}"
            )


def _batch_einop_contract_split(
    *,
    generator: TensorGenerator,
    sizes: BenchSizes,
) -> tuple[Array, ...]:
    b_dim = generator.draw_dimension(base=sizes.b)
    n_dim = generator.draw_dimension(base=sizes.n)
    d_dim = generator.draw_dimension(base=sizes.d)
    lhs = generator.randn_numpy((b_dim, (sizes.h + sizes.w) * sizes.r, n_dim))
    rhs = generator.randn_numpy((n_dim, d_dim))
    return generator.backend_batch((lhs, rhs))


def _reference_contract_only(inputs: tuple[NumpyArray, ...]) -> Output:
    lhs, rhs = inputs
    return np.einsum("btn,nd->btd", lhs, rhs)


def _reference_contract_split(
    inputs: tuple[NumpyArray, ...],
    *,
    sizes: BenchSizes,
) -> Output:
    contracted = _reference_contract_only(inputs)
    if not isinstance(contracted, np.ndarray):
        raise TypeError("contract-only reference must produce one ndarray")
    split_index = sizes.h * sizes.r
    return (contracted[:, :split_index, :], contracted[:, split_index:, :])


def _build_gap_cases(*, sizes: BenchSizes) -> tuple[ExpressionParityCase, ...]:
    b, n, d, h, w, r = axes("b", "n", "d", "h", "w", "r")
    split_index = sizes.h * sizes.r
    second_split = sizes.w * sizes.r

    def torch_pair(inputs: tuple[Array, ...]) -> tuple[TorchTensor, TorchTensor]:
        lhs, rhs = inputs
        if torch is None:
            raise RuntimeError("torch is not available")
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            raise TypeError("expression parity benchmark requires torch tensor inputs")
        return lhs, rhs

    def batch_factory(generator: TensorGenerator) -> tuple[Array, ...]:
        return _batch_einop_contract_split(generator=generator, sizes=sizes)

    def make_einf_runner() -> Runner:
        op = einop(
            (ax[b, ((h + w) * r), n], ax[n, d]),
            (ax[b, (h * r), d], ax[b, (w * r), d]),
        ).with_sizes(h=sizes.h, w=sizes.w, r=sizes.r)
        return lambda inputs: op(inputs[0], inputs[1])

    def make_einops_runner() -> Runner:
        if einops is None:
            raise RuntimeError("einops is not available")
        einops_module = einops

        def run(inputs: tuple[Array, ...]) -> Output:
            contracted = einops_module.einsum(
                inputs[0], inputs[1], "b t n, n d -> b t d"
            )
            return (contracted[:, :split_index, :], contracted[:, split_index:, :])

        return run

    def make_einx_runner() -> Runner:
        if einx is None:
            raise RuntimeError("einx is not available")
        einx_module = einx

        def run(inputs: tuple[Array, ...]) -> Output:
            contracted = as_single_array(
                einx_module.dot("b t n, n d -> b t d", inputs[0], inputs[1])
            )
            return (contracted[:, :split_index, :], contracted[:, split_index:, :])

        return run

    def make_torch_matmul_only_runner() -> Runner:
        if torch is None:
            raise RuntimeError("torch is not available")

        def run(inputs: tuple[Array, ...]) -> Output:
            lhs, rhs = torch_pair(inputs)
            return torch.matmul(lhs, rhs)

        return run

    def make_torch_matmul_split_runner() -> Runner:
        if torch is None:
            raise RuntimeError("torch is not available")

        def run(inputs: tuple[Array, ...]) -> Output:
            lhs, rhs = torch_pair(inputs)
            contracted = torch.matmul(lhs, rhs)
            return contracted.split((split_index, second_split), dim=1)

        return run

    def make_torch_matmul_slice_runner() -> Runner:
        if torch is None:
            raise RuntimeError("torch is not available")

        def run(inputs: tuple[Array, ...]) -> Output:
            lhs, rhs = torch_pair(inputs)
            contracted = torch.matmul(lhs, rhs)
            return (contracted[:, :split_index, :], contracted[:, split_index:, :])

        return run

    return (
        ExpressionParityCase(
            name="einop_contract_split_dynamic",
            description=(
                "Dynamic-shape gap case for two-stage contract+split. "
                "Compares library implementations against plain torch expression "
                "variants to separate DSL/runtime overhead from expression choice."
            ),
            batch_factory=batch_factory,
            runner_specs=(
                ExpressionRunnerSpec(
                    name="einf",
                    role="equivalent",
                    description="einf lowering + runtime",
                    call_repr=(
                        "einop((ax[b, ((h + w) * r), n], ax[n, d]), "
                        "(ax[b, (h * r), d], ax[b, (w * r), d]))"
                        ".with_sizes(h=h, w=w, r=r)(lhs, rhs)"
                    ),
                    available=True,
                    reason="available",
                    reference=lambda inputs: _reference_contract_split(
                        inputs, sizes=sizes
                    ),
                    make_runner=make_einf_runner,
                ),
                ExpressionRunnerSpec(
                    name="einops",
                    role="equivalent",
                    description="einops two-stage contract then slicing",
                    call_repr=(
                        'tmp = einops.einsum(lhs, rhs, "b t n, n d -> b t d"); '
                        "(tmp[:, :h*r, :], tmp[:, h*r:, :])"
                    ),
                    available=einops is not None,
                    reason="available" if einops is not None else "not installed",
                    reference=lambda inputs: _reference_contract_split(
                        inputs, sizes=sizes
                    ),
                    make_runner=make_einops_runner,
                ),
                ExpressionRunnerSpec(
                    name="einx",
                    role="equivalent",
                    description="einx dot then slicing",
                    call_repr=(
                        'tmp = einx.dot("b t n, n d -> b t d", lhs, rhs); '
                        "(tmp[:, :h*r, :], tmp[:, h*r:, :])"
                    ),
                    available=einx is not None,
                    reason="available" if einx is not None else "not installed",
                    reference=lambda inputs: _reference_contract_split(
                        inputs, sizes=sizes
                    ),
                    make_runner=make_einx_runner,
                ),
                ExpressionRunnerSpec(
                    name="torch_matmul_only",
                    role="baseline",
                    description="lower bound: contract only, no split tail",
                    call_repr="torch.matmul(lhs, rhs)",
                    available=torch is not None,
                    reason="available" if torch is not None else "torch not installed",
                    reference=_reference_contract_only,
                    make_runner=make_torch_matmul_only_runner,
                ),
                ExpressionRunnerSpec(
                    name="torch_matmul_split",
                    role="equivalent",
                    description="plain torch matmul plus split_with_sizes",
                    call_repr=(
                        "tmp = torch.matmul(lhs, rhs); tmp.split((h*r, w*r), dim=1)"
                    ),
                    available=torch is not None,
                    reason="available" if torch is not None else "torch not installed",
                    reference=lambda inputs: _reference_contract_split(
                        inputs, sizes=sizes
                    ),
                    make_runner=make_torch_matmul_split_runner,
                ),
                ExpressionRunnerSpec(
                    name="torch_matmul_slice",
                    role="equivalent",
                    description="plain torch matmul plus direct slicing",
                    call_repr=(
                        "tmp = torch.matmul(lhs, rhs); "
                        "(tmp[:, :h*r, :], tmp[:, h*r:, :])"
                    ),
                    available=torch is not None,
                    reason="available" if torch is not None else "torch not installed",
                    reference=lambda inputs: _reference_contract_split(
                        inputs, sizes=sizes
                    ),
                    make_runner=make_torch_matmul_slice_runner,
                ),
            ),
        ),
    )


def build_gap_cases(*, sizes: BenchSizes) -> tuple[ExpressionParityCase, ...]:
    """Public helper exposing expression-parity gap cases for sibling audits."""
    return _build_gap_cases(sizes=sizes)


def _find_case(*, sizes: BenchSizes, case_name: str) -> ExpressionParityCase:
    for case in build_gap_cases(sizes=sizes):
        if case.name == case_name:
            return case
    raise ValueError(f"unknown expression parity case: {case_name!r}")


def _run_dynamic_case(
    *,
    case: ExpressionParityCase,
    config: DynamicTaskConfig,
    profiler: Profiler,
    backend: BackendSpec,
    case_index: int,
) -> ExpressionCaseResult:
    available_specs = tuple(spec for spec in case.runner_specs if spec.available)
    runs: dict[str, ExpressionRun] = {
        spec.name: ExpressionRun(
            available=spec.available,
            reason=spec.reason,
            role=spec.role,
            dynamic=None,
        )
        for spec in case.runner_specs
        if not spec.available
    }

    if not available_specs:
        return ExpressionCaseResult(case=case, runs=runs, round_orders=[])

    round_batches = [
        [
            case.batch_factory(
                TensorGenerator.from_seed(
                    backend=backend,
                    seed=(
                        config.seed
                        + case_index * CASE_SEED_STRIDE
                        + round_index * ROUND_BATCH_SEED_STRIDE
                        + batch_index
                    ),
                )
            )
            for batch_index in range(config.batches)
        ]
        for round_index in range(config.rounds)
    ]
    round_orders = _round_orders(
        strategy_names=[spec.name for spec in available_specs],
        rounds=config.rounds,
        seed=config.round_order_seed + case_index * CASE_SEED_STRIDE,
    )
    runners = {spec.name: spec.make_runner() for spec in available_specs}
    reference_batches = round_batches[0]
    checks = min(len(reference_batches), config.parity_checks)
    spec_by_name = {spec.name: spec for spec in case.runner_specs}

    for spec in available_specs:
        runner = runners[spec.name]
        for batch_index in range(checks):
            batch = reference_batches[batch_index]
            output = runner(batch)
            _validate_output(
                backend=backend,
                runner_spec=spec,
                batch=batch,
                output=output,
            )

    samples_by_name: dict[str, list[float]] = {
        spec.name: [] for spec in available_specs
    }
    for round_index, batches in enumerate(round_batches):
        for name in round_orders[round_index]:
            samples_by_name[name].extend(
                profiler.measure_dynamic(
                    runner=runners[name],
                    batches=batches,
                    warmup_batches=config.warmup_batches,
                    repeats=config.repeats,
                )
            )

    for spec in available_specs:
        runs[spec.name] = ExpressionRun(
            available=True,
            reason="available",
            role=spec_by_name[spec.name].role,
            dynamic=profiler.summarize(samples_by_name[spec.name]),
        )

    return ExpressionCaseResult(
        case=case,
        runs=runs,
        round_orders=round_orders,
    )


def _to_json(report: ExpressionParityReport) -> dict[str, object]:
    return {
        "title": report.title,
        "configuration": list(report.configuration),
        "methodology": list(report.methodology),
        "case_results": [
            {
                "case": {
                    "name": case_result.case.name,
                    "description": case_result.case.description,
                    "runner_specs": [
                        {
                            "name": spec.name,
                            "role": spec.role,
                            "description": spec.description,
                            "call_repr": spec.call_repr,
                            "available": spec.available,
                            "reason": spec.reason,
                        }
                        for spec in case_result.case.runner_specs
                    ],
                },
                "runs": {
                    name: {
                        "available": run.available,
                        "reason": run.reason,
                        "role": run.role,
                        "dynamic": None if run.dynamic is None else asdict(run.dynamic),
                    }
                    for name, run in case_result.runs.items()
                },
                "round_orders": [list(order) for order in case_result.round_orders],
            }
            for case_result in report.case_results
        ],
        "notes": list(report.notes),
    }


def _render_markdown(report: ExpressionParityReport) -> str:
    lines = [report.title, "", "## Configuration", ""]
    lines.extend(f"- {entry}" for entry in report.configuration)

    lines.extend(["", "## Methodology", ""])
    lines.extend(f"- {entry}" for entry in report.methodology)

    lines.extend(["", "## Results", ""])
    for case_result in report.case_results:
        lines.extend(
            [
                f"### {case_result.case.name}",
                "",
                case_result.case.description,
                "",
                "Expression strategies:",
                "",
            ]
        )
        for spec in case_result.case.runner_specs:
            lines.append(f"- `{spec.name}` (`{spec.role}`): `{spec.call_repr}`")
        lines.extend(
            [
                "",
                "| Strategy | Role | Samples | Median (ms) | Mean (ms) | P25 (ms) | P75 (ms) | P95 (ms) | Min (ms) | Max (ms) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in case_result.case.runner_specs:
            run = case_result.runs[spec.name]
            if not run.available or run.dynamic is None:
                lines.append(
                    f"| {spec.name} | {spec.role} | n/a ({run.reason}) | - | - | - | - | - | - | - |"
                )
                continue
            summary = run.dynamic
            lines.append(
                f"| {spec.name} | {spec.role} | {summary.count} | "
                f"{summary.median_ms:.4f} | {summary.mean_ms:.4f} | "
                f"{summary.p25_ms:.4f} | {summary.p75_ms:.4f} | "
                f"{summary.p95_ms:.4f} | {summary.min_ms:.4f} | {summary.max_ms:.4f} |"
            )
        lines.extend(["", "Round strategy order (deterministic shuffle):", ""])
        max_orders = 12
        for round_index, order in enumerate(
            case_result.round_orders[:max_orders], start=1
        ):
            lines.append(f"- round {round_index}: `{' -> '.join(order)}`")
        if len(case_result.round_orders) > max_orders:
            hidden = len(case_result.round_orders) - max_orders
            lines.append(f"- ... `{hidden}` additional rounds omitted for brevity")
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.extend(f"- {entry}" for entry in report.notes)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark gap-case expression strategies on dynamic torch batches.",
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
        help="Gap case name to run.",
    )
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--round-order-seed",
        type=int,
        default=None,
        help="Optional seed controlling deterministic per-round strategy order.",
    )
    parser.add_argument(
        "--parity-checks",
        type=int,
        default=8,
        help="Number of first batches used for per-strategy parity validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional raw JSON output path.",
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

    backend = BackendSpec(name="torch")
    backend.validate_available()
    profiler = Profiler(backend=backend)
    sizes = dynamic_sizes_for_scale(args.scale)
    round_order_seed = (
        args.seed if args.round_order_seed is None else args.round_order_seed
    )
    config = DynamicTaskConfig(
        backend="torch",
        scale=args.scale,
        seed=args.seed,
        batches=args.batches,
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
        rounds=args.rounds,
        round_order_seed=round_order_seed,
        parity_checks=args.parity_checks,
    )

    case = _find_case(sizes=sizes, case_name=args.case)
    case_result = _run_dynamic_case(
        case=case,
        config=config,
        profiler=profiler,
        backend=backend,
        case_index=0,
    )
    measured_batches_per_repeat = args.batches - args.warmup_batches
    report = ExpressionParityReport(
        title="# Gap Expression Parity Benchmark",
        configuration=[
            f"Python: `{platform.python_version()}`",
            f"NumPy: `{version_or_missing('numpy')}`",
            f"torch: `{version_or_missing('torch')}`",
            f"einops: `{version_or_missing('einops')}`",
            f"einx: `{version_or_missing('einx')}`",
            "backend: `torch`",
            f"case: `{args.case}`",
            (
                f"sizes(base): `b={sizes.b}, n={sizes.n}, d={sizes.d}, "
                f"h={sizes.h}, w={sizes.w}, r={sizes.r}, j={sizes.j}`"
            ),
            "dynamic dimensions sample per batch in `[0.6x, 1.4x]` of base size",
            f"seed: `{args.seed}`",
            f"round order seed: `{round_order_seed}`",
            f"total batches per case: `{args.batches}`",
            f"warmup batches: `{args.warmup_batches}`",
            f"repeats: `{args.repeats}`",
            f"rounds: `{args.rounds}`",
            f"measured batches per repeat: `{measured_batches_per_repeat}`",
            "table units: `ms`",
        ],
        methodology=[
            "Timing uses eager CPU wall-clock latency for each strategy call; output observation is excluded.",
            "Each expression strategy is benchmarked over identical deterministic torch batches.",
            "Per round, strategy order is shuffled deterministically and all post-warmup per-batch latencies are recorded.",
            "Equivalent strategies validate against the full split result; baseline strategies validate against their own stage output.",
            "This benchmark is gap-attribution oriented: it compares total runtime across DSL and plain torch expression choices.",
        ],
        case_results=(case_result,),
        notes=[
            "torch_matmul_only is a lower-bound baseline and does not produce the final split tuple.",
            "Use this report before profiling internals: if plain torch split/slice already differs materially, the gap is expression-level before DSL overhead.",
        ],
    )

    markdown = _render_markdown(report)
    print(markdown)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(_to_json(report), indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
