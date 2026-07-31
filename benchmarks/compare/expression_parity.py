#!/usr/bin/env python3
"""Benchmark gap-case expression strategies on dynamic torch batches."""

import argparse
import json
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

from benchmarks.harness import (
    BackendSpec,
    DynamicTaskConfig,
    PairedComparison,
    Profiler,
    TensorGenerator,
    TimingSummary,
    dynamic_sizes_for_scale,
    torch,
)
from benchmarks.harness.comparison import compare_paired_timings
from benchmarks.harness.config import BenchSizes
from benchmarks.harness.generator import derive_coordinate_seed
from benchmarks.harness.receipt import (
    execution_target_payload,
    resolve_raw_output_path,
    synchronized_measurement_contract_payload,
)
from benchmarks.harness.types import Array, NumpyArray, Output, Runner
from benchmarks.shared import as_single_array, einf_source_metadata, version_or_missing
from einf import ax, axes, einop

try:
    import einops
except ImportError:
    einops = None

try:
    import einx as _einx
except ImportError:
    einx = None
else:

    class _EinxModule(Protocol):
        def dot(self, expression: str, *tensors: Array) -> Output: ...

    einx = cast(_EinxModule, _einx)


ExpressionSemantics = Literal["output_equivalent", "lower_bound"]
Reference = Callable[[tuple[NumpyArray, ...]], Output]
BatchFactory = Callable[[TensorGenerator], tuple[Array, ...]]
RunnerFactory = Callable[[], Runner]

CASE_SEED_STRIDE = 1009


@dataclass(frozen=True, slots=True)
class ExpressionRunnerSpec:
    """One expression strategy benchmarked within one gap case."""

    name: str
    semantics: ExpressionSemantics
    description: str
    call_repr: str
    available: bool
    reason: str
    reference: Reference
    make_runner: RunnerFactory

    def __post_init__(self) -> None:
        """Reject strategy definitions without stable display identities."""
        if not self.name:
            raise ValueError("expression strategy name must be non-empty")
        if not self.reason:
            raise ValueError(
                "expression strategy availability reason must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class ExpressionParityCase:
    """Dynamic benchmark case with multiple competing expression strategies."""

    name: str
    description: str
    target_name: str
    batch_factory: BatchFactory
    runner_specs: tuple[ExpressionRunnerSpec, ...]

    def __post_init__(self) -> None:
        """Require unique strategy names and one comparison target."""
        strategy_names = tuple(spec.name for spec in self.runner_specs)
        if len(frozenset(strategy_names)) != len(strategy_names):
            raise ValueError("expression strategy names must be unique")
        if self.target_name not in strategy_names:
            raise ValueError("expression parity target must name one strategy")
        if self.target.semantics != "output_equivalent":
            raise ValueError("expression parity target must be output-equivalent")
        if not self.target.available:
            raise ValueError("expression parity target strategy must be available")

    @property
    def target(self) -> ExpressionRunnerSpec:
        """Return the strategy used as the paired comparison target."""
        return next(spec for spec in self.runner_specs if spec.name == self.target_name)


@dataclass(frozen=True, slots=True)
class ExpressionRun:
    """Timing summaries for one available expression strategy."""

    summary: TimingSummary
    round_summaries: tuple[TimingSummary, ...]


@dataclass(frozen=True, slots=True)
class ExpressionObservation:
    """One timed expression call with pairing and execution identity."""

    round_index: int
    unit_index: int
    repeat_index: int
    strategy: str
    order_position: int
    latency_ms: float

    def __post_init__(self) -> None:
        """Reject invalid observation coordinates and timing values."""
        for field_name, value in (
            ("round_index", self.round_index),
            ("unit_index", self.unit_index),
            ("repeat_index", self.repeat_index),
            ("order_position", self.order_position),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if not self.strategy:
            raise ValueError("strategy must be non-empty")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError(
                f"latency_ms must be finite and positive, got {self.latency_ms}"
            )


@dataclass(frozen=True, slots=True)
class ExpressionCaseResult:
    """One gap case measured across expression strategies."""

    case: ExpressionParityCase
    runs: dict[str, ExpressionRun]
    round_orders: list[tuple[str, ...]]
    observations: tuple[ExpressionObservation, ...]
    comparisons: tuple[PairedComparison[str], ...]

    def __post_init__(self) -> None:
        """Keep measured evidence aligned with the case definition."""
        available_names = frozenset(
            spec.name for spec in self.case.runner_specs if spec.available
        )
        if frozenset(self.runs) != available_names:
            raise ValueError(
                "expression runs must match the available strategy definitions"
            )
        if any(
            len(round_order) != len(available_names)
            or frozenset(round_order) != available_names
            for round_order in self.round_orders
        ):
            raise ValueError(
                "expression round orders must be full available strategy permutations"
            )
        if any(
            len(run.round_summaries) != len(self.round_orders)
            for run in self.runs.values()
        ):
            raise ValueError(
                "expression round summaries must match the measured round count"
            )
        observed_names = frozenset(
            observation.strategy for observation in self.observations
        )
        if observed_names != available_names:
            raise ValueError(
                "expression observations must cover every available strategy"
            )
        if any(
            observation.round_index >= len(self.round_orders)
            for observation in self.observations
        ):
            raise ValueError("expression observation references an unknown round")

        target_name = self.case.target.name
        expected_competitors = available_names - {target_name}
        actual_competitors = frozenset(
            comparison.competitor for comparison in self.comparisons
        )
        if (
            any(comparison.baseline != target_name for comparison in self.comparisons)
            or len(actual_competitors) != len(self.comparisons)
            or actual_competitors != expected_competitors
        ):
            raise ValueError(
                "expression comparisons must cover each available non-target strategy"
            )


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
    coordinates_per_round: int,
    seed: int,
) -> list[tuple[str, ...]]:
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if coordinates_per_round < 0:
        raise ValueError(
            f"coordinates_per_round must be non-negative, got {coordinates_per_round}"
        )
    if not strategy_names:
        raise ValueError("strategy_names must be non-empty")
    random_state = np.random.RandomState(seed)
    base_order = list(strategy_names)
    random_state.shuffle(base_order)
    base_order_tuple = tuple(base_order)
    return [
        _rotate_order(
            base_order_tuple,
            offset=round_index * coordinates_per_round,
        )
        for round_index in range(rounds)
    ]


def _rotate_order(order: tuple[str, ...], *, offset: int) -> tuple[str, ...]:
    if not order:
        return ()
    index = offset % len(order)
    return order[index:] + order[:index]


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
    expected: tuple[NumpyArray, ...],
    output: Output,
) -> None:
    backend.validate_output_target(output)
    got = backend.to_numpy_output(output)
    if len(expected) != len(got):
        raise ValueError(
            f"{runner_spec.name} output arity mismatch: "
            f"expected {len(expected)}, got {len(got)}"
        )

    uses_torch_backend = backend.name == "torch"
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

    def torch_pair(inputs: tuple[Array, ...]) -> tuple[Array, Array]:
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
            target_name="einf",
            batch_factory=batch_factory,
            runner_specs=(
                ExpressionRunnerSpec(
                    name="einf",
                    semantics="output_equivalent",
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
                    semantics="output_equivalent",
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
                    semantics="output_equivalent",
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
                    semantics="lower_bound",
                    description="lower bound: contract only, no split tail",
                    call_repr="torch.matmul(lhs, rhs)",
                    available=torch is not None,
                    reason="available" if torch is not None else "torch not installed",
                    reference=_reference_contract_only,
                    make_runner=make_torch_matmul_only_runner,
                ),
                ExpressionRunnerSpec(
                    name="torch_matmul_split",
                    semantics="output_equivalent",
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
                    semantics="output_equivalent",
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
    runs: dict[str, ExpressionRun] = {}

    def make_batch(*, round_index: int, batch_index: int) -> tuple[Array, ...]:
        return case.batch_factory(
            TensorGenerator.from_seed(
                backend=backend,
                seed=derive_coordinate_seed(
                    seed=config.seed,
                    case_index=case_index,
                    round_index=round_index,
                    stream_index=batch_index,
                ),
            )
        )

    measured_batch_count = config.batches - config.warmup_batches
    strategy_names = [spec.name for spec in available_specs]
    order_seed = config.round_order_seed + case_index * CASE_SEED_STRIDE
    round_orders = _round_orders(
        strategy_names=strategy_names,
        rounds=config.rounds,
        coordinates_per_round=config.repeats * measured_batch_count,
        seed=order_seed,
    )
    warmup_round_orders = _round_orders(
        strategy_names=strategy_names,
        rounds=config.rounds,
        coordinates_per_round=config.warmup_batches,
        seed=order_seed,
    )
    expected_strategies = frozenset(spec.name for spec in available_specs)
    if any(
        len(round_order) != len(available_specs)
        or frozenset(round_order) != expected_strategies
        for round_order in round_orders
    ):
        raise RuntimeError(
            "expression benchmark round orders must be full strategy permutations"
        )
    validation_runners = {
        spec.name: spec.make_runner() for spec in available_specs
    }
    checks = min(config.batches, config.parity_checks)
    for batch_index in range(checks):
        batch = make_batch(round_index=0, batch_index=batch_index)
        numpy_batch = _numpy_batch(backend=backend, batch=batch)
        expected_by_name = {
            spec.name: tuple(
                array.copy()
                for array in backend.to_numpy_output(spec.reference(numpy_batch))
            )
            for spec in available_specs
        }
        for spec in available_specs:
            output = validation_runners[spec.name](batch)
            backend.synchronize()
            _validate_output(
                backend=backend,
                runner_spec=spec,
                expected=expected_by_name[spec.name],
                output=output,
            )
            del output
        del batch
    del validation_runners

    runners = {spec.name: spec.make_runner() for spec in available_specs}
    observations: list[ExpressionObservation] = []
    for round_index, round_order in enumerate(round_orders):
        for batch_index in range(config.warmup_batches):
            batch = make_batch(
                round_index=round_index,
                batch_index=batch_index,
            )
            batch_order = _rotate_order(
                warmup_round_orders[round_index],
                offset=batch_index,
            )
            for name in batch_order:
                output = runners[name](batch)
                backend.synchronize()
                backend.validate_output_target(output)
                del output
            del batch

        for repeat_index in range(config.repeats):
            for measured_batch_index in range(measured_batch_count):
                batch = make_batch(
                    round_index=round_index,
                    batch_index=(config.warmup_batches + measured_batch_index),
                )
                batch_order = _rotate_order(
                    round_order,
                    offset=(repeat_index * measured_batch_count + measured_batch_index),
                )
                for order_position, name in enumerate(batch_order):
                    observations.append(
                        ExpressionObservation(
                            round_index=round_index,
                            unit_index=measured_batch_index,
                            repeat_index=repeat_index,
                            strategy=name,
                            order_position=order_position,
                            latency_ms=profiler.measure_call(
                                runner=runners[name],
                                batch=batch,
                            ),
                        )
                    )
                del batch

    observation_tuple = tuple(observations)
    available_names = tuple(spec.name for spec in available_specs)
    for spec in available_specs:
        spec_observations = [
            observation
            for observation in observations
            if observation.strategy == spec.name
        ]
        runs[spec.name] = ExpressionRun(
            summary=profiler.summarize(
                [observation.latency_ms for observation in spec_observations]
            ),
            round_summaries=tuple(
                profiler.summarize(
                    [
                        observation.latency_ms
                        for observation in spec_observations
                        if observation.round_index == round_index
                    ]
                )
                for round_index in range(config.rounds)
            ),
        )

    def strategy_of(observation: ExpressionObservation) -> str:
        return observation.strategy

    comparisons = compare_paired_timings(
        observations=observation_tuple,
        members=available_names,
        baseline=case.target.name,
        member_of=strategy_of,
        bootstrap_seed=config.seed + case_index * CASE_SEED_STRIDE,
    )
    return ExpressionCaseResult(
        case=case,
        runs=runs,
        round_orders=round_orders,
        observations=observation_tuple,
        comparisons=comparisons,
    )


def _to_json(
    report: ExpressionParityReport,
    *,
    backend: BackendSpec,
) -> dict[str, object]:
    return {
        "schema_version": 4,
        "title": report.title,
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
        "configuration": list(report.configuration),
        "methodology": list(report.methodology),
        "execution_target": execution_target_payload(backend),
        "measurement_contract": {
            **synchronized_measurement_contract_payload(),
            "schedule": "paired_coordinate_rotating_order",
            "order": "continuous_rotation_from_one_deterministic_shuffled_order",
            "pairing_identity": [
                "round_index",
                "unit_index",
                "repeat_index",
            ],
            "estimand": (
                "competitor_mean_batch_latency_over_target_mean_batch_latency"
            ),
            "repeat_handling": "average_within_round_and_measured_batch",
            "batch_stream": "regenerated_from_the_same_seed_for_each_repeat",
            "uncertainty": "round_stratified_batch_bootstrap",
        },
        "case_results": [
            {
                "case": {
                    "name": case_result.case.name,
                    "description": case_result.case.description,
                    "target": case_result.case.target_name,
                    "runner_specs": [
                        {
                            "name": spec.name,
                            "is_target": spec.name == case_result.case.target_name,
                            "semantics": spec.semantics,
                            "description": spec.description,
                            "call_repr": spec.call_repr,
                            "available": spec.available,
                            "reason": spec.reason,
                        }
                        for spec in case_result.case.runner_specs
                    ],
                },
                "runs": {
                    spec.name: (
                        {
                            "status": "available",
                            "summary": asdict(case_result.runs[spec.name].summary),
                            "round_summaries": [
                                asdict(round_summary)
                                for round_summary in case_result.runs[
                                    spec.name
                                ].round_summaries
                            ],
                        }
                        if spec.available
                        else {
                            "status": "unavailable",
                            "reason": spec.reason,
                        }
                    )
                    for spec in case_result.case.runner_specs
                },
                "measured_round_start_orders": [
                    list(order) for order in case_result.round_orders
                ],
                "observations": [
                    {
                        "round_index": observation.round_index,
                        "unit_index": observation.unit_index,
                        "repeat_index": observation.repeat_index,
                        "strategy": observation.strategy,
                        "order_position": observation.order_position,
                        "latency_ms": observation.latency_ms,
                    }
                    for observation in case_result.observations
                ],
                "comparisons": [
                    {
                        "target": comparison.baseline,
                        "competitor": comparison.competitor,
                        "call_pair_count": comparison.call_pair_count,
                        "paired_unit_count": comparison.paired_unit_count,
                        "latency_ratio": comparison.latency_ratio,
                        "confidence_level": comparison.confidence_level,
                        "confidence_interval": {
                            "low": comparison.confidence_interval_low,
                            "high": comparison.confidence_interval_high,
                        },
                        "bootstrap_resamples": comparison.bootstrap_resamples,
                        "bootstrap_seed": comparison.bootstrap_seed,
                    }
                    for comparison in case_result.comparisons
                ],
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
            availability = "" if spec.available else f"; unavailable: {spec.reason}"
            target = "; target" if spec.name == case_result.case.target_name else ""
            lines.append(
                f"- `{spec.name}` (`{spec.semantics}`{target}{availability}): "
                f"`{spec.call_repr}`"
            )
        lines.extend(
            [
                "",
                "| Strategy | Target | Semantics | Samples | Median (ms) | Mean (ms) | P25 (ms) | P75 (ms) | P95 (ms) | Min (ms) | Max (ms) |",
                "|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in case_result.case.runner_specs:
            target = "yes" if spec.name == case_result.case.target_name else ""
            if not spec.available:
                lines.append(
                    f"| {spec.name} | {target} | {spec.semantics} | "
                    f"n/a ({spec.reason}) | - | - | - | - | - | - | - |"
                )
                continue
            summary = case_result.runs[spec.name].summary
            lines.append(
                f"| {spec.name} | {target} | {spec.semantics} | {summary.count} | "
                f"{summary.median_ms:.4f} | {summary.mean_ms:.4f} | "
                f"{summary.p25_ms:.4f} | {summary.p75_ms:.4f} | "
                f"{summary.p95_ms:.4f} | {summary.min_ms:.4f} | {summary.max_ms:.4f} |"
            )

        spec_by_name = {spec.name: spec for spec in case_result.case.runner_specs}
        comparison_sections: tuple[
            tuple[ExpressionSemantics, str, str],
            ...,
        ] = (
            (
                "output_equivalent",
                "Equivalent-output comparisons",
                (
                    "Ratios compare each output-equivalent strategy with the "
                    f"target `{case_result.case.target.name}`. Values below 1.0 "
                    "favor the competitor."
                ),
            ),
            (
                "lower_bound",
                "Lower-bound diagnostic",
                (
                    "This ratio removes part of the target work. It estimates a "
                    "floor, not the performance of an equivalent replacement."
                ),
            ),
        )
        for semantics, heading, explanation in comparison_sections:
            comparisons = [
                comparison
                for comparison in case_result.comparisons
                if spec_by_name[comparison.competitor].semantics == semantics
            ]
            if not comparisons:
                continue
            lines.extend(
                [
                    "",
                    f"#### {heading}",
                    "",
                    explanation,
                    "",
                    "| Competitor | Ratio vs target | 95% CI | Paired batches | Paired call coordinates |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for comparison in comparisons:
                lines.append(
                    f"| {comparison.competitor} | "
                    f"{comparison.latency_ratio:.4f} | "
                    f"[{comparison.confidence_interval_low:.4f}, "
                    f"{comparison.confidence_interval_high:.4f}] | "
                    f"{comparison.paired_unit_count} | "
                    f"{comparison.call_pair_count} |"
                )

        lines.extend(
            [
                "",
                (
                    "Measured starting order for each round (then rotated once "
                    "per batch coordinate):"
                ),
                "",
            ]
        )
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
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch execution device, such as cpu, mps, or cuda:0.",
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
        help="Optional seed controlling the initial deterministic strategy order.",
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
        help="Optional raw JSON path; defaults to --output with a .json suffix.",
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
    if args.batches - args.warmup_batches < 2:
        raise ValueError(
            "at least two measured batches are required for paired uncertainty"
        )
    if args.repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {args.repeats}")
    if args.rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {args.rounds}")
    if args.parity_checks < 0:
        raise ValueError(f"parity-checks must be >= 0, got {args.parity_checks}")
    raw_output_path = resolve_raw_output_path(
        output=args.output,
        raw_output=args.raw_output,
    )

    backend = BackendSpec(name="torch", requested_device=args.device)
    profiler = Profiler(backend=backend)
    sizes = dynamic_sizes_for_scale(args.scale)
    round_order_seed = (
        args.seed if args.round_order_seed is None else args.round_order_seed
    )
    config = DynamicTaskConfig(
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
            f"requested device: `{backend.requested_device}`",
            f"resolved device: `{backend.resolved_device}`",
            f"case: `{args.case}`",
            (
                f"sizes(base): `b={sizes.b}, n={sizes.n}, d={sizes.d}, "
                f"h={sizes.h}, w={sizes.w}, r={sizes.r}, j={sizes.j}`"
            ),
            "dynamic dimensions sample per batch in `[0.6x, 1.4x]` of base size",
            f"seed: `{args.seed}`",
            f"strategy order seed: `{round_order_seed}`",
            f"total batches per case: `{args.batches}`",
            f"warmup batches: `{args.warmup_batches}`",
            f"repeats: `{args.repeats}`",
            f"rounds: `{args.rounds}`",
            f"measured batches per repeat: `{measured_batches_per_repeat}`",
            *(
                [f"raw JSON artifact: `{raw_output_path}`"]
                if raw_output_path is not None
                else []
            ),
            "table units: `ms`",
        ],
        methodology=[
            (
                "Each repeat regenerates the logical batch from the same seed. "
                "One target batch is then shared by every strategy at that paired "
                "coordinate."
            ),
            (
                "Strategies run back-to-back on each logical batch, and their "
                "order rotates continuously across measured coordinates from one "
                "deterministically shuffled base order."
            ),
            (
                "The timer starts after target synchronization and stops when the "
                "strategy's submitted work has completed. Batch generation, device "
                "transfer, and output validation stay outside the interval."
            ),
            (
                "Parity checks run before timing on disposable runners. A mismatch "
                "aborts without warming the runner instances used for measurement."
            ),
            (
                "Every timed call retains its round, measured batch, repeat, "
                "strategy, execution position, and latency identity."
            ),
            (
                "The paired ratio averages repeats within each round and batch, "
                "then divides the competitor's mean batch latency by the target's."
            ),
            (
                "The 95% interval resamples measured batches within each round. "
                "Repeated calls are not treated as independent batches."
            ),
            (
                "Equivalent strategies validate the full split result. The "
                "lower-bound strategy validates only the contraction it performs "
                "and is reported separately."
            ),
        ],
        case_results=(case_result,),
        notes=[
            (
                "`torch_matmul_only` is a lower bound and does not produce the "
                "final split tuple."
            ),
            (
                "Use this report before profiling internals. A material gap in "
                "plain torch split or slice variants points to expression-level "
                "work before `einf` internals are considered."
            ),
        ],
    )

    markdown = _render_markdown(report)
    print(markdown)
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(
            json.dumps(_to_json(report, backend=backend), indent=2),
            encoding="utf-8",
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
