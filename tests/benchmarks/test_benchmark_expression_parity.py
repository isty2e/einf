import json
from collections.abc import Callable
from pathlib import Path
from weakref import ReferenceType, ref

import numpy as np
import pytest

import benchmarks.compare.expression_parity as expression_parity_module
import benchmarks.harness.profiler as profiler_module
from benchmarks.compare.expression_parity import (
    ExpressionCaseResult,
    ExpressionObservation,
    ExpressionParityCase,
    ExpressionParityReport,
    ExpressionRun,
    ExpressionRunnerSpec,
    ExpressionSemantics,
    _find_case,
    _render_markdown,
    _round_orders,
    _run_dynamic_case,
    _to_json,
    _validate_output,
    main,
)
from benchmarks.harness import (
    Array,
    BackendSpec,
    DynamicTaskConfig,
    Profiler,
    TensorGenerator,
    TimingSummary,
    dynamic_sizes_for_scale,
)


def _summary(*, mean_ms: float = 5.6) -> TimingSummary:
    return TimingSummary(
        count=12,
        p25_ms=5.0,
        median_ms=5.5,
        p75_ms=6.0,
        iqr_ms=1.0,
        p95_ms=6.8,
        mean_ms=mean_ms,
        min_ms=4.9,
        max_ms=7.1,
    )


def _case(
    *,
    events: list[tuple[str, np.ndarray]] | None = None,
    factory_events: list[str] | None = None,
) -> ExpressionParityCase:
    def make_spec(
        name: str,
        semantics: ExpressionSemantics,
    ) -> ExpressionRunnerSpec:
        def make_runner() -> Callable[[tuple[Array, ...]], Array]:
            if factory_events is not None:
                factory_events.append(name)

            def run(batch: tuple[Array, ...]) -> Array:
                array = batch[0]
                assert isinstance(array, np.ndarray)
                if events is not None:
                    events.append((name, array))
                return array

            return run

        return ExpressionRunnerSpec(
            name=name,
            semantics=semantics,
            description=f"{semantics} strategy",
            call_repr=f"{name}(x)",
            available=True,
            reason="available",
            reference=lambda inputs: inputs[0],
            make_runner=make_runner,
        )

    def batch_factory(generator: TensorGenerator) -> tuple[Array, ...]:
        return generator.backend_batch((generator.randn_numpy((2,)),))

    return ExpressionParityCase(
        name="expression_case",
        description="Synthetic expression comparison.",
        target_name="target",
        batch_factory=batch_factory,
        runner_specs=(
            make_spec("target", "output_equivalent"),
            make_spec("equivalent", "output_equivalent"),
            make_spec("lower_bound", "lower_bound"),
        ),
    )


def test_find_case_exposes_target_and_strategy_semantics() -> None:
    case = _find_case(
        sizes=dynamic_sizes_for_scale("large"),
        case_name="einop_contract_split_dynamic",
    )

    assert [spec.name for spec in case.runner_specs] == [
        "einf",
        "einops",
        "einx",
        "torch_matmul_only",
        "torch_matmul_split",
        "torch_matmul_slice",
    ]
    assert case.target.name == "einf"
    assert (
        next(
            spec for spec in case.runner_specs if spec.name == "torch_matmul_only"
        ).semantics
        == "lower_bound"
    )


def test_expression_case_requires_unique_names_and_valid_target() -> None:
    case = _case()
    target, equivalent, lower_bound = case.runner_specs

    with pytest.raises(ValueError, match="names must be unique"):
        ExpressionParityCase(
            name="duplicate",
            description="duplicate names",
            target_name="target",
            batch_factory=case.batch_factory,
            runner_specs=(target, target),
        )

    with pytest.raises(ValueError, match="target must name one strategy"):
        ExpressionParityCase(
            name="missing_target",
            description="missing target",
            target_name="missing",
            batch_factory=case.batch_factory,
            runner_specs=(equivalent, lower_bound),
        )

    with pytest.raises(ValueError, match="target must be output-equivalent"):
        ExpressionParityCase(
            name="lower_bound_target",
            description="lower-bound target",
            target_name="lower_bound",
            batch_factory=case.batch_factory,
            runner_specs=(target, lower_bound),
        )

    unavailable_target = ExpressionRunnerSpec(
        name="unavailable_target",
        semantics="output_equivalent",
        description="unavailable target",
        call_repr="target(x)",
        available=False,
        reason="not installed",
        reference=target.reference,
        make_runner=target.make_runner,
    )
    with pytest.raises(ValueError, match="target strategy must be available"):
        ExpressionParityCase(
            name="unavailable_target",
            description="unavailable target",
            target_name="unavailable_target",
            batch_factory=case.batch_factory,
            runner_specs=(unavailable_target, equivalent),
        )


def test_expression_strategy_requires_stable_identity() -> None:
    case = _case()
    target = case.target

    with pytest.raises(ValueError, match="name must be non-empty"):
        ExpressionRunnerSpec(
            name="",
            semantics="output_equivalent",
            description=target.description,
            call_repr=target.call_repr,
            available=True,
            reason="available",
            reference=target.reference,
            make_runner=target.make_runner,
        )


def test_round_orders_balance_strategy_positions_across_rounds() -> None:
    strategy_names = ["target", "equivalent", "lower_bound"]

    orders = _round_orders(
        strategy_names=strategy_names,
        rounds=6,
        coordinates_per_round=2,
        seed=17,
    )

    assert all(frozenset(order) == frozenset(strategy_names) for order in orders)
    for strategy_name in strategy_names:
        positions = [order.index(strategy_name) for order in orders]
        assert positions.count(0) == 2
        assert positions.count(1) == 2
        assert positions.count(2) == 2


@pytest.mark.parametrize(
    ("strategy_count", "rounds", "coordinates_per_round"),
    (
        (2, 3, 5),
        (4, 2, 2),
        (5, 3, 7),
        (6, 7, 11),
    ),
)
def test_rotated_schedule_bounds_execution_position_imbalance(
    strategy_count: int,
    rounds: int,
    coordinates_per_round: int,
) -> None:
    strategy_names = [f"strategy_{index}" for index in range(strategy_count)]
    round_orders = _round_orders(
        strategy_names=strategy_names,
        rounds=rounds,
        coordinates_per_round=coordinates_per_round,
        seed=17,
    )
    counts_by_strategy = {
        strategy_name: [0] * strategy_count for strategy_name in strategy_names
    }

    for round_order in round_orders:
        for coordinate_index in range(coordinates_per_round):
            actual_order = expression_parity_module._rotate_order(
                round_order,
                offset=coordinate_index,
            )
            for order_position, strategy_name in enumerate(actual_order):
                counts_by_strategy[strategy_name][order_position] += 1

    assert all(
        max(position_counts) - min(position_counts) <= 1
        for position_counts in counts_by_strategy.values()
    )


def test_validate_output_uses_runner_specific_reference() -> None:
    case = _find_case(
        sizes=dynamic_sizes_for_scale("medium"),
        case_name="einop_contract_split_dynamic",
    )
    runner_spec = next(
        spec for spec in case.runner_specs if spec.name == "torch_matmul_only"
    )

    lhs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    rhs = np.arange(20, dtype=np.float32).reshape(4, 5)
    batch = (lhs, rhs)
    output = runner_spec.reference(batch)

    _validate_output(
        backend=BackendSpec(name="numpy"),
        runner_spec=runner_spec,
        expected=(np.asarray(output),),
        output=output,
    )


def test_run_dynamic_case_validates_against_unmodified_canonical_batch() -> None:
    case = _case()
    target = case.target

    def make_mutating_runner() -> Callable[[tuple[Array, ...]], Array]:
        def run(batch: tuple[Array, ...]) -> Array:
            array = batch[0]
            assert isinstance(array, np.ndarray)
            array.fill(0.0)
            return array

        return run

    mutating_target = ExpressionRunnerSpec(
        name=target.name,
        semantics=target.semantics,
        description=target.description,
        call_repr=target.call_repr,
        available=target.available,
        reason=target.reason,
        reference=target.reference,
        make_runner=make_mutating_runner,
    )
    mutation_case = ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name=case.target_name,
        batch_factory=case.batch_factory,
        runner_specs=(mutating_target, *case.runner_specs[1:]),
    )

    with pytest.raises(ValueError, match="value mismatch"):
        _run_dynamic_case(
            case=mutation_case,
            config=DynamicTaskConfig(
                backend="numpy",
                scale="medium",
                seed=7,
                batches=3,
                warmup_batches=1,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
                parity_checks=1,
            ),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_parity_checks_use_disposable_runner_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_events: list[str] = []
    clock_tick = iter(index / 1000.0 for index in range(12))
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        lambda: next(clock_tick),
    )

    _run_dynamic_case(
        case=_case(factory_events=factory_events),
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=1,
            rounds=1,
            round_order_seed=1234,
            parity_checks=1,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    assert factory_events == [
        "target",
        "equivalent",
        "lower_bound",
        "target",
        "equivalent",
        "lower_bound",
    ]


def test_run_dynamic_case_interleaves_paired_batches_and_preserves_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, np.ndarray]] = []
    case = _case(events=events)
    clock_values: list[float] = []
    for call_index in range(12):
        started = float(call_index)
        clock_values.extend((started, started + (call_index + 1) / 1000.0))
    clock_iterator = iter(clock_values)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        lambda: next(clock_iterator),
    )

    result = _run_dynamic_case(
        case=case,
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
            parity_checks=0,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    round_order = result.round_orders[0]
    expected_orders = [
        round_order,
        round_order,
        expression_parity_module._rotate_order(round_order, offset=1),
        expression_parity_module._rotate_order(round_order, offset=2),
        expression_parity_module._rotate_order(round_order, offset=3),
    ]
    assert len(events) == 15
    for coordinate_index, expected_order in enumerate(expected_orders):
        coordinate_events = events[
            coordinate_index * len(round_order) : (coordinate_index + 1)
            * len(round_order)
        ]
        assert [name for name, _ in coordinate_events] == list(expected_order)
        arrays = [array for _, array in coordinate_events]
        assert np.array_equal(arrays[0], arrays[1])
        assert np.array_equal(arrays[0], arrays[2])
        assert arrays[0] is arrays[1]
        assert arrays[0] is arrays[2]

    assert [observation.latency_ms for observation in result.observations] == (
        pytest.approx(list(range(1, 13)))
    )
    measured_orders = expected_orders[1:]
    for coordinate_index, expected_order in enumerate(measured_orders):
        start = coordinate_index * len(round_order)
        coordinate_observations = result.observations[start : start + len(round_order)]
        assert [item.strategy for item in coordinate_observations] == list(
            expected_order
        )
        assert [item.order_position for item in coordinate_observations] == [0, 1, 2]
    first_repeat_batch = events[3:6]
    second_repeat_batch = events[9:12]
    assert np.array_equal(first_repeat_batch[0][1], second_repeat_batch[0][1])

    assert set(result.runs) == {"target", "equivalent", "lower_bound"}
    assert all(run.summary.count == 4 for run in result.runs.values())
    assert all(len(run.round_summaries) == 1 for run in result.runs.values())
    assert [comparison.baseline for comparison in result.comparisons] == [
        "target",
        "target",
    ]
    assert {comparison.competitor for comparison in result.comparisons} == {
        "equivalent",
        "lower_bound",
    }


def test_run_dynamic_case_preserves_latency_identity_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values: list[float] = []
    for call_index in range(12):
        started = float(call_index)
        clock_values.extend((started, started + (call_index + 1) / 1000.0))
    clock_iterator = iter(clock_values)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        lambda: next(clock_iterator),
    )

    result = _run_dynamic_case(
        case=_case(),
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=1,
            rounds=2,
            round_order_seed=1234,
            parity_checks=0,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    assert [observation.latency_ms for observation in result.observations] == (
        pytest.approx(list(range(1, 13)))
    )
    assert [observation.round_index for observation in result.observations] == (
        [0] * 6 + [1] * 6
    )


def test_run_dynamic_case_releases_batches_before_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    batch_references: list[ReferenceType[np.ndarray]] = []
    live_batches_before_generation: list[int] = []

    def batch_factory(generator: TensorGenerator) -> tuple[Array, ...]:
        live_batches_before_generation.append(
            sum(reference() is not None for reference in batch_references)
        )
        batch = generator.randn_numpy((2,))
        batch_references.append(ref(batch))
        return generator.backend_batch((batch,))

    bounded_case = ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name=case.target_name,
        batch_factory=batch_factory,
        runner_specs=case.runner_specs,
    )
    clock_tick = iter(index / 1000.0 for index in range(24))
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        lambda: next(clock_tick),
    )

    _run_dynamic_case(
        case=bounded_case,
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
            parity_checks=0,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    assert live_batches_before_generation == [0, 0, 0, 0, 0]


def test_run_dynamic_case_rejects_partial_round_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        expression_parity_module,
        "_round_orders",
        lambda **_: [("target", "equivalent")],
    )

    with pytest.raises(RuntimeError, match="full strategy permutations"):
        _run_dynamic_case(
            case=_case(),
            config=DynamicTaskConfig(
                backend="numpy",
                scale="medium",
                seed=7,
                batches=3,
                warmup_batches=1,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
                parity_checks=0,
            ),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_expression_observation_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="strategy must be non-empty"):
        ExpressionObservation(
            round_index=0,
            unit_index=0,
            repeat_index=0,
            strategy="",
            order_position=0,
            latency_ms=1.0,
        )


def test_expression_result_rejects_missing_available_run() -> None:
    case = _case()
    summary = _summary()

    with pytest.raises(ValueError, match="runs must match"):
        ExpressionCaseResult(
            case=case,
            runs={
                "target": ExpressionRun(
                    summary=summary,
                    round_summaries=(summary,),
                )
            },
            round_orders=[tuple(spec.name for spec in case.runner_specs)],
            observations=(
                ExpressionObservation(
                    round_index=0,
                    unit_index=0,
                    repeat_index=0,
                    strategy="target",
                    order_position=0,
                    latency_ms=1.0,
                ),
            ),
            comparisons=(),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ["--batches", "2", "--warmup-batches", "1"],
            "at least two measured batches",
        ),
        (["--parity-checks", "-1"], "parity-checks must be >= 0"),
        (
            ["--output", "report.txt", "--raw-output", "report.txt"],
            "must use different paths",
        ),
    ),
)
def test_main_rejects_invalid_evidence_configuration(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["expression_parity.py", *arguments],
    )

    with pytest.raises(ValueError, match=message):
        main()


def test_expression_parity_renderers_preserve_inference_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    clock_values: list[float] = []
    for call_index in range(12):
        started = float(call_index)
        clock_values.extend((started, started + (call_index + 1) / 1000.0))
    clock_iterator = iter(clock_values)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        lambda: next(clock_iterator),
    )
    case_result = _run_dynamic_case(
        case=case,
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
            parity_checks=0,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )
    report = ExpressionParityReport(
        title="# Gap Expression Parity Benchmark",
        configuration=["backend: `torch`"],
        methodology=["interleaved per-logical-batch timing"],
        case_results=(case_result,),
        notes=["lower bound is not output-equivalent"],
    )

    payload = _to_json(report, backend=BackendSpec(name="numpy"))
    json.dumps(payload)
    assert payload["schema_version"] == 3
    environment = payload["environment"]
    assert isinstance(environment, dict)
    assert {"python", "platform", "machine", "numpy", "torch", "einops", "einx", "einf"} <= set(
        environment
    )
    assert payload["execution_target"] == {
        "backend": "numpy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
    }
    measurement_contract = payload["measurement_contract"]
    assert isinstance(measurement_contract, dict)
    assert measurement_contract["schedule"] == "paired_coordinate_rotating_order"
    assert (
        measurement_contract["synchronization"]
        == "before_timer_start_and_before_timer_stop"
    )
    case_results = payload["case_results"]
    assert isinstance(case_results, list)
    case_payload = case_results[0]
    assert case_payload["case"]["target"] == "target"
    assert case_payload["case"]["runner_specs"][0]["is_target"] is True
    assert case_payload["case"]["runner_specs"][0]["semantics"] == "output_equivalent"
    assert case_payload["observations"][0]["strategy"] == "target"
    assert case_payload["observations"][0]["unit_index"] == 0
    assert case_payload["comparisons"][0]["target"] == "target"

    markdown = _render_markdown(report)
    path = tmp_path / "expression-parity.md"
    path.write_text(markdown, encoding="utf-8")

    assert "#### Equivalent-output comparisons" in markdown
    assert "#### Lower-bound diagnostic" in markdown
    assert "not the performance of an equivalent replacement" in markdown
    assert "Measured starting order for each round" in markdown
