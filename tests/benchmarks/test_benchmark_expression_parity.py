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


def _dynamic_config(
    *,
    batches: int = 2,
    warmup_batches: int = 0,
    repeats: int = 1,
    rounds: int = 1,
) -> DynamicTaskConfig:
    return DynamicTaskConfig(
        scale="medium",
        seed=7,
        batches=batches,
        warmup_batches=warmup_batches,
        repeats=repeats,
        rounds=rounds,
        round_order_seed=1234,
    )


def _case(
    *,
    events: list[tuple[str, np.ndarray]] | None = None,
    factory_events: list[str] | None = None,
    runner_events: list[tuple[str, int, np.ndarray]] | None = None,
) -> ExpressionParityCase:
    factory_counts: dict[str, int] = {}

    def make_spec(
        name: str,
        semantics: ExpressionSemantics,
    ) -> ExpressionRunnerSpec:
        def make_runner() -> Callable[[tuple[Array, ...]], Array]:
            factory_counts[name] = factory_counts.get(name, 0) + 1
            factory_generation = factory_counts[name]
            if factory_events is not None:
                factory_events.append(name)

            def run(batch: tuple[Array, ...]) -> Array:
                array = batch[0]
                assert isinstance(array, np.ndarray)
                if events is not None:
                    events.append((name, array))
                if runner_events is not None:
                    runner_events.append((name, factory_generation, array))
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


def _with_target_runner(
    case: ExpressionParityCase,
    /,
    *,
    make_runner: Callable[[], Callable[[tuple[Array, ...]], Array]],
    batch_factory: Callable[[TensorGenerator], tuple[Array, ...]] | None = None,
) -> ExpressionParityCase:
    target = case.target
    replacement = ExpressionRunnerSpec(
        name=target.name,
        semantics=target.semantics,
        description=target.description,
        call_repr=target.call_repr,
        available=target.available,
        reason=target.reason,
        reference=target.reference,
        make_runner=make_runner,
    )
    return ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name=case.target_name,
        batch_factory=case.batch_factory if batch_factory is None else batch_factory,
        runner_specs=(replacement, *case.runner_specs[1:]),
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


def test_run_dynamic_case_rejects_runner_input_mutation() -> None:
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

    with pytest.raises(ValueError, match="read-only"):
        _run_dynamic_case(
            case=mutation_case,
            config=DynamicTaskConfig(
                scale="medium",
                seed=7,
                batches=3,
                warmup_batches=1,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
            ),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_run_dynamic_case_isolates_strategy_reference_inputs() -> None:
    case = _case()

    def mutating_reference(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        array = inputs[0]
        expected = array.copy()
        array.resize((3,), refcheck=False)
        array[:] = (10.0, 20.0, 30.0)
        return expected

    runner_specs = tuple(
        ExpressionRunnerSpec(
            name=spec.name,
            semantics=spec.semantics,
            description=spec.description,
            call_repr=spec.call_repr,
            available=spec.available,
            reason=spec.reason,
            reference=mutating_reference,
            make_runner=spec.make_runner,
        )
        for spec in case.runner_specs
    )
    mutation_case = ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name=case.target_name,
        batch_factory=case.batch_factory,
        runner_specs=runner_specs,
    )

    result = _run_dynamic_case(
        case=mutation_case,
        config=_dynamic_config(),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    assert len(result.observations) == 2 * len(runner_specs)


def test_run_dynamic_case_validates_every_measured_coordinate() -> None:
    case = _case()
    factory_count = 0

    def make_stateful_runner() -> Callable[[tuple[Array, ...]], Array]:
        nonlocal factory_count
        factory_count += 1
        call_count = 0

        def run(batch: tuple[Array, ...]) -> Array:
            nonlocal call_count
            call_count += 1
            array = batch[0]
            assert isinstance(array, np.ndarray)
            if call_count == 1:
                return array
            return np.zeros_like(array)

        return run

    late_wrong_case = _with_target_runner(
        case,
        make_runner=make_stateful_runner,
    )

    with pytest.raises(ValueError, match="value mismatch"):
        _run_dynamic_case(
            case=late_wrong_case,
            config=_dynamic_config(),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )

    assert factory_count == 1


def test_run_dynamic_case_rejects_shape_first_timed_output() -> None:
    def batch_factory(generator: TensorGenerator) -> tuple[Array, ...]:
        size = generator.draw_dimension(base=4)
        return generator.backend_batch((np.arange(1, size + 1, dtype=np.float32),))

    seen_shapes: set[tuple[int, ...]] = set()

    def make_shape_cached_runner() -> Callable[[tuple[Array, ...]], Array]:
        def run(batch: tuple[Array, ...]) -> Array:
            array = batch[0]
            assert isinstance(array, np.ndarray)
            shape = tuple(int(dimension) for dimension in array.shape)
            if shape not in seen_shapes:
                seen_shapes.add(shape)
                return np.zeros_like(array)
            return array

        return run

    case = _with_target_runner(
        _case(),
        make_runner=make_shape_cached_runner,
        batch_factory=batch_factory,
    )

    with pytest.raises(ValueError, match="value mismatch"):
        _run_dynamic_case(
            case=case,
            config=_dynamic_config(),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_run_dynamic_case_rejects_repeat_specific_timed_output() -> None:
    call_count = 0

    def make_repeat_sensitive_runner() -> Callable[[tuple[Array, ...]], Array]:
        def run(batch: tuple[Array, ...]) -> Array:
            nonlocal call_count
            call_count += 1
            array = batch[0]
            assert isinstance(array, np.ndarray)
            if call_count == 3:
                return np.zeros_like(array)
            return array

        return run

    case = _with_target_runner(
        _case(),
        make_runner=make_repeat_sensitive_runner,
    )

    with pytest.raises(ValueError, match="value mismatch"):
        _run_dynamic_case(
            case=case,
            config=_dynamic_config(repeats=2),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_run_dynamic_case_rejects_input_shape_drift() -> None:
    case = _case()
    generation_count = 0

    def batch_factory(generator: TensorGenerator) -> tuple[Array, ...]:
        nonlocal generation_count
        generation_count += 1
        size = 2 if generation_count <= 2 else 3
        return generator.backend_batch((generator.randn_numpy((size,)),))

    drifting_case = ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name=case.target_name,
        batch_factory=batch_factory,
        runner_specs=case.runner_specs,
    )

    with pytest.raises(RuntimeError, match="repeated input produced different shapes"):
        _run_dynamic_case(
            case=drifting_case,
            config=_dynamic_config(repeats=2),
            profiler=Profiler(backend=BackendSpec(name="numpy")),
            backend=BackendSpec(name="numpy"),
            case_index=0,
        )


def test_run_dynamic_case_validates_every_measured_execution() -> None:
    runner_events: list[tuple[str, int, np.ndarray]] = []

    result = _run_dynamic_case(
        case=_case(runner_events=runner_events),
        config=_dynamic_config(
            batches=3,
            warmup_batches=1,
            repeats=3,
            rounds=2,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    calls = [name for name, _, _ in runner_events]
    assert {name: calls.count(name) for name in set(calls)} == {
        "target": 14,
        "equivalent": 14,
        "lower_bound": 14,
    }
    assert len(result.observations) == 36
    assert {generation for _, generation, _ in runner_events} == {1}
    assert [(unit.round_index, unit.unit_index) for unit in result.realized_units] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_measurement_constructs_each_runner_once(
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
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=1,
            rounds=1,
            round_order_seed=1234,
        ),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        backend=BackendSpec(name="numpy"),
        case_index=0,
    )

    assert factory_events == [
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
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
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
    measured_orders = expected_orders[1:5]
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
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=1,
            rounds=2,
            round_order_seed=1234,
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
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
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
                scale="medium",
                seed=7,
                batches=3,
                warmup_batches=1,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
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
            realized_units=(),
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
    config = DynamicTaskConfig(
        scale="medium",
        seed=7,
        batches=3,
        warmup_batches=1,
        repeats=2,
        rounds=1,
        round_order_seed=1234,
    )
    case_result = _run_dynamic_case(
        case=case,
        config=config,
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

    source_metadata: dict[str, str | bool | None] = {
        "kind": "git_checkout",
        "distribution_version": "0.2.0",
        "git_revision": "abc123",
        "git_dirty": False,
        "content_sha256": "1" * 64,
    }
    payload = _to_json(
        report,
        backend=BackendSpec(name="numpy"),
        config=config,
        einf_source=source_metadata,
    )
    json.dumps(payload)
    assert payload["schema_version"] == 6
    environment = payload["environment"]
    assert isinstance(environment, dict)
    assert {
        "python",
        "platform",
        "machine",
        "numpy",
        "torch",
        "einops",
        "einx",
        "einf",
    } <= set(environment)
    assert environment["einf"] == source_metadata
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
    assert [
        (unit["round_index"], unit["unit_index"])
        for unit in case_payload["realized_input_units"]
    ] == [(0, 0), (0, 1)]
    assert case_payload["validation"] == {
        "scope": "all_measured_executions",
        "coordinate_source": "realized_input_units",
        "coordinate_count": 2,
        "members": ["target", "equivalent", "lower_bound"],
        "executions_per_member_per_coordinate": 2,
        "phase": "immediately_after_each_timed_execution",
        "output_source": "timed_call_return_value",
        "output_check": "numerical_reference_match",
    }
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
