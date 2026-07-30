from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import benchmarks.harness.backend as backend_module
import benchmarks.harness.profiler as profiler_module
import benchmarks.harness.runner as runner_module
from benchmarks.harness import (
    Array,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicCaseSpec,
    DynamicRun,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    FixedCaseSpec,
    FixedTaskConfig,
    LibraryName,
    MarkdownPrinter,
    Output,
    PairedComparison,
    Profiler,
    TimingSummary,
)
from benchmarks.harness.result import TestResult as BenchmarkTestResult


def _summary(*, median_ms: float) -> TimingSummary:
    return TimingSummary(
        count=3,
        p25_ms=median_ms - 0.1,
        median_ms=median_ms,
        p75_ms=median_ms + 0.1,
        iqr_ms=0.2,
        p95_ms=median_ms + 0.2,
        mean_ms=median_ms,
        min_ms=median_ms - 0.2,
        max_ms=median_ms + 0.2,
    )


_UNIT_SIZES = BenchSizes(b=1, n=1, d=1, h=1, w=1, r=1, j=1)


def _vector_workload() -> DynamicShapeWorkload:
    return DynamicShapeWorkload(
        sampled_dimensions=(),
        input_shapes=lambda dimensions: ((1,),),
        output_shapes=lambda dimensions: ((1,),),
    )


def _recording_clock(
    *,
    events: list[str],
    values: tuple[float, ...],
) -> Callable[[], float]:
    samples = iter(values)
    call_index = 0

    def clock() -> float:
        nonlocal call_index
        phase = "clock_start" if call_index % 2 == 0 else "clock_stop"
        call_index += 1
        events.append(phase)
        return next(samples)

    return clock


def _single_library_runner(
    *,
    backend: BackendSpec,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={
            "einf": (True, "available"),
            "einops": (False, "not installed"),
            "einx": (False, "not installed"),
        },
    )


def _timing_case(*, events: list[str]) -> BenchmarkCase:
    def factory():
        events.append("factory")

        def run(inputs: tuple[Array, ...]) -> Output:
            events.append("call")
            return inputs[0]

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    return BenchmarkCase(
        name="timing_case",
        description="timing contract",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=factory,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )


def test_fixed_timing_stops_before_output_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    runner = _single_library_runner(backend=backend)
    monkeypatch.setattr(
        runner_module.time,
        "perf_counter",
        _recording_clock(
            events=events,
            values=(1.0, 1.001, 2.0, 2.002),
        ),
    )

    def record_touch(self: BackendSpec, output: Output) -> None:
        _ = self, output
        events.append("touch")

    monkeypatch.setattr(BackendSpec, "touch_output", record_touch)

    runner.run_fixed_case(
        case_spec=FixedCaseSpec(
            case=_timing_case(events=events),
            inputs=(np.asarray([1.0], dtype=np.float32),),
        ),
        config=FixedTaskConfig(
            backend="numpy",
            scale="small",
            seed=1,
            rounds=1,
            cold_repeats=1,
            warmup=0,
            warm_repeats=1,
            warm_iterations=1,
        ),
        order_seed=1,
    )

    assert events == [
        "clock_start",
        "factory",
        "call",
        "clock_stop",
        "touch",
        "factory",
        "clock_start",
        "call",
        "clock_stop",
        "touch",
    ]


def test_dynamic_timing_stops_before_output_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    runner = _single_library_runner(backend=backend)
    monkeypatch.setattr(
        runner_module.time,
        "perf_counter",
        _recording_clock(events=events, values=(1.0, 1.003)),
    )

    def record_touch(self: BackendSpec, output: Output) -> None:
        _ = self, output
        events.append("touch")

    monkeypatch.setattr(BackendSpec, "touch_output", record_touch)

    runner.run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=_timing_case(events=events),
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=DynamicTaskConfig(
            backend="numpy",
            scale="small",
            seed=1,
            batches=2,
            warmup_batches=1,
            repeats=1,
            rounds=1,
            round_order_seed=1,
            parity_checks=0,
        ),
        case_index=0,
    )

    assert events == [
        "factory",
        "call",
        "touch",
        "clock_start",
        "call",
        "clock_stop",
        "touch",
    ]


def test_profiler_dynamic_timing_stops_before_output_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        _recording_clock(events=events, values=(1.0, 1.004)),
    )

    def record_touch(self: BackendSpec, output: Output) -> None:
        _ = self, output
        events.append("touch")

    monkeypatch.setattr(BackendSpec, "touch_output", record_touch)

    def run(batch: tuple[Array, ...]) -> Output:
        events.append("call")
        return batch[0]

    Profiler(backend=backend).measure_dynamic(
        runner=run,
        batches=[
            (np.asarray([1.0], dtype=np.float32),),
            (np.asarray([2.0], dtype=np.float32),),
        ],
        warmup_batches=1,
        repeats=1,
    )

    assert events == [
        "call",
        "touch",
        "clock_start",
        "call",
        "clock_stop",
        "touch",
    ]


def test_eager_timing_rejects_asynchronous_torch_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    class FakeTensor:
        shape = (1,)
        device = FakeDevice()

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(Tensor=FakeTensor),
    )

    with pytest.raises(
        RuntimeError,
        match="eager benchmark timing requires synchronous CPU outputs",
    ):
        BackendSpec(name="torch").touch_output(cast(Output, FakeTensor()))


def test_round_orders_rotate_balanced_positions() -> None:
    runner = BenchmarkRunner(
        backend=BackendSpec(name="numpy"),
        profiler=Profiler(backend=BackendSpec(name="numpy")),
        available={
            "einf": (True, "available"),
            "einops": (True, "available"),
            "einx": (True, "available"),
        },
    )

    orders = runner._round_orders(
        library_names=["einf", "einops", "einx"],
        rounds=3,
        seed=1234,
    )

    assert len(orders) == 3
    for name in ("einf", "einops", "einx"):
        positions = {order.index(name) for order in orders}
        assert positions == {0, 1, 2}


def test_run_fixed_case_uses_paired_inputs_per_library() -> None:
    backend = BackendSpec(name="numpy")
    profiler = Profiler(backend=backend)
    runner = BenchmarkRunner(
        backend=backend,
        profiler=profiler,
        available={
            "einf": (True, "available"),
            "einops": (True, "available"),
            "einx": (True, "available"),
        },
    )

    original = np.asarray([1.0, 2.0], dtype=np.float32)
    captured: dict[str, list[np.ndarray]] = {
        "einf": [],
        "einops": [],
        "einx": [],
    }
    call_order: list[str] = []

    def _make_runner(name: str):
        def factory():
            def run(batch: tuple[Array, ...]) -> np.ndarray:
                array = batch[0]
                assert isinstance(array, np.ndarray)
                captured[name].append(array)
                call_order.append(name)
                return array

            return run

        return factory

    case = BenchmarkCase(
        name="fixed_case",
        description="demo",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=_make_runner("einf"),
        make_einops_runner=_make_runner("einops"),
        make_einx_runner=_make_runner("einx"),
    )

    result = runner.run_fixed_case(
        case_spec=FixedCaseSpec(case=case, inputs=(original,)),
        config=FixedTaskConfig(
            backend="numpy",
            scale="small",
            seed=1,
            rounds=1,
            cold_repeats=1,
            warmup=0,
            warm_repeats=1,
            warm_iterations=1,
        ),
        order_seed=1234,
    )

    order = result.round_orders[0]
    assert call_order == [order[0], order[1], order[2], order[0], order[1], order[2]]

    for lib_name in ("einf", "einops", "einx"):
        cold_array, warm_array = captured[lib_name]
        assert not np.shares_memory(cold_array, original)
        assert not np.shares_memory(warm_array, original)
        assert not np.shares_memory(cold_array, warm_array)
        assert np.array_equal(cold_array, original)
        assert np.array_equal(warm_array, original)


def test_run_dynamic_case_uses_same_batch_paired_order() -> None:
    backend = BackendSpec(name="numpy")
    profiler = Profiler(backend=backend)
    runner = BenchmarkRunner(
        backend=backend,
        profiler=profiler,
        available={
            "einf": (True, "available"),
            "einops": (True, "available"),
            "einx": (True, "available"),
        },
    )

    events: list[tuple[str, int, np.ndarray]] = []

    def _make_runner(name: str):
        def factory():
            def run(batch: tuple[Array, ...]) -> np.ndarray:
                array = batch[0]
                assert isinstance(array, np.ndarray)
                events.append((name, int(array[0]), array))
                return array

            return run

        return factory

    case = BenchmarkCase(
        name="dynamic_case",
        description="demo",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=_make_runner("einf"),
        make_einops_runner=_make_runner("einops"),
        make_einx_runner=_make_runner("einx"),
    )

    result = runner.run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=case,
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
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
        case_index=0,
    )

    order = result.round_orders[0]
    expected_orders = [
        order,
        order,
        runner._rotate_order(order, offset=1),
    ]
    assert len(events) == 9

    for batch_index, expected_order in enumerate(expected_orders):
        batch_events = events[batch_index * 3 : (batch_index + 1) * 3]
        assert [name for name, _, _ in batch_events] == list(expected_order)
        values = {value for _, value, _ in batch_events}
        assert len(values) == 1

        arrays = [array for _, _, array in batch_events]
        assert not np.shares_memory(arrays[0], arrays[1])
        assert not np.shares_memory(arrays[0], arrays[2])
        assert not np.shares_memory(arrays[1], arrays[2])

    assert len(result.observations) == 6
    measured_orders = expected_orders[1:]
    for measured_batch_index, expected_order in enumerate(measured_orders):
        start = measured_batch_index * 3
        batch_observations = result.observations[start : start + 3]
        assert [item.library for item in batch_observations] == list(expected_order)
        assert [item.order_position for item in batch_observations] == [0, 1, 2]
        assert {item.measured_batch_index for item in batch_observations} == {
            measured_batch_index
        }
        assert {item.repeat_index for item in batch_observations} == {0}
        assert {item.round_index for item in batch_observations} == {0}


def test_run_dynamic_case_preserves_latency_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BackendSpec(name="numpy")
    runner = BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={
            "einf": (True, "available"),
            "einops": (True, "available"),
            "einx": (True, "available"),
        },
    )
    clock_values: list[float] = []
    for call_index in range(12):
        started = float(call_index)
        clock_values.extend((started, started + (call_index + 1) / 1000.0))
    clock_iterator = iter(clock_values)
    monkeypatch.setattr(
        runner_module.time,
        "perf_counter",
        lambda: next(clock_iterator),
    )

    result = runner.run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=_timing_case(events=[]),
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=DynamicTaskConfig(
            backend="numpy",
            scale="medium",
            seed=7,
            batches=2,
            warmup_batches=0,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
            parity_checks=0,
        ),
        case_index=0,
    )

    assert [item.latency_ms for item in result.observations] == pytest.approx(
        list(range(1, 13))
    )


def test_run_dynamic_case_rejects_partial_round_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def partial_round_orders(
        self: BenchmarkRunner,
        *,
        library_names: list[LibraryName],
        rounds: int,
        seed: int,
    ) -> list[tuple[LibraryName, ...]]:
        _ = self, library_names, rounds, seed
        return [("einf",)]

    monkeypatch.setattr(BenchmarkRunner, "_round_orders", partial_round_orders)
    backend = BackendSpec(name="numpy")
    runner = BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={
            "einf": (True, "available"),
            "einops": (True, "available"),
            "einx": (False, "not installed"),
        },
    )

    with pytest.raises(RuntimeError, match="full library permutations"):
        runner.run_dynamic_case(
            case_spec=DynamicCaseSpec(
                case=_timing_case(events=[]),
                sizes=_UNIT_SIZES,
                workload=_vector_workload(),
            ),
            config=DynamicTaskConfig(
                backend="numpy",
                scale="medium",
                seed=7,
                batches=2,
                warmup_batches=0,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
                parity_checks=0,
            ),
            case_index=0,
        )


def test_markdown_printer_renders_round_level_summaries() -> None:
    case = BenchmarkCase(
        name="dynamic_case",
        description="demo",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=lambda: lambda inputs: inputs[0],
        make_einops_runner=lambda: lambda inputs: inputs[0],
        make_einx_runner=lambda: lambda inputs: inputs[0],
    )
    workload = _vector_workload().metadata(sizes=_UNIT_SIZES)
    workload_comparison = workload.compare_to(
        workload,
        scale="medium",
        reference_scale="medium",
    )
    case_result = DynamicCaseResult(
        case=case,
        workload=workload,
        runs={
            "einf": DynamicRun(
                summary=_summary(median_ms=1.0),
                round_summaries=(
                    _summary(median_ms=1.0),
                    _summary(median_ms=1.1),
                ),
            ),
            "einops": DynamicRun(
                summary=_summary(median_ms=2.0),
                round_summaries=(
                    _summary(median_ms=2.0),
                    _summary(median_ms=2.1),
                ),
            ),
            "einx": DynamicRun(
                summary=_summary(median_ms=3.0),
                round_summaries=(
                    _summary(median_ms=3.0),
                    _summary(median_ms=3.1),
                ),
            ),
        },
        round_orders=[("einf", "einops", "einx"), ("einops", "einx", "einf")],
        observations=(),
        comparisons=(
            PairedComparison(
                baseline="einf",
                competitor="einops",
                call_pair_count=12,
                paired_batch_count=6,
                latency_ratio=2.0,
                confidence_level=0.95,
                confidence_interval_low=1.8,
                confidence_interval_high=2.2,
                bootstrap_resamples=10_000,
                bootstrap_seed=7,
            ),
        ),
    )
    report = BenchmarkTestResult(
        title="# Demo",
        configuration=["backend: `numpy`"],
        methodology=["demo"],
        case_results=[case_result],
        notes=["demo"],
    )

    markdown = MarkdownPrinter().render_dynamic(
        report,
        workload_comparisons={"dynamic_case": workload_comparison},
    )

    assert "Round base order (paired execution rotates within each round):" in markdown
    assert "Round median summaries (ms):" in markdown
    assert (
        "| einops / einf | 12 | 6 | 2.0000 | 95% | [1.8000, 2.2000] | +100.00% |"
        in markdown
    )
    assert "einf=1.0000" in markdown
    assert "einops=2.1000" in markdown
    assert "Total base input elements: `1`; ratio `1` (1.000x)" in markdown
