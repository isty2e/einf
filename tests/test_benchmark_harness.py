import numpy as np

from benchmarks.harness import (
    Array,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    CaseCalls,
    CaseResult,
    DynamicCaseSpec,
    DynamicTaskConfig,
    FixedCaseSpec,
    FixedTaskConfig,
    LibraryRun,
    MarkdownPrinter,
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
            batch_factory=lambda generator: generator.backend_batch(
                (
                    np.asarray(
                        [generator.random_state.randint(0, 1000)],
                        dtype=np.float32,
                    ),
                )
            ),
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
    case_result = CaseResult(
        case=case,
        runs={
            "einf": LibraryRun(
                available=True,
                reason="available",
                cold=None,
                warm=None,
                dynamic=_summary(median_ms=1.0),
                dynamic_rounds=(_summary(median_ms=1.0), _summary(median_ms=1.1)),
            ),
            "einops": LibraryRun(
                available=True,
                reason="available",
                cold=None,
                warm=None,
                dynamic=_summary(median_ms=2.0),
                dynamic_rounds=(_summary(median_ms=2.0), _summary(median_ms=2.1)),
            ),
            "einx": LibraryRun(
                available=True,
                reason="available",
                cold=None,
                warm=None,
                dynamic=_summary(median_ms=3.0),
                dynamic_rounds=(_summary(median_ms=3.0), _summary(median_ms=3.1)),
            ),
        },
        round_orders=[("einf", "einops", "einx"), ("einops", "einx", "einf")],
    )
    report = BenchmarkTestResult(
        title="# Demo",
        configuration=["backend: `numpy`"],
        methodology=["demo"],
        case_results=[case_result],
        notes=["demo"],
    )

    markdown = MarkdownPrinter().render_dynamic(report)

    assert "Round base order (paired execution rotates within each round):" in markdown
    assert "Round median summaries (ms):" in markdown
    assert "einf=1.0000" in markdown
    assert "einops=2.1000" in markdown
