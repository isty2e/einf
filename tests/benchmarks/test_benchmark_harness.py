from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from weakref import ReferenceType, ref

import numpy as np
import pytest

import benchmarks.harness.backend as backend_module
import benchmarks.harness.profiler as profiler_module
from benchmarks.harness import (
    Array,
    AvailableRun,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicCaseSpec,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    FixedCaseSpec,
    FixedTaskConfig,
    LibraryName,
    MarkdownPrinter,
    Output,
    PairedComparison,
    PairedEvidence,
    Profiler,
    TimingSummary,
)
from benchmarks.harness.generator import derive_coordinate_seed
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


def test_fixed_runner_validates_timed_output_after_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    runner = _single_library_runner(backend=backend)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        _recording_clock(
            events=events,
            values=(1.0, 1.002),
        ),
    )

    result = runner.run_fixed_case(
        case_spec=FixedCaseSpec(
            case=_timing_case(events=events),
            inputs=(np.asarray([1.0], dtype=np.float32),),
        ),
        config=FixedTaskConfig(
            scale="small",
            seed=1,
            rounds=1,
            warmup=0,
            repeats=1,
            iterations=1,
        ),
        order_seed=1,
    )

    assert events == [
        "factory",
        "clock_start",
        "call",
        "clock_stop",
    ]
    assert [item.latency_ms for item in result.evidence.observations] == [
        pytest.approx(2.0)
    ]


def test_fixed_runner_rejects_wrong_timed_output_after_correct_warmup() -> None:
    call_count = 0

    def make_runner() -> Callable[[tuple[Array, ...]], Output]:
        def run(inputs: tuple[Array, ...]) -> Output:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return np.zeros_like(inputs[0])
            return inputs[0]

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="timed_wrong_result",
        description="Returns an incorrect result only for the timed call.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=make_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match="value mismatch"):
        _single_library_runner(backend=backend).run_fixed_case(
            case_spec=FixedCaseSpec(
                case=case,
                inputs=(np.asarray([1.0], dtype=np.float32),),
            ),
            config=FixedTaskConfig(
                scale="small",
                seed=1,
                rounds=1,
                warmup=1,
                repeats=1,
                iterations=1,
            ),
            order_seed=1,
        )


def test_fixed_runner_rejects_input_mutation_during_warmup() -> None:
    backend = BackendSpec(name="numpy")
    inputs = (np.asarray([1.0], dtype=np.float32),)

    def run(batch: tuple[Array, ...]) -> Output:
        array = batch[0]
        assert isinstance(array, np.ndarray)
        array.fill(0.0)
        return array

    with pytest.raises(ValueError, match="read-only"):
        _single_library_runner(backend=backend)._warmup_coordinate(
            runners={"einf": run},
            batch=inputs,
            order=("einf",),
        )

    assert inputs[0].flags.writeable


def test_fixed_runner_isolates_reference_from_measured_inputs() -> None:
    measured_inputs: list[np.ndarray] = []

    def reference(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        array = inputs[0]
        expected = array.copy()
        array.resize((3,), refcheck=False)
        array[:] = (10.0, 20.0, 30.0)
        return expected

    def make_runner() -> Callable[[tuple[Array, ...]], Output]:
        def run(inputs: tuple[Array, ...]) -> Output:
            array = inputs[0]
            assert isinstance(array, np.ndarray)
            measured_inputs.append(array.copy())
            return array

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="mutating_fixed_reference",
        description="Reference mutation must not change the measured input.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=reference,
        make_einf_runner=make_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )
    inputs = (np.asarray([1.0, 2.0], dtype=np.float32),)

    _single_library_runner(backend=BackendSpec(name="numpy")).run_fixed_case(
        case_spec=FixedCaseSpec(case=case, inputs=inputs),
        config=FixedTaskConfig(
            scale="small",
            seed=1,
            rounds=1,
            warmup=0,
            repeats=1,
            iterations=1,
        ),
        order_seed=1,
    )

    np.testing.assert_array_equal(inputs[0], np.asarray([1.0, 2.0]))
    assert [array.shape for array in measured_inputs] == [(2,)]
    np.testing.assert_array_equal(measured_inputs[0], np.asarray([1.0, 2.0]))


def test_dynamic_runner_warms_before_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    runner = _single_library_runner(backend=backend)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        _recording_clock(events=events, values=(1.0, 1.003)),
    )

    runner.run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=_timing_case(events=events),
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=DynamicTaskConfig(
            scale="small",
            seed=1,
            batches=2,
            warmup_batches=1,
            repeats=1,
            rounds=1,
            round_order_seed=1,
        ),
        case_index=0,
    )

    assert events == [
        "factory",
        "call",
        "clock_start",
        "call",
        "clock_stop",
    ]


def test_dynamic_runner_validates_every_measured_coordinate() -> None:
    factory_count = 0

    def make_stateful_runner() -> Callable[[tuple[Array, ...]], Output]:
        nonlocal factory_count
        factory_count += 1
        call_count = 0

        def run(inputs: tuple[Array, ...]) -> Output:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return inputs[0]
            return np.zeros_like(inputs[0])

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="late_wrong_result",
        description="Returns an incorrect result after the first coordinate.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=make_stateful_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match="value mismatch"):
        _single_library_runner(backend=backend).run_dynamic_case(
            case_spec=DynamicCaseSpec(
                case=case,
                sizes=_UNIT_SIZES,
                workload=_vector_workload(),
            ),
            config=_dynamic_config(),
            case_index=0,
        )

    assert factory_count == 1


def test_dynamic_runner_rejects_shape_first_timed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape_by_seed: dict[int, int] = {}

    def generate_batch(
        self: BenchmarkRunner,
        *,
        case_spec: DynamicCaseSpec,
        seed: int,
    ) -> tuple[np.ndarray, ...]:
        _ = self, case_spec
        size = shape_by_seed.setdefault(seed, len(shape_by_seed) + 1)
        return (np.arange(1, size + 1, dtype=np.float32),)

    monkeypatch.setattr(
        BenchmarkRunner,
        "_make_dynamic_numpy_batch",
        generate_batch,
    )
    seen_shapes: set[tuple[int, ...]] = set()

    def make_shape_cached_runner() -> Callable[[tuple[Array, ...]], Output]:
        def run(inputs: tuple[Array, ...]) -> Output:
            shape = tuple(int(dimension) for dimension in inputs[0].shape)
            if shape not in seen_shapes:
                seen_shapes.add(shape)
                return np.zeros_like(inputs[0])
            return inputs[0]

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="shape_first_wrong_result",
        description="Returns an incorrect result on the first call for each shape.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=make_shape_cached_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match="value mismatch"):
        _single_library_runner(backend=backend).run_dynamic_case(
            case_spec=DynamicCaseSpec(
                case=case,
                sizes=_UNIT_SIZES,
                workload=_vector_workload(),
            ),
            config=_dynamic_config(),
            case_index=0,
        )


def test_dynamic_runner_rejects_repeat_specific_timed_output() -> None:
    call_count = 0

    def make_repeat_sensitive_runner() -> Callable[[tuple[Array, ...]], Output]:
        def run(inputs: tuple[Array, ...]) -> Output:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return np.zeros_like(inputs[0])
            return inputs[0]

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="repeat_specific_wrong_result",
        description="Returns an incorrect result in one measured repeat.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=make_repeat_sensitive_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match="value mismatch"):
        _single_library_runner(backend=backend).run_dynamic_case(
            case_spec=DynamicCaseSpec(
                case=case,
                sizes=_UNIT_SIZES,
                workload=_vector_workload(),
            ),
            config=_dynamic_config(repeats=2),
            case_index=0,
        )


def test_dynamic_runner_rejects_input_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_count = 0

    def generate_batch(
        self: BenchmarkRunner,
        *,
        case_spec: DynamicCaseSpec,
        seed: int,
    ) -> tuple[np.ndarray, ...]:
        nonlocal generation_count
        _ = self, case_spec, seed
        generation_count += 1
        size = 1 if generation_count <= 2 else 2
        return (np.ones((size,), dtype=np.float32),)

    monkeypatch.setattr(
        BenchmarkRunner,
        "_make_dynamic_numpy_batch",
        generate_batch,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(RuntimeError, match="repeated input produced different shapes"):
        _single_library_runner(backend=backend).run_dynamic_case(
            case_spec=DynamicCaseSpec(
                case=_timing_case(events=[]),
                sizes=_UNIT_SIZES,
                workload=_vector_workload(),
            ),
            config=_dynamic_config(repeats=2),
            case_index=0,
        )


def test_dynamic_runner_isolates_reference_from_measured_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured_inputs: list[np.ndarray] = []

    def make_dynamic_numpy_batch(
        self: BenchmarkRunner,
        *,
        case_spec: DynamicCaseSpec,
        seed: int,
    ) -> tuple[np.ndarray, ...]:
        _ = self, case_spec, seed
        return (np.asarray([1.0, 2.0], dtype=np.float32),)

    monkeypatch.setattr(
        BenchmarkRunner,
        "_make_dynamic_numpy_batch",
        make_dynamic_numpy_batch,
    )

    def reference(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        array = inputs[0]
        expected = array.copy()
        array.resize((3,), refcheck=False)
        array[:] = (10.0, 20.0, 30.0)
        return expected

    def make_runner() -> Callable[[tuple[Array, ...]], Output]:
        def run(inputs: tuple[Array, ...]) -> Output:
            array = inputs[0]
            assert isinstance(array, np.ndarray)
            measured_inputs.append(array.copy())
            return array

        return run

    unavailable_factory = lambda: lambda inputs: inputs[0]
    case = BenchmarkCase(
        name="mutating_dynamic_reference",
        description="Reference mutation must not change the measured input.",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=reference,
        make_einf_runner=make_runner,
        make_einops_runner=unavailable_factory,
        make_einx_runner=unavailable_factory,
    )

    result = _single_library_runner(backend=BackendSpec(name="numpy")).run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=case,
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=_dynamic_config(),
        case_index=0,
    )

    assert [unit.input_shapes for unit in result.realized_units] == [
        ((2,),),
        ((2,),),
    ]
    assert [array.shape for array in measured_inputs] == [(2,), (2,)]
    for array in measured_inputs:
        np.testing.assert_array_equal(array, np.asarray([1.0, 2.0]))


def test_dynamic_runner_validates_every_measured_execution() -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")

    result = _single_library_runner(backend=backend).run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=_timing_case(events=events),
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=_dynamic_config(
            batches=3,
            warmup_batches=1,
            repeats=3,
            rounds=2,
        ),
        case_index=0,
    )

    assert events == ["factory", *("call" for _ in range(14))]
    assert len(result.evidence.observations) == 12
    assert [(unit.round_index, unit.unit_index) for unit in result.realized_units] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_profiler_measure_call_synchronizes_around_timed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = BackendSpec(name="numpy")
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        _recording_clock(events=events, values=(1.0, 1.004)),
    )

    def record_synchronize(self: BackendSpec) -> None:
        _ = self
        events.append("sync")

    monkeypatch.setattr(BackendSpec, "synchronize", record_synchronize)

    def run(batch: tuple[Array, ...]) -> Output:
        events.append("call")
        return batch[0]

    input_array = np.asarray([1.0], dtype=np.float32)

    def validate_output(output: Output) -> None:
        assert output is input_array
        events.append("validate")

    elapsed_ms = Profiler(backend=backend).measure_call(
        runner=run,
        batch=(input_array,),
        validate_output=validate_output,
    )

    assert events == [
        "sync",
        "clock_start",
        "call",
        "sync",
        "clock_stop",
        "validate",
    ]
    assert elapsed_ms == pytest.approx(4.0)


def test_numpy_profiler_detects_runner_that_removes_input_protection() -> None:
    input_array = np.asarray([1.0], dtype=np.float32)

    def run(batch: tuple[Array, ...]) -> Output:
        array = batch[0]
        assert isinstance(array, np.ndarray)
        array.setflags(write=True)
        array.fill(0.0)
        return array

    with pytest.raises(RuntimeError, match="removed input write protection"):
        Profiler(backend=BackendSpec(name="numpy")).measure_call(
            runner=run,
            batch=(input_array,),
            validate_output=lambda _: None,
        )

    assert input_array.flags.writeable


def test_numpy_profiler_restores_duplicate_input_writeability() -> None:
    input_array = np.asarray([1.0], dtype=np.float32)

    Profiler(backend=BackendSpec(name="numpy")).measure_call(
        runner=lambda batch: batch[0],
        batch=(input_array, input_array),
        validate_output=lambda _: None,
    )

    assert input_array.flags.writeable


def test_numpy_input_guard_restores_state_after_partial_setup_failure() -> None:
    input_array = np.asarray([1.0], dtype=np.float32)
    invalid_input = cast(Array, SimpleNamespace())

    with (
        pytest.raises(TypeError, match="unsupported type at index 1"),
        BackendSpec(name="numpy").preserve_input_batch((input_array, invalid_input)),
    ):
        pass

    assert input_array.flags.writeable


def test_torch_profiler_detects_in_place_input_mutation() -> None:
    if backend_module.torch is None:
        pytest.skip("torch is not installed")

    backend = BackendSpec(name="torch")
    input_tensor = backend_module.torch.tensor(
        [1.0], dtype=backend_module.torch.float32
    )

    def run(batch: tuple[Array, ...]) -> Output:
        value = batch[0]
        assert not isinstance(value, np.ndarray)
        value.add_(1.0)
        return value

    with pytest.raises(RuntimeError, match="mutated input at index 0"):
        Profiler(backend=backend).measure_call(
            runner=run,
            batch=(input_tensor,),
            validate_output=lambda _: None,
        )


def test_torch_profiler_detects_mutation_without_real_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    class FakeTensor:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device
            self._version = 0

        def add_(self, value: float) -> "FakeTensor":
            _ = value
            self._version += 1
            return self

    resolved_device = FakeDevice()
    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=FakeTensor,
            cpu=SimpleNamespace(synchronize=lambda device: None),
            device=lambda label: resolved_device,
            empty=lambda size, *, device: FakeTensor(device),
        ),
    )
    backend = BackendSpec(name="torch")
    input_tensor = FakeTensor(resolved_device)

    def run(batch: tuple[Array, ...]) -> Output:
        value = cast(FakeTensor, batch[0])
        return cast(Output, value.add_(1.0))

    with pytest.raises(RuntimeError, match="mutated input at index 0"):
        Profiler(backend=backend).measure_call(
            runner=run,
            batch=cast(tuple[Array, ...], (input_tensor,)),
            validate_output=lambda _: None,
        )


def test_profiler_releases_timed_output_after_validation() -> None:
    output_reference: ReferenceType[np.ndarray] | None = None

    def run(batch: tuple[Array, ...]) -> Output:
        nonlocal output_reference
        array = batch[0]
        assert isinstance(array, np.ndarray)
        output = array.copy()
        output_reference = ref(output)
        return output

    Profiler(backend=BackendSpec(name="numpy")).measure_call(
        runner=run,
        batch=(np.asarray([1.0], dtype=np.float32),),
        validate_output=lambda _: None,
    )

    assert output_reference is not None
    assert output_reference() is None


def test_torch_backend_resolves_open_accelerator_device_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, label: str) -> None:
            self.label = label
            self.type = label.partition(":")[0]

        def __str__(self) -> str:
            return self.label

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and self.label == other.label

    class FakeTensor:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device

    synchronizations: list[str] = []

    def synchronize(device: FakeDevice) -> None:
        synchronizations.append(str(device))

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=FakeTensor,
            accelerator=SimpleNamespace(synchronize=synchronize),
            cpu=SimpleNamespace(synchronize=synchronize),
            device=FakeDevice,
            empty=lambda size, *, device: FakeTensor(device),
        ),
    )

    backend = BackendSpec(name="torch", requested_device="privateuseone:3")

    assert backend.resolved_device == "privateuseone:3"
    backend.synchronize()
    assert synchronizations == ["privateuseone:3", "privateuseone:3"]


def test_torch_backend_falls_back_to_no_argument_device_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "mps"

        def __str__(self) -> str:
            return "mps:0"

    synchronizations: list[str] = []
    resolved_device = FakeDevice()
    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            device=lambda label: resolved_device,
            empty=lambda size, *, device: SimpleNamespace(device=device),
            mps=SimpleNamespace(
                synchronize=lambda: synchronizations.append("mps:0"),
            ),
        ),
    )

    backend = BackendSpec(name="torch", requested_device="mps")
    backend.synchronize()

    assert synchronizations == ["mps:0", "mps:0"]


def test_torch_backend_imports_unexported_device_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "mps"

        def __str__(self) -> str:
            return "mps:0"

    synchronizations: list[str] = []
    resolved_device = FakeDevice()
    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            device=lambda label: resolved_device,
            empty=lambda size, *, device: SimpleNamespace(device=device),
        ),
    )

    def import_device_module(name: str) -> SimpleNamespace:
        assert name == "torch.mps"
        return SimpleNamespace(
            synchronize=lambda: synchronizations.append("mps:0"),
        )

    monkeypatch.setattr(backend_module, "import_module", import_device_module)

    backend = BackendSpec(name="torch", requested_device="mps")
    backend.synchronize()

    assert synchronizations == ["mps:0", "mps:0"]


def test_torch_backend_falls_back_to_device_argument_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:2"

    synchronizations: list[str] = []
    resolved_device = FakeDevice()

    def synchronize(device: FakeDevice) -> None:
        synchronizations.append(str(device))

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            device=lambda label: resolved_device,
            empty=lambda size, *, device: SimpleNamespace(device=device),
            cuda=SimpleNamespace(synchronize=synchronize),
        ),
    )

    backend = BackendSpec(name="torch", requested_device="cuda:2")
    backend.synchronize()

    assert synchronizations == ["cuda:2", "cuda:2"]


def test_torch_cpu_backend_does_not_require_synchronization_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    resolved_device = FakeDevice()
    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            device=lambda label: resolved_device,
            empty=lambda size, *, device: SimpleNamespace(device=device),
        ),
    )

    BackendSpec(name="torch").synchronize()


def test_torch_backend_rejects_target_without_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "mps"

        def __str__(self) -> str:
            return "mps"

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            accelerator=SimpleNamespace(),
            cpu=SimpleNamespace(),
            device=lambda label: FakeDevice(),
            empty=lambda size, *, device: SimpleNamespace(device=device),
        ),
    )

    def reject_device_module(name: str) -> None:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(backend_module, "import_module", reject_device_module)

    with pytest.raises(TypeError, match="has no synchronization capability"):
        BackendSpec(name="torch", requested_device="mps")


def test_torch_backend_rejects_unavailable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_target(size: int, *, device: str) -> None:
        _ = size, device
        raise RuntimeError("target unavailable")

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=object,
            device=lambda label: label,
            empty=reject_target,
        ),
    )

    with pytest.raises(RuntimeError, match="'xpu:7' is unavailable"):
        BackendSpec(name="torch", requested_device="xpu:7")


def test_numpy_backend_rejects_non_cpu_target() -> None:
    with pytest.raises(ValueError, match="only supports the cpu device"):
        BackendSpec(name="numpy", requested_device="mps")


def test_numpy_backend_adopts_fresh_batch_without_copying() -> None:
    array = np.asarray([1.0], dtype=np.float32)

    batch = BackendSpec(name="numpy").to_backend_batch((array,))

    assert batch[0] is array


def test_torch_backend_materializes_fresh_batch_without_host_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        type = "mps"

        def __str__(self) -> str:
            return "mps"

    class FakeTensor:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device

        def to(self, *, device: FakeDevice) -> "FakeTensor":
            self.device = device
            return self

    resolved_device = FakeDevice()
    source_arrays: list[np.ndarray] = []

    def from_numpy(array: np.ndarray) -> FakeTensor:
        source_arrays.append(array)
        return FakeTensor(resolved_device)

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=FakeTensor,
            accelerator=SimpleNamespace(synchronize=lambda device: None),
            device=lambda label: resolved_device,
            empty=lambda size, *, device: FakeTensor(device),
            from_numpy=from_numpy,
        ),
    )
    backend = BackendSpec(name="torch", requested_device="mps")
    array = np.asarray([1.0], dtype=np.float32)

    backend.to_backend_batch((array,))

    assert len(source_arrays) == 1
    assert source_arrays[0] is array


def test_backend_rejects_output_on_different_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, label: str) -> None:
            self.label = label
            self.type = label

        def __str__(self) -> str:
            return self.label

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and self.label == other.label

    class FakeTensor:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device

    monkeypatch.setattr(
        backend_module,
        "torch",
        SimpleNamespace(
            Tensor=FakeTensor,
            accelerator=SimpleNamespace(synchronize=lambda device: None),
            cpu=SimpleNamespace(synchronize=lambda device: None),
            device=FakeDevice,
            empty=lambda size, *, device: FakeTensor(device),
        ),
    )
    backend = BackendSpec(name="torch", requested_device="mps")

    with pytest.raises(RuntimeError, match="expected mps, got cpu"):
        backend.validate_output_target(cast(Output, FakeTensor(FakeDevice("cpu"))))


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
            scale="small",
            seed=1,
            rounds=1,
            warmup=0,
            repeats=2,
            iterations=2,
        ),
        order_seed=1234,
    )

    order = result.round_orders[0]
    expected_orders = [
        order,
        runner._rotate_order(order, offset=1),
        runner._rotate_order(order, offset=2),
        order,
    ]
    assert call_order == [
        name for expected_order in expected_orders for name in expected_order
    ]

    for lib_name in ("einf", "einops", "einx"):
        arrays = captured[lib_name]
        assert len(arrays) == 4
        assert all(array is original for array in arrays)

    assert len(result.evidence.observations) == 12
    assert len(result.evidence.comparisons) == 2

    for timing_index, expected_order in enumerate(expected_orders):
        start = timing_index * 3
        observations = result.evidence.observations[start : start + 3]
        assert tuple(item.library for item in observations) == expected_order
        assert {item.unit_index for item in observations} == {timing_index // 2}
        assert {item.repeat_index for item in observations} == {timing_index % 2}
        assert tuple(item.order_position for item in observations) == (0, 1, 2)

    report = BenchmarkTestResult(
        title="# Fixed",
        configuration=["backend: `numpy`"],
        methodology=["paired"],
        case_results=[result],
        notes=[],
    )
    markdown = MarkdownPrinter().render_fixed(report)
    assert "Paired steady latency ratios (competitor / einf):" in markdown
    assert "Paired timing units" in markdown


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
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
        ),
        case_index=0,
    )

    order = result.round_orders[0]
    expected_orders = [
        order,
        order,
        runner._rotate_order(order, offset=1),
        runner._rotate_order(order, offset=2),
        runner._rotate_order(order, offset=3),
    ]
    assert len(events) == 15

    for batch_index, expected_order in enumerate(expected_orders):
        batch_events = events[batch_index * 3 : (batch_index + 1) * 3]
        assert [name for name, _, _ in batch_events] == list(expected_order)
        values = {value for _, value, _ in batch_events}
        assert len(values) == 1

        arrays = [array for _, _, array in batch_events]
        assert arrays[0] is arrays[1]
        assert arrays[0] is arrays[2]

    first_repeat_events = events[3:9]
    second_repeat_events = events[9:15]
    for unit_index in range(2):
        first_array = first_repeat_events[unit_index * 3][2]
        second_array = second_repeat_events[unit_index * 3][2]
        assert first_array is not second_array
        np.testing.assert_array_equal(first_array, second_array)

    assert len(result.realized_units) == 2
    assert [unit.stream_index for unit in result.realized_units] == [1, 2]
    assert [unit.seed for unit in result.realized_units] == [
        derive_coordinate_seed(
            seed=7,
            case_index=0,
            round_index=0,
            stream_index=stream_index,
        )
        for stream_index in (1, 2)
    ]
    assert [unit.input_shapes for unit in result.realized_units] == [
        ((1,),),
        ((1,),),
    ]

    assert len(result.evidence.observations) == 12
    measured_orders = expected_orders[1:5]
    for coordinate_index, expected_order in enumerate(measured_orders):
        start = coordinate_index * 3
        batch_observations = result.evidence.observations[start : start + 3]
        assert [item.library for item in batch_observations] == list(expected_order)
        assert [item.order_position for item in batch_observations] == [0, 1, 2]
        assert {item.unit_index for item in batch_observations} == {
            coordinate_index % 2
        }
        assert {item.repeat_index for item in batch_observations} == {
            coordinate_index // 2
        }
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
        profiler_module.time,
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
            scale="medium",
            seed=7,
            batches=2,
            warmup_batches=0,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
        ),
        case_index=0,
    )

    assert [item.latency_ms for item in result.evidence.observations] == pytest.approx(
        list(range(1, 13))
    )


def test_run_dynamic_case_releases_prepared_batch_between_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_references: list[ReferenceType[np.ndarray]] = []
    prepared_references: list[ReferenceType[np.ndarray]] = []
    live_host_batches_before_generation: list[int] = []
    live_batches_before_preparation: list[int] = []
    make_dynamic_numpy_batch = BenchmarkRunner._make_dynamic_numpy_batch

    def generate_batch(
        self: BenchmarkRunner,
        *,
        case_spec: DynamicCaseSpec,
        seed: int,
    ) -> tuple[np.ndarray, ...]:
        live_host_batches_before_generation.append(
            sum(reference() is not None for reference in host_references)
        )
        batch = make_dynamic_numpy_batch(self, case_spec=case_spec, seed=seed)
        host_references.extend(ref(array) for array in batch)
        return batch

    def prepare_batch(
        self: BackendSpec,
        batch: tuple[np.ndarray, ...],
    ) -> tuple[Array, ...]:
        _ = self
        live_batches_before_preparation.append(
            sum(reference() is not None for reference in prepared_references)
        )
        prepared_references.extend(ref(array) for array in batch)
        return batch

    monkeypatch.setattr(
        BenchmarkRunner,
        "_make_dynamic_numpy_batch",
        generate_batch,
    )
    monkeypatch.setattr(BackendSpec, "to_backend_batch", prepare_batch)
    backend = BackendSpec(name="numpy")
    runner = _single_library_runner(backend=backend)

    runner.run_dynamic_case(
        case_spec=DynamicCaseSpec(
            case=_timing_case(events=[]),
            sizes=_UNIT_SIZES,
            workload=_vector_workload(),
        ),
        config=DynamicTaskConfig(
            scale="medium",
            seed=7,
            batches=3,
            warmup_batches=1,
            repeats=2,
            rounds=1,
            round_order_seed=1234,
        ),
        case_index=0,
    )

    assert live_host_batches_before_generation == [0, 0, 0, 0, 0]
    assert live_batches_before_preparation == [0, 0, 0, 0, 0]


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
                scale="medium",
                seed=7,
                batches=2,
                warmup_batches=0,
                repeats=1,
                rounds=1,
                round_order_seed=1234,
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
        realized_units=(),
        runs={
            "einf": AvailableRun(
                summary=_summary(median_ms=1.0),
                round_summaries=(
                    _summary(median_ms=1.0),
                    _summary(median_ms=1.1),
                ),
            ),
            "einops": AvailableRun(
                summary=_summary(median_ms=2.0),
                round_summaries=(
                    _summary(median_ms=2.0),
                    _summary(median_ms=2.1),
                ),
            ),
            "einx": AvailableRun(
                summary=_summary(median_ms=3.0),
                round_summaries=(
                    _summary(median_ms=3.0),
                    _summary(median_ms=3.1),
                ),
            ),
        },
        round_orders=[("einf", "einops", "einx"), ("einops", "einx", "einf")],
        evidence=PairedEvidence(
            observations=(),
            comparisons=(
                PairedComparison(
                    baseline="einf",
                    competitor="einops",
                    call_pair_count=12,
                    paired_unit_count=6,
                    latency_ratio=2.0,
                    confidence_level=0.95,
                    confidence_interval_low=1.8,
                    confidence_interval_high=2.2,
                    bootstrap_resamples=10_000,
                    bootstrap_seed=7,
                ),
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
