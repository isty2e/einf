from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from typing import Literal, cast
from weakref import ReferenceType, ref

import pytest

pytest.importorskip("numpy")

import numpy as np

import benchmarks.compare.expression_parity as expression_module
import benchmarks.harness.input_identity as identity_module
import benchmarks.harness.profiler as profiler_module
import benchmarks.harness.runner as runner_module
from benchmarks.compare.expression_parity import (
    ExpressionCaseResult,
    ExpressionParityCase,
    ExpressionRunnerSpec,
)
from benchmarks.harness import (
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicCaseSpec,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    Profiler,
    TensorGenerator,
)
from benchmarks.harness.input_identity import numpy_batch_fingerprint
from benchmarks.harness.types import Array, BackendName, NumpyArray, Output, Runner

_Family = Literal["libraries", "expression"]
_Factory = Callable[[TensorGenerator], tuple[NumpyArray, ...]]
_CONFIG = DynamicTaskConfig(
    scale="medium",
    seed=7,
    batches=3,
    warmup_batches=1,
    repeats=3,
    rounds=2,
    round_order_seed=1234,
)


@pytest.fixture(autouse=True)
def synthetic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    tick = 0

    def clock() -> float:
        nonlocal tick
        tick += 1
        return tick / 1000

    monkeypatch.setattr(profiler_module.time, "perf_counter", clock)


def _run_case(
    *,
    family: _Family,
    factory: _Factory,
    events: list[str],
    config: DynamicTaskConfig = _CONFIG,
    backend_name: BackendName = "numpy",
) -> DynamicCaseResult | ExpressionCaseResult:
    backend = BackendSpec(name=backend_name)

    def reference(inputs: tuple[NumpyArray, ...]) -> NumpyArray:
        events.append("reference")
        return inputs[0]

    def make_runner() -> Runner:
        def run(inputs: tuple[Array, ...]) -> Output:
            events.append("run")
            return inputs[0]

        return run

    if family == "expression":
        case = ExpressionParityCase(
            name="repeat_identity",
            description="Deterministic repetition of each generated input.",
            target_name="einf",
            batch_factory=factory,
            runner_specs=tuple(
                ExpressionRunnerSpec(
                    name=name,
                    semantics="output_equivalent",
                    description="Identity on the shared input.",
                    call_repr="identity(x)",
                    available=True,
                    reason="available",
                    reference=reference,
                    make_runner=make_runner,
                )
                for name in ("einf", "einops", "einx")
            ),
        )
        return expression_module._run_dynamic_case(
            case=case,
            config=config,
            profiler=Profiler(backend=backend),
            backend=backend,
            case_index=0,
        )

    class FactoryCase(DynamicCaseSpec):
        def make_numpy_batch(
            self, generator: TensorGenerator, /
        ) -> tuple[NumpyArray, ...]:
            return factory(generator)

    benchmark_case = BenchmarkCase(
        name="repeat_identity",
        description="Deterministic repetition of each generated input.",
        calls=CaseCalls(einf="identity(x)", einops="identity(x)", einx="identity(x)"),
        reference=reference,
        make_einf_runner=make_runner,
        make_einops_runner=make_runner,
        make_einx_runner=make_runner,
    )
    runner = BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={name: (True, "available") for name in ("einf", "einops", "einx")},
    )
    return runner.run_dynamic_case(
        case_spec=FactoryCase(
            case=benchmark_case,
            sizes=BenchSizes(b=1, n=1, d=1, h=1, w=1, r=1, j=1),
            workload=DynamicShapeWorkload(
                sampled_dimensions=(),
                input_shapes=lambda dimensions: ((2, 3), (2, 3)),
                output_shapes=lambda dimensions: ((2, 3),),
            ),
        ),
        config=config,
        case_index=0,
    )


def _batch(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
    first = generator.randn_numpy((2, 3))
    return first, first + np.float32(10)


@pytest.mark.parametrize("family", ["libraries", "expression"])
@pytest.mark.parametrize("changed_call", [4, 5, 6, 11])
@pytest.mark.parametrize(
    "change",
    ["value", "sub_tolerance", "second_operand", "dtype", "layout", "order", "shape"],
)
def test_repeated_input_drift_fails_before_reference_transfer_or_timing(
    family: _Family,
    changed_call: int,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    generation_count = 0
    to_backend_batch = BackendSpec.to_backend_batch

    def transfer(self: BackendSpec, batch: tuple[NumpyArray, ...]) -> tuple[Array, ...]:
        events.append("transfer")
        return to_backend_batch(self, batch)

    monkeypatch.setattr(BackendSpec, "to_backend_batch", transfer)

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        nonlocal generation_count
        generation_count += 1
        events.append("generate")
        first, second = _batch(generator)
        if generation_count == changed_call:
            if change == "value":
                first[0, 0] += 1
            elif change == "sub_tolerance":
                first[0, 0] = np.nextafter(first[0, 0], np.float32(np.inf))
            elif change == "second_operand":
                second[0, 0] += 1
            elif change == "dtype":
                first = cast(NumpyArray, first.astype(np.float64))
            elif change == "layout":
                first = np.asfortranarray(first)
            elif change == "order":
                first, second = second, first
            elif change == "shape":
                first = first.reshape(3, 2)
        return first, second

    with pytest.raises(RuntimeError, match="repeated input.*round.*unit.*repeat"):
        _run_case(family=family, factory=factory, events=events)

    assert generation_count == changed_call
    assert events[-1] == "generate"
    assert events.count("transfer") == changed_call - 1


@pytest.mark.parametrize("family", ["libraries", "expression"])
def test_repeat_identity_preserves_schedule_round_streams_and_payload_release(
    family: _Family,
) -> None:
    events: list[str] = []
    weak_arrays: list[ReferenceType[NumpyArray]] = []
    signatures: list[tuple[bytes, ...]] = []

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        assert all(reference() is None for reference in weak_arrays)
        arrays = _batch(generator)
        weak_arrays.extend(ref(array) for array in arrays)
        signatures.append(tuple(array.tobytes() for array in arrays))
        return arrays

    result = _run_case(family=family, factory=factory, events=events)
    observations = (
        result.evidence.observations
        if isinstance(result, DynamicCaseResult)
        else result.observations
    )

    expected_schedule = [
        (round_index, unit_index, repeat_index)
        for round_index in range(2)
        for repeat_index in range(3)
        for unit_index in range(2)
        for _ in range(3)
    ]
    assert [
        (item.round_index, item.unit_index, item.repeat_index) for item in observations
    ] == expected_schedule
    assert [item.latency_ms for item in observations] == pytest.approx([1.0] * 36)
    for start in (0, 7):
        assert signatures[start + 1 : start + 3] == signatures[start + 3 : start + 5]
        assert signatures[start + 1 : start + 3] == signatures[start + 5 : start + 7]
    assert signatures[1:3] != signatures[8:10]
    assert all(reference() is None for reference in weak_arrays)


@pytest.mark.parametrize("family", ["libraries", "expression"])
def test_single_repeat_needs_no_fingerprint_scan(
    family: _Family, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fingerprint(batch: tuple[NumpyArray, ...]) -> bytes:
        pytest.fail("single-repeat cases need no input-identity scan")

    monkeypatch.setattr(runner_module, "numpy_batch_fingerprint", fingerprint)
    monkeypatch.setattr(expression_module, "numpy_batch_fingerprint", fingerprint)
    _run_case(
        family=family,
        factory=_batch,
        events=[],
        config=replace(_CONFIG, repeats=1),
    )


@pytest.mark.parametrize("family", ["libraries", "expression"])
def test_fingerprinting_is_outside_every_timed_interval(
    family: _Family, monkeypatch: pytest.MonkeyPatch
) -> None:
    timing = False
    ticks = 0
    fingerprints = 0

    def clock() -> float:
        nonlocal timing, ticks
        timing = not timing
        ticks += 1
        return ticks / 1000

    def fingerprint(batch: tuple[NumpyArray, ...]) -> bytes:
        nonlocal fingerprints
        assert not timing
        fingerprints += 1
        return numpy_batch_fingerprint(batch)

    monkeypatch.setattr(profiler_module.time, "perf_counter", clock)
    monkeypatch.setattr(runner_module, "numpy_batch_fingerprint", fingerprint)
    monkeypatch.setattr(expression_module, "numpy_batch_fingerprint", fingerprint)
    _run_case(family=family, factory=_batch, events=[])

    assert fingerprints == 12
    assert ticks == 72
    assert not timing


@pytest.mark.parametrize("family", ["libraries", "expression"])
@pytest.mark.parametrize("drift", [False, True], ids=["stable", "changed"])
def test_torch_repeat_input_identity(family: _Family, drift: bool) -> None:
    pytest.importorskip("torch")
    generations = 0

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        nonlocal generations
        generations += 1
        first, second = _batch(generator)
        if drift and generations == 4:
            first[0, 0] += 1
        return first, second

    if drift:
        with pytest.raises(RuntimeError, match="repeated input changed.*round 0"):
            _run_case(family=family, factory=factory, events=[], backend_name="torch")
        assert generations == 4
    else:
        _run_case(family=family, factory=factory, events=[], backend_name="torch")
        assert generations == 14


@pytest.mark.parametrize(
    "array",
    [
        np.arange(24, dtype=np.float32).reshape(4, 6),
        np.arange(24, dtype=np.float32).reshape(4, 6).T,
        np.arange(48, dtype=np.float32)[::3],
        np.arange(24, dtype=np.float32)[::-1],
        np.broadcast_to(np.asarray([2.0], dtype=np.float32), (24,)),
        np.arange(24, dtype=">f4")[::-2],
        np.asarray(7.0, dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        cast(NumpyArray, np.asarray([False, True])),
        cast(NumpyArray, np.asarray([1, 2], dtype=np.int64)),
        cast(NumpyArray, np.asarray([1 + 2j], dtype=np.complex128)),
    ],
    ids=[
        "contiguous",
        "fortran",
        "sliced",
        "negative_stride",
        "broadcast",
        "big_endian",
        "scalar",
        "empty",
        "bool",
        "integer",
        "complex",
    ],
)
def test_fingerprint_matches_framed_exact_c_order_bytes(array: NumpyArray) -> None:
    metadata = repr((array.dtype.str, array.shape, array.strides)).encode("ascii")
    expected = sha256()
    expected.update((1).to_bytes(8, "big"))
    expected.update(len(metadata).to_bytes(8, "big"))
    expected.update(metadata)
    expected.update(array.tobytes(order="C"))

    assert numpy_batch_fingerprint((array,)) == expected.digest()


def test_fingerprint_distinguishes_representation_not_just_equal_values() -> None:
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 10
    baseline = numpy_batch_fingerprint((first, second))
    assert numpy_batch_fingerprint((first.copy(), second.copy())) == baseline

    variants = (
        (first, second + 1),
        (second, first),
        (first,),
        (first, second, first),
        (first.reshape(3, 2), second),
        (np.asfortranarray(first), second),
        (cast(NumpyArray, first.astype(np.float64)), second),
        (cast(NumpyArray, first.astype(">f4")), second),
    )
    assert all(numpy_batch_fingerprint(batch) != baseline for batch in variants)


@pytest.mark.parametrize(
    ("first_bits", "second_bits"),
    [(0x00000000, 0x80000000), (0x7FC00000, 0x7FC00001)],
    ids=["signed_zero", "nan_payload"],
)
def test_fingerprint_preserves_exact_float_bits(
    first_bits: int, second_bits: int
) -> None:
    first = np.asarray([first_bits], dtype=np.uint32).view(np.float32)
    second = np.asarray([second_bits], dtype=np.uint32).view(np.float32)

    assert numpy_batch_fingerprint((first,)) != numpy_batch_fingerprint((second,))


def test_strided_fingerprint_uses_bounded_buffers_and_releases_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer_sizes: list[int] = []

    class RecordingHash:
        def __init__(self) -> None:
            self.digest_state = sha256()

        def update(self, value: bytes | memoryview) -> None:
            if isinstance(value, memoryview):
                assert value.c_contiguous
                buffer_sizes.append(value.nbytes)
            else:
                buffer_sizes.append(len(value))
            self.digest_state.update(value)

        def digest(self) -> bytes:
            return self.digest_state.digest()

    monkeypatch.setattr(identity_module, "sha256", RecordingHash)
    array = np.arange(257 * 513, dtype=np.float32).reshape(257, 513).T
    array_ref = ref(array)
    digest = numpy_batch_fingerprint((array,))
    payload_sizes = buffer_sizes[3:]

    assert len(digest) == 32
    assert len(payload_sizes) > 1
    assert sum(payload_sizes) == array.nbytes
    assert max(payload_sizes) <= 16_384 * array.itemsize
    del array
    assert array_ref() is None


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([object()], dtype=object),
        np.asarray(["not numeric"]),
        np.asarray([(1, 2)], dtype=[("first", "i4"), ("second", "i4")]),
    ],
    ids=["object", "text", "structured"],
)
def test_fingerprint_rejects_non_numeric_storage(array: NumpyArray) -> None:
    with pytest.raises(TypeError, match=r"input\[0\].*numeric or boolean NumPy array"):
        numpy_batch_fingerprint((array,))
