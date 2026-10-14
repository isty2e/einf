from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("numpy")

import numpy as np

import benchmarks.audit.expression_layout as layout_module
import benchmarks.compare.expression_parity as expression_module
from benchmarks.compare.expression_parity import (
    BatchFactory,
    ExpressionParityCase,
    ExpressionRunnerSpec,
    Reference,
    build_gap_cases,
    generate_expression_numpy_batch,
)
from benchmarks.harness import (
    BackendSpec,
    BenchSizes,
    DynamicTaskConfig,
    Profiler,
    torch,
)
from benchmarks.harness.generator import TensorGenerator
from benchmarks.harness.types import Array, BackendName, NumpyArray, Output, Runner

_SIZES = BenchSizes(b=2, n=3, d=2, h=2, w=1, r=1, j=1)
_CONFIG = DynamicTaskConfig(
    scale="medium",
    seed=7,
    batches=3,
    warmup_batches=1,
    repeats=2,
    rounds=2,
    round_order_seed=1234,
)


def _identity_case(
    *,
    batch_factory: BatchFactory,
    reference: Reference = lambda inputs: inputs[0],
) -> ExpressionParityCase:
    def make_runner() -> Runner:
        def run(inputs: tuple[Array, ...]) -> Output:
            array = inputs[0]
            if isinstance(array, np.ndarray):
                return array.copy()
            return array.clone()

        return run

    return ExpressionParityCase(
        name="host_inputs",
        description="Host input preparation with an independent reference snapshot.",
        target_name="identity",
        batch_factory=batch_factory,
        runner_specs=(
            ExpressionRunnerSpec(
                name="identity",
                semantics="output_equivalent",
                description="Copy the input on its existing backend.",
                call_repr="copy(x)",
                available=True,
                reason="available",
                reference=reference,
                make_runner=make_runner,
            ),
        ),
    )


def test_host_generation_preserves_factory_array_identity_and_layout() -> None:
    arrays = (np.arange(12, dtype=np.float32).reshape(3, 4).T,)
    generator = TensorGenerator.from_seed(backend=BackendSpec(name="numpy"), seed=7)
    received: list[TensorGenerator] = []

    def factory(current: TensorGenerator) -> tuple[NumpyArray, ...]:
        received.append(current)
        return arrays

    batch = generate_expression_numpy_batch(
        case=_identity_case(batch_factory=factory),
        generator=generator,
    )

    assert batch is arrays
    assert batch[0] is arrays[0]
    assert batch[0].strides == arrays[0].strides
    assert not batch[0].flags.c_contiguous
    assert received == [generator]


def test_host_generation_rejects_target_generator_before_calling_factory() -> None:
    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        pytest.fail("factory must not run with a target generator")

    generator = cast(
        TensorGenerator,
        SimpleNamespace(backend=SimpleNamespace(name="torch")),
    )

    with pytest.raises(ValueError, match="NumPy generator"):
        generate_expression_numpy_batch(
            case=_identity_case(batch_factory=factory),
            generator=generator,
        )


@pytest.mark.parametrize(
    "payload",
    [None, [np.ones(1, dtype=np.float32)], (np.ones(1, dtype=np.float32), "device")],
    ids=["none", "list", "invalid_second_input"],
)
def test_host_generation_rejects_non_numpy_factory_payload(payload: object) -> None:
    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        return cast(tuple[NumpyArray, ...], payload)

    with pytest.raises(TypeError, match="factory must return"):
        generate_expression_numpy_batch(
            case=_identity_case(batch_factory=factory),
            generator=TensorGenerator.from_seed(
                backend=BackendSpec(name="numpy"), seed=7
            ),
        )


def test_host_generation_rejects_torch_factory_output() -> None:
    pytest.importorskip("torch")
    assert torch is not None
    tensor = torch.ones((2, 3))

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        return cast(tuple[NumpyArray, ...], (tensor,))

    with pytest.raises(TypeError, match="factory must return NumPy arrays"):
        generate_expression_numpy_batch(
            case=_identity_case(batch_factory=factory),
            generator=TensorGenerator.from_seed(
                backend=BackendSpec(name="numpy"), seed=7
            ),
        )


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reference_normalization_and_resize_do_not_change_target_inputs(
    backend_name: BackendName,
    dtype: type[np.float32] | type[np.float64],
) -> None:
    if backend_name == "torch":
        pytest.importorskip("torch")
    backend = BackendSpec(name=backend_name)
    generated: list[NumpyArray] = []
    reference_dtypes: list[np.dtype] = []

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        assert generator.backend.name == "numpy"
        array = cast(NumpyArray, np.asarray([1.00000002, 2.00000004], dtype=dtype))
        generated.append(array)
        return (array,)

    def reference(inputs: tuple[NumpyArray, ...]) -> Output:
        array = inputs[0]
        reference_dtypes.append(array.dtype)
        expected = array.copy()
        array.resize((3,), refcheck=False)
        array[:] = (10.0, 20.0, 30.0)
        return expected

    case = _identity_case(batch_factory=factory, reference=reference)
    case = replace(
        case,
        runner_specs=(case.target, replace(case.target, name="equivalent")),
    )
    result = expression_module._run_dynamic_case(
        case=case,
        config=_CONFIG,
        profiler=Profiler(backend=backend),
        backend=backend,
        case_index=0,
    )

    assert len(result.observations) == 16
    assert reference_dtypes == [np.dtype(np.float32)] * 16
    assert len(generated) == 10
    for array in generated:
        assert array.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(
            array, np.asarray([1.00000002, 2.00000004], dtype=dtype)
        )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
def test_expression_transfers_each_coordinate_once_without_input_readback(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    assert torch is not None
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    backend = BackendSpec(name="torch", requested_device=device)
    case = build_gap_cases(sizes=_SIZES)[0]
    case = replace(
        case,
        target_name="torch_matmul_slice",
        runner_specs=tuple(
            spec
            for spec in case.runner_specs
            if spec.name in ("torch_matmul_only", "torch_matmul_slice")
        ),
    )
    materialized: list[tuple[tuple[int, ...], ...]] = []
    active_input_ids: set[int] = set()
    readbacks: list[tuple[bool, int]] = []
    original_transfer = BackendSpec.to_backend_batch
    original_readback = BackendSpec.to_numpy_array

    def transfer(
        current: BackendSpec,
        arrays: tuple[NumpyArray, ...],
    ) -> tuple[Array, ...]:
        batch = original_transfer(current, arrays)
        if current.name == "torch":
            materialized.append(tuple(array.shape for array in arrays))
            active_input_ids.clear()
            active_input_ids.update(id(array) for array in batch)
        return batch

    def readback(current: BackendSpec, value: Array) -> NumpyArray:
        if not isinstance(value, np.ndarray):
            readbacks.append(
                (id(value) in active_input_ids, value.numel() * value.element_size())
            )
        return original_readback(current, value)

    monkeypatch.setattr(BackendSpec, "to_backend_batch", transfer)
    monkeypatch.setattr(BackendSpec, "to_numpy_array", readback)

    result = expression_module._run_dynamic_case(
        case=case,
        config=_CONFIG,
        profiler=Profiler(backend=backend),
        backend=backend,
        case_index=0,
    )

    assert len(materialized) == 10
    assert all(len(shapes) == 2 for shapes in materialized)
    assert sum(byte_count for is_input, byte_count in readbacks if is_input) == 0
    assert len(readbacks) == 24
    expected_output_bytes = (
        sum(
            unit.input_shapes[0][0]
            * unit.input_shapes[0][1]
            * unit.input_shapes[1][1]
            * 4
            for unit in result.realized_units
        )
        * _CONFIG.repeats
        * len(case.runner_specs)
    )
    assert sum(byte_count for _, byte_count in readbacks) == expected_output_bytes
    assert len(result.observations) == 16


def test_layout_audit_materializes_host_factories_on_torch() -> None:
    pytest.importorskip("torch")
    assert torch is not None

    def factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        assert generator.backend.name == "numpy"
        return (generator.randn_numpy((2, 3)),)

    batches = layout_module._make_batches(
        case=_identity_case(batch_factory=factory), seed=7, count=2
    )

    for index, batch in enumerate(batches):
        tensor = batch[0]
        assert not isinstance(tensor, np.ndarray)
        assert isinstance(tensor, torch.Tensor)
        expected = TensorGenerator.from_seed(
            backend=BackendSpec(name="numpy"), seed=7 + index
        ).randn_numpy((2, 3))
        np.testing.assert_array_equal(tensor.numpy(), expected)
