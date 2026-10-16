import gc
import itertools
import sys
from typing import Literal
from weakref import ReferenceType, ref

import pytest

pytest.importorskip("numpy")

import numpy as np

import benchmarks.compare.expression_parity as expression_module
import benchmarks.harness.profiler as profiler_module
from benchmarks.compare.expression_parity import (
    ExpressionParityCase,
    ExpressionRunnerSpec,
)
from benchmarks.harness import (
    Array,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    BenchSizes,
    CaseCalls,
    DynamicCaseSpec,
    DynamicTaskConfig,
    FixedCaseSpec,
    FixedTaskConfig,
    Output,
    Profiler,
    TensorGenerator,
)
from benchmarks.harness.types import BackendName, NumpyArray, NumpyOutput, Runner, torch
from benchmarks.harness.workload import DynamicShapeWorkload


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("family", ["fixed", "dynamic", "expression"])
@pytest.mark.parametrize("mismatch", ["shape", "value"])
@pytest.mark.parametrize("multiple_outputs", [False, True], ids=["single", "tuple"])
def test_retained_case_failure_releases_coordinate_payloads(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: BackendName,
    family: Literal["fixed", "dynamic", "expression"],
    mismatch: Literal["shape", "value"],
    multiple_outputs: bool,
) -> None:
    if backend_name == "torch":
        pytest.importorskip("torch")
    ticks = itertools.count()
    monkeypatch.setattr(profiler_module.time, "perf_counter", lambda: next(ticks))
    observed: dict[str, list[ReferenceType[Array]]] = {}

    def track(label: str, arrays: tuple[Array, ...]) -> None:
        observed.setdefault(label, []).extend(ref(array) for array in arrays)

    def make_numpy_batch() -> tuple[NumpyArray, ...]:
        batch = (np.ones(2, dtype=np.float32),)
        track("host inputs", batch)
        return batch

    def reference(inputs: tuple[NumpyArray, ...]) -> NumpyOutput:
        track("reference inputs", inputs)
        return (inputs[0], inputs[0] * 2) if multiple_outputs else inputs[0]

    def make_runner() -> Runner:
        def run(inputs: tuple[Array, ...]) -> Output:
            track("prepared inputs", inputs)
            size = 3 if mismatch == "shape" else 2
            if isinstance(inputs[0], np.ndarray):
                incorrect: Array = np.zeros(size, dtype=np.float32)
            else:
                assert torch is not None
                incorrect = torch.zeros(size, device=inputs[0].device)
            output = (inputs[0], incorrect) if multiple_outputs else incorrect
            track("timed outputs", output if isinstance(output, tuple) else (output,))
            return output

        return run

    original_to_numpy_output = BackendSpec.to_numpy_output

    def to_numpy_output(self: BackendSpec, output: Output) -> tuple[NumpyArray, ...]:
        # Independent conversion buffers must be released, not just device outputs.
        converted = tuple(
            array.copy() for array in original_to_numpy_output(self, output)
        )
        track("converted outputs", converted)
        return converted

    monkeypatch.setattr(BackendSpec, "to_numpy_output", to_numpy_output)
    original_validate = BenchmarkRunner._validate_output

    def validate(
        self: BenchmarkRunner,
        output: Output,
        *,
        case: BenchmarkCase,
        expected: tuple[NumpyArray, ...],
        library_name: str,
    ) -> None:
        track("expected", expected)
        original_validate(
            self, case=case, expected=expected, output=output, library_name=library_name
        )

    monkeypatch.setattr(BenchmarkRunner, "_validate_output", validate)
    original_expression_validate = expression_module._validate_output

    def validate_expression(
        output: Output,
        *,
        backend: BackendSpec,
        runner_spec: ExpressionRunnerSpec,
        expected: tuple[NumpyArray, ...],
    ) -> None:
        track("expected", expected)
        original_expression_validate(
            backend=backend, runner_spec=runner_spec, expected=expected, output=output
        )

    monkeypatch.setattr(expression_module, "_validate_output", validate_expression)
    backend = BackendSpec(name=backend_name)
    profiler = Profiler(backend=backend)
    config = DynamicTaskConfig(
        scale="medium",
        seed=1,
        batches=1,
        warmup_batches=0,
        repeats=1,
        rounds=1,
        round_order_seed=1,
    )

    def execute() -> None:
        if family == "expression":
            expression_module._run_dynamic_case(
                case=ExpressionParityCase(
                    name="retention",
                    description="Retained validation failure.",
                    target_name="target",
                    batch_factory=lambda generator: make_numpy_batch(),
                    runner_specs=(
                        ExpressionRunnerSpec(
                            name="target",
                            semantics="output_equivalent",
                            description="",
                            call_repr="faulty(x)",
                            available=True,
                            reason="available",
                            reference=reference,
                            make_runner=make_runner,
                        ),
                    ),
                ),
                config=config,
                profiler=profiler,
                backend=backend,
                case_index=0,
            )
            return

        case = BenchmarkCase(
            name="retention",
            description="Retained validation failure.",
            calls=CaseCalls(einf="faulty(x)", einops="faulty(x)", einx="faulty(x)"),
            reference=reference,
            make_einf_runner=make_runner,
            make_einops_runner=make_runner,
            make_einx_runner=make_runner,
        )
        runner = BenchmarkRunner(
            backend=backend,
            profiler=profiler,
            available={
                "einf": (True, "available"),
                "einops": (False, "unused"),
                "einx": (False, "unused"),
            },
        )
        if family == "fixed":
            runner.run_fixed_case(
                case_spec=FixedCaseSpec(
                    case=case, inputs=backend.to_backend_batch(make_numpy_batch())
                ),
                config=FixedTaskConfig(
                    scale="small", seed=1, rounds=1, warmup=0, repeats=1, iterations=1
                ),
                order_seed=1,
            )
        else:

            class CaseSpec(DynamicCaseSpec):
                def make_numpy_batch(
                    self, generator: TensorGenerator, /
                ) -> tuple[NumpyArray, ...]:
                    return make_numpy_batch()

            runner.run_dynamic_case(
                case_spec=CaseSpec(
                    case=case,
                    sizes=BenchSizes(b=2, n=2, d=2, h=2, w=2, r=2, j=2),
                    workload=DynamicShapeWorkload(
                        sampled_dimensions=(),
                        input_shapes=lambda values: ((2,),),
                        output_shapes=lambda values: (
                            ((2,), (2,)) if multiple_outputs else ((2,),)
                        ),
                    ),
                ),
                config=config,
                case_index=0,
            )

    owner = "target" if family == "expression" else "einf/retention"
    output_index = 1 if multiple_outputs else 0
    with pytest.raises(
        ValueError, match=rf"{owner} output\[{output_index}\] {mismatch} mismatch"
    ) as caught:
        execute()

    gc.collect()
    assert set(observed) == {
        "host inputs",
        "reference inputs",
        "prepared inputs",
        "timed outputs",
        "converted outputs",
        "expected",
    }
    assert all(observed.values())
    alive = {
        label: sum(reference() is not None for reference in references)
        for label, references in observed.items()
    }
    assert not any(alive.values()), alive
    assert caught.value.__traceback__ is not None


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("stage", ["reference", "runner"])
def test_case_releases_payloads_when_callback_reraises_caller_exception(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: BackendName,
    stage: Literal["reference", "runner"],
) -> None:
    if backend_name == "torch":
        pytest.importorskip("torch")
    ticks = itertools.count()
    monkeypatch.setattr(profiler_module.time, "perf_counter", lambda: next(ticks))
    payloads: dict[str, list[ReferenceType[Array]]] = {}

    def track(label: str, array: Array) -> None:
        payloads.setdefault(label, []).append(ref(array))

    def reference(inputs: tuple[NumpyArray, ...]) -> NumpyOutput:
        track("reference", inputs[0])
        if stage == "reference":
            active = sys.exception()
            assert active is not None
            raise active
        return inputs[0]

    def make_runner() -> Runner:
        def run(inputs: tuple[Array, ...]) -> Output:
            track("runner", inputs[0])
            active = sys.exception()
            assert active is not None
            raise active

        return run

    backend = BackendSpec(name=backend_name)
    runner = BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={
            "einf": (True, "available"),
            "einops": (False, "unused"),
            "einx": (False, "unused"),
        },
    )
    case = BenchmarkCase(
        name="reraised_caller_exception",
        description="A callback re-raises the caller's active exception.",
        calls=CaseCalls(einf="raise", einops="unused", einx="unused"),
        reference=reference,
        make_einf_runner=make_runner,
        make_einops_runner=make_runner,
        make_einx_runner=make_runner,
    )

    def materialize() -> tuple[Array, ...]:
        batch = backend.to_backend_batch((np.ones(2, dtype=np.float32),))
        track("prepared", batch[0])
        return batch

    def caller_helper() -> None:
        payload = np.ones(2, dtype=np.float32)
        track("caller", payload)
        raise KeyError("caller failure")

    try:
        caller_helper()
    except KeyError as outer:
        assert outer.__traceback__ is not None
        assert outer.__traceback__.tb_next is not None
        caller_frame = outer.__traceback__.tb_next.tb_frame
        with pytest.raises(KeyError, match="caller failure") as caught:
            runner.run_fixed_case(
                case_spec=FixedCaseSpec(case=case, inputs=materialize()),
                config=FixedTaskConfig(
                    scale="small", seed=1, rounds=1, warmup=0, repeats=1, iterations=1
                ),
                order_seed=1,
            )

        gc.collect()
        assert caught.value is outer
        assert caller_frame.f_locals["payload"] is payloads["caller"][0]()
        expected_labels = {"caller", "prepared", "reference"}
        if stage == "runner":
            expected_labels.add("runner")
        assert set(payloads) == expected_labels
        for label in expected_labels - {"caller"}:
            assert payloads[label] and all(
                weak() is None for weak in payloads[label]
            ), label
    else:
        pytest.fail("caller helper did not raise")
