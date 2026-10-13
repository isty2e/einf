import numpy as np
import pytest

import benchmarks.compare.expression_parity as expression_parity_module
from benchmarks.compare.expression_parity import (
    ExpressionParityCase,
    ExpressionRunnerSpec,
    ExpressionSemantics,
)
from benchmarks.harness import (
    Array,
    BackendSpec,
    BenchmarkCase,
    BenchmarkRunner,
    CaseCalls,
    DynamicTaskConfig,
    FixedCaseSpec,
    FixedTaskConfig,
    LibraryName,
    Output,
    Profiler,
    TensorGenerator,
)
from benchmarks.harness.types import NumpyArray, NumpyOutput, Runner

_LIBRARIES: tuple[LibraryName, ...] = ("einf", "einops", "einx")
_EXPRESSION_CONFIG = DynamicTaskConfig(
    scale="medium",
    seed=7,
    batches=3,
    warmup_batches=1,
    repeats=1,
    rounds=1,
    round_order_seed=1234,
)


@pytest.mark.parametrize("library_name", _LIBRARIES)
@pytest.mark.parametrize(
    ("rounds", "repeats", "iterations"),
    [(1, 1, 3), (1, 3, 1), (3, 1, 1)],
    ids=["iteration", "repeat", "round"],
)
def test_fixed_runner_rejects_wrong_second_timed_call(
    library_name: LibraryName,
    rounds: int,
    repeats: int,
    iterations: int,
) -> None:
    call_count = 0

    def make_faulty_runner() -> Runner:
        def run(inputs: tuple[Array, ...]) -> Output:
            nonlocal call_count
            call_count += 1
            array = inputs[0]
            assert isinstance(array, np.ndarray)
            # Warmup and the first timed call are correct; later calls recover.
            return np.zeros_like(array) if call_count == 3 else array

        return run

    def make_correct_runner() -> Runner:
        return lambda inputs: inputs[0]

    case = BenchmarkCase(
        name="second_timed_call",
        description="Only the second timed output is incorrect.",
        calls=CaseCalls(einf="identity(x)", einops="identity(x)", einx="identity(x)"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=(
            make_faulty_runner if library_name == "einf" else make_correct_runner
        ),
        make_einops_runner=(
            make_faulty_runner if library_name == "einops" else make_correct_runner
        ),
        make_einx_runner=(
            make_faulty_runner if library_name == "einx" else make_correct_runner
        ),
    )
    backend = BackendSpec(name="numpy")
    runner = BenchmarkRunner(
        backend=backend,
        profiler=Profiler(backend=backend),
        available={
            name: (
                name == library_name,
                "selected" if name == library_name else "unused",
            )
            for name in _LIBRARIES
        },
    )

    with pytest.raises(ValueError, match=rf"{library_name}/.*value mismatch"):
        runner.run_fixed_case(
            case_spec=FixedCaseSpec(
                case=case,
                inputs=(np.asarray([1.0, 2.0], dtype=np.float32),),
            ),
            config=FixedTaskConfig(
                scale="small",
                seed=1,
                rounds=rounds,
                warmup=1,
                repeats=repeats,
                iterations=iterations,
            ),
            order_seed=1,
        )

    assert call_count == 3


def _expression_case(
    *,
    calls: dict[str, int],
    wrong_strategy: str | None = None,
    wrong_call: int = 3,
) -> ExpressionParityCase:
    def make_spec(name: str, semantics: ExpressionSemantics) -> ExpressionRunnerSpec:
        def reference(inputs: tuple[NumpyArray, ...]) -> NumpyOutput:
            if semantics == "lower_bound":
                return inputs[0]
            return (inputs[0], inputs[0] * 2.0)

        def make_runner() -> Runner:
            def run(inputs: tuple[Array, ...]) -> Output:
                calls[name] = calls.get(name, 0) + 1
                array = inputs[0]
                assert isinstance(array, np.ndarray)
                faulty = name == wrong_strategy and calls[name] == wrong_call
                if semantics == "lower_bound":
                    return np.zeros_like(array) if faulty else array
                second_output = np.zeros_like(array) if faulty else array * 2.0
                return (array, second_output)

            return run

        return ExpressionRunnerSpec(
            name=name,
            semantics=semantics,
            description="One synthetic expression strategy.",
            call_repr=f"{name}(x)",
            available=True,
            reason="available",
            reference=reference,
            make_runner=make_runner,
        )

    def batch_factory(generator: TensorGenerator) -> tuple[NumpyArray, ...]:
        return (np.asarray([1.0, 2.0], dtype=np.float32),)

    return ExpressionParityCase(
        name="non_target_validation",
        description="Each strategy must satisfy its own output reference.",
        target_name="target",
        batch_factory=batch_factory,
        runner_specs=(
            make_spec("target", "output_equivalent"),
            make_spec("equivalent", "output_equivalent"),
            make_spec("lower_bound", "lower_bound"),
        ),
    )


@pytest.mark.parametrize("wrong_strategy", ["equivalent", "lower_bound"])
@pytest.mark.parametrize("wrong_call", [2, 3], ids=["first_timed", "second_timed"])
def test_expression_runner_rejects_wrong_non_target_timed_output(
    wrong_strategy: str,
    wrong_call: int,
) -> None:
    calls: dict[str, int] = {}
    case = _expression_case(
        calls=calls,
        wrong_strategy=wrong_strategy,
        wrong_call=wrong_call,
    )
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match=rf"{wrong_strategy} .*value mismatch"):
        expression_parity_module._run_dynamic_case(
            case=case,
            config=_EXPRESSION_CONFIG,
            profiler=Profiler(backend=backend),
            backend=backend,
            case_index=0,
        )

    assert calls[wrong_strategy] == wrong_call


def test_expression_runner_accepts_correct_strategy_specific_outputs() -> None:
    calls: dict[str, int] = {}
    backend = BackendSpec(name="numpy")

    result = expression_parity_module._run_dynamic_case(
        case=_expression_case(calls=calls),
        config=_EXPRESSION_CONFIG,
        profiler=Profiler(backend=backend),
        backend=backend,
        case_index=0,
    )

    assert calls == {"target": 3, "equivalent": 3, "lower_bound": 3}
    assert len(result.observations) == 6
