from dataclasses import dataclass

import numpy as np

from .backend import BackendSpec
from .case import BenchmarkCase, DynamicCaseSpec, FixedCaseSpec, RunnerFactory
from .comparison import compare_library_timings
from .config import DynamicTaskConfig, FixedTaskConfig
from .generator import TensorGenerator, derive_coordinate_seed
from .profiler import Profiler
from .result import (
    AvailableRun,
    DynamicCaseResult,
    DynamicInputUnit,
    FixedCaseResult,
    LibraryTimingObservation,
    PairedEvidence,
    RunResult,
    UnavailableRun,
)
from .types import Array, LibraryName, NumpyArray, Runner

CASE_SEED_STRIDE = 1009


@dataclass(frozen=True, slots=True)
class BenchmarkRunner:
    """Runner for fixed and dynamic benchmark case specs."""

    backend: BackendSpec
    profiler: Profiler
    available: dict[str, tuple[bool, str]]

    def _rotate_order(
        self,
        order: tuple[LibraryName, ...],
        *,
        offset: int,
    ) -> tuple[LibraryName, ...]:
        """Rotate one round base order by offset steps."""
        if not order:
            return ()
        index = offset % len(order)
        return order[index:] + order[:index]

    def _round_orders(
        self,
        *,
        library_names: list[LibraryName],
        rounds: int,
        seed: int,
    ) -> list[tuple[LibraryName, ...]]:
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")
        random_state = np.random.RandomState(seed)
        base_order = list(library_names)
        random_state.shuffle(base_order)
        orders: list[tuple[LibraryName, ...]] = []
        for round_index in range(rounds):
            if round_index > 0 and round_index % len(base_order) == 0:
                random_state.shuffle(base_order)
            offset = round_index % len(base_order)
            order = base_order[offset:] + base_order[:offset]
            orders.append(tuple(order))
        return orders

    def _libraries_for_case(
        self,
        case: BenchmarkCase,
    ) -> dict[LibraryName, RunnerFactory]:
        return {
            "einf": case.make_einf_runner,
            "einops": case.make_einops_runner,
            "einx": case.make_einx_runner,
        }

    def _require_full_round_orders(
        self,
        *,
        round_orders: list[tuple[LibraryName, ...]],
        libraries: list[LibraryName],
    ) -> None:
        expected_libraries = frozenset(libraries)
        if any(
            len(round_order) != len(libraries)
            or frozenset(round_order) != expected_libraries
            for round_order in round_orders
        ):
            raise RuntimeError(
                "benchmark round orders must be full library permutations"
            )

    def _paired_evidence(
        self,
        *,
        observations: list[LibraryTimingObservation],
        libraries: list[LibraryName],
        bootstrap_seed: int,
    ) -> PairedEvidence:
        observation_tuple = tuple(observations)
        return PairedEvidence(
            observations=observation_tuple,
            comparisons=compare_library_timings(
                observations=observation_tuple,
                libraries=tuple(libraries),
                baseline="einf",
                bootstrap_seed=bootstrap_seed,
            ),
        )

    def _validate_output(
        self,
        *,
        case: BenchmarkCase,
        expected: tuple[NumpyArray, ...],
        output: Array | tuple[Array, ...],
        library_name: str,
    ) -> None:
        self.backend.validate_output_target(output)
        got = self.backend.to_numpy_output(output)
        if len(expected) != len(got):
            raise ValueError(
                f"{library_name}/{case.name} output arity mismatch: "
                f"expected {len(expected)}, got {len(got)}"
            )

        uses_torch_backend = self.backend.name == "torch"
        atol = 1e-4 if uses_torch_backend else 1e-5
        rtol = 1e-4 if uses_torch_backend else 1e-5

        for index, (expected_array, got_array) in enumerate(
            zip(expected, got, strict=True)
        ):
            if expected_array.shape != got_array.shape:
                raise ValueError(
                    f"{library_name}/{case.name} output[{index}] shape mismatch: "
                    f"expected {expected_array.shape}, got {got_array.shape}"
                )
            if not np.allclose(expected_array, got_array, atol=atol, rtol=rtol):
                max_abs = float(np.max(np.abs(expected_array - got_array)))
                raise ValueError(
                    f"{library_name}/{case.name} output[{index}] value mismatch: "
                    f"max abs diff={max_abs}, atol={atol}, rtol={rtol}"
                )

    def _validate_runners(
        self,
        *,
        case: BenchmarkCase,
        runners: dict[LibraryName, Runner],
        batch: tuple[Array, ...],
        expected: tuple[NumpyArray, ...],
        order: tuple[LibraryName, ...],
    ) -> None:
        """Validate available runners on one shared prepared batch."""
        for library_name in order:
            output = runners[library_name](batch)
            self.backend.synchronize()
            self._validate_output(
                case=case,
                expected=expected,
                output=output,
                library_name=library_name,
            )

    def _warmup_coordinate(
        self,
        *,
        runners: dict[LibraryName, Runner],
        batch: tuple[Array, ...],
        order: tuple[LibraryName, ...],
    ) -> None:
        """Warm available runners on one shared prepared batch."""
        for library_name in order:
            output = runners[library_name](batch)
            self.backend.synchronize()
            self.backend.validate_output_target(output)

    def _measure_coordinate(
        self,
        *,
        runners: dict[LibraryName, Runner],
        batch: tuple[Array, ...],
        order: tuple[LibraryName, ...],
        round_index: int,
        unit_index: int,
        repeat_index: int,
    ) -> list[LibraryTimingObservation]:
        """Measure every library at one paired prepared-input coordinate."""
        observations: list[LibraryTimingObservation] = []
        for order_position, library_name in enumerate(order):
            observations.append(
                LibraryTimingObservation(
                    round_index=round_index,
                    unit_index=unit_index,
                    repeat_index=repeat_index,
                    library=library_name,
                    order_position=order_position,
                    latency_ms=self.profiler.measure_call(
                        runner=runners[library_name],
                        batch=batch,
                    ),
                )
            )
        return observations

    def _available_run(
        self,
        *,
        observations: list[LibraryTimingObservation],
        library_name: LibraryName,
        rounds: int,
    ) -> AvailableRun:
        """Summarize one available library from direct observations."""
        library_observations = [
            observation
            for observation in observations
            if observation.library == library_name
        ]
        return AvailableRun(
            summary=self.profiler.summarize(
                [observation.latency_ms for observation in library_observations]
            ),
            round_summaries=tuple(
                self.profiler.summarize(
                    [
                        observation.latency_ms
                        for observation in library_observations
                        if observation.round_index == round_index
                    ]
                )
                for round_index in range(rounds)
            ),
        )

    def run_fixed_case(
        self,
        *,
        case_spec: FixedCaseSpec,
        config: FixedTaskConfig,
        order_seed: int,
    ) -> FixedCaseResult:
        """Run one fixed-shape case across libraries."""
        case = case_spec.case
        inputs = case_spec.inputs

        runs: dict[LibraryName, RunResult] = {}
        available_libs: list[LibraryName] = []
        for lib_name in ("einf", "einops", "einx"):
            is_available, reason = self.available[lib_name]
            if not is_available:
                runs[lib_name] = UnavailableRun(reason=reason)
                continue
            available_libs.append(lib_name)

        if not available_libs:
            empty_evidence = PairedEvidence(observations=(), comparisons=())
            return FixedCaseResult(
                case=case,
                runs=runs,
                round_orders=[],
                evidence=empty_evidence,
            )

        runner_factories = self._libraries_for_case(case)
        validation_runners: dict[LibraryName, Runner] = {
            library_name: runner_factories[library_name]()
            for library_name in available_libs
        }
        round_orders = self._round_orders(
            library_names=available_libs,
            rounds=config.rounds,
            seed=order_seed,
        )
        self._require_full_round_orders(
            round_orders=round_orders,
            libraries=available_libs,
        )

        expected = tuple(
            array.copy()
            for array in self.backend.to_numpy_output(
                case.reference(
                    tuple(self.backend.to_numpy_array(item) for item in inputs)
                )
            )
        )
        self._validate_runners(
            case=case,
            runners=validation_runners,
            batch=inputs,
            expected=expected,
            order=round_orders[0],
        )
        del validation_runners

        runners: dict[LibraryName, Runner] = {
            library_name: runner_factories[library_name]()
            for library_name in available_libs
        }

        for warmup_index in range(config.warmup):
            self._warmup_coordinate(
                runners=runners,
                batch=inputs,
                order=self._rotate_order(
                    round_orders[0],
                    offset=warmup_index,
                ),
            )

        observations: list[LibraryTimingObservation] = []
        for round_index, round_order in enumerate(round_orders):
            for unit_index in range(config.repeats):
                for repeat_index in range(config.iterations):
                    observations.extend(
                        self._measure_coordinate(
                            runners=runners,
                            batch=inputs,
                            order=self._rotate_order(
                                round_order,
                                offset=(
                                    unit_index * config.iterations + repeat_index
                                ),
                            ),
                            round_index=round_index,
                            unit_index=unit_index,
                            repeat_index=repeat_index,
                        )
                    )

        for library_name in available_libs:
            runs[library_name] = self._available_run(
                observations=observations,
                library_name=library_name,
                rounds=config.rounds,
            )

        return FixedCaseResult(
            case=case,
            runs=runs,
            round_orders=round_orders,
            evidence=self._paired_evidence(
                observations=observations,
                libraries=available_libs,
                bootstrap_seed=order_seed,
            ),
        )

    def _make_dynamic_numpy_batch(
        self,
        *,
        case_spec: DynamicCaseSpec,
        seed: int,
    ) -> tuple[NumpyArray, ...]:
        generator = TensorGenerator.from_seed(
            backend=BackendSpec(name="numpy"),
            seed=seed,
        )
        return case_spec.make_numpy_batch(generator)

    @staticmethod
    def _dynamic_batch_seed(
        *,
        config: DynamicTaskConfig,
        case_index: int,
        round_index: int,
        stream_index: int,
    ) -> int:
        """Return the deterministic seed for one logical dynamic input unit."""
        return derive_coordinate_seed(
            seed=config.seed,
            case_index=case_index,
            round_index=round_index,
            stream_index=stream_index,
        )

    def run_dynamic_case(
        self,
        *,
        case_spec: DynamicCaseSpec,
        config: DynamicTaskConfig,
        case_index: int,
    ) -> DynamicCaseResult:
        """Run one dynamic-shape case across libraries."""
        case = case_spec.case

        runs: dict[LibraryName, RunResult] = {}
        available_libs: list[LibraryName] = [
            lib_name
            for lib_name in ("einf", "einops", "einx")
            if self.available[lib_name][0]
        ]
        for lib_name in ("einf", "einops", "einx"):
            if self.available[lib_name][0]:
                continue
            runs[lib_name] = UnavailableRun(reason=self.available[lib_name][1])

        if not available_libs:
            return DynamicCaseResult(
                case=case,
                workload=case_spec.workload_metadata,
                realized_units=(),
                runs=runs,
                round_orders=[],
                evidence=PairedEvidence(observations=(), comparisons=()),
            )

        runner_factories = self._libraries_for_case(case)
        round_orders = self._round_orders(
            library_names=available_libs,
            rounds=config.rounds,
            seed=config.round_order_seed + case_index * CASE_SEED_STRIDE,
        )
        self._require_full_round_orders(
            round_orders=round_orders,
            libraries=available_libs,
        )

        checks = min(config.batches, config.parity_checks)
        if checks:
            validation_runners: dict[LibraryName, Runner] = {
                library_name: runner_factories[library_name]()
                for library_name in available_libs
            }
            for stream_index in range(checks):
                numpy_batch = self._make_dynamic_numpy_batch(
                    case_spec=case_spec,
                    seed=self._dynamic_batch_seed(
                        config=config,
                        case_index=case_index,
                        round_index=0,
                        stream_index=stream_index,
                    ),
                )
                expected = tuple(
                    array.copy()
                    for array in self.backend.to_numpy_output(
                        case.reference(numpy_batch)
                    )
                )
                batch = self.backend.to_backend_batch(numpy_batch)
                del numpy_batch
                self._validate_runners(
                    case=case,
                    runners=validation_runners,
                    batch=batch,
                    expected=expected,
                    order=round_orders[0],
                )
                del batch, expected
            del validation_runners

        runners: dict[LibraryName, Runner] = {
            library_name: runner_factories[library_name]()
            for library_name in available_libs
        }

        observations: list[LibraryTimingObservation] = []
        realized_units: list[DynamicInputUnit] = []
        measured_batches = config.batches - config.warmup_batches
        for round_index, round_order in enumerate(round_orders):
            for stream_index in range(config.warmup_batches):
                numpy_batch = self._make_dynamic_numpy_batch(
                    case_spec=case_spec,
                    seed=self._dynamic_batch_seed(
                        config=config,
                        case_index=case_index,
                        round_index=round_index,
                        stream_index=stream_index,
                    ),
                )
                batch = self.backend.to_backend_batch(numpy_batch)
                del numpy_batch
                self._warmup_coordinate(
                    runners=runners,
                    batch=batch,
                    order=self._rotate_order(
                        round_order,
                        offset=stream_index,
                    ),
                )
                del batch

            for repeat_index in range(config.repeats):
                for unit_index in range(measured_batches):
                    stream_index = config.warmup_batches + unit_index
                    batch_seed = self._dynamic_batch_seed(
                        config=config,
                        case_index=case_index,
                        round_index=round_index,
                        stream_index=stream_index,
                    )
                    numpy_batch = self._make_dynamic_numpy_batch(
                        case_spec=case_spec,
                        seed=batch_seed,
                    )
                    input_shapes = tuple(array.shape for array in numpy_batch)
                    if repeat_index == 0:
                        realized_units.append(
                            DynamicInputUnit(
                                round_index=round_index,
                                unit_index=unit_index,
                                stream_index=stream_index,
                                seed=batch_seed,
                                input_shapes=input_shapes,
                            )
                        )
                    batch = self.backend.to_backend_batch(numpy_batch)
                    del numpy_batch
                    observations.extend(
                        self._measure_coordinate(
                            runners=runners,
                            batch=batch,
                            order=self._rotate_order(
                                round_order,
                                offset=(
                                    repeat_index * measured_batches + unit_index
                                ),
                            ),
                            round_index=round_index,
                            unit_index=unit_index,
                            repeat_index=repeat_index,
                        ),
                    )
                    del batch

        for library_name in available_libs:
            runs[library_name] = self._available_run(
                observations=observations,
                library_name=library_name,
                rounds=config.rounds,
            )

        return DynamicCaseResult(
            case=case,
            workload=case_spec.workload_metadata,
            realized_units=tuple(realized_units),
            runs=runs,
            round_orders=round_orders,
            evidence=self._paired_evidence(
                observations=observations,
                libraries=available_libs,
                bootstrap_seed=config.seed + case_index * CASE_SEED_STRIDE,
            ),
        )
