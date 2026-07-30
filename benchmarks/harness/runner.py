import time
from dataclasses import dataclass

import numpy as np

from .backend import BackendSpec
from .case import BenchmarkCase, DynamicCaseSpec, FixedCaseSpec, RunnerFactory
from .comparison import compare_paired_observations
from .config import DynamicTaskConfig, FixedTaskConfig
from .generator import TensorGenerator
from .profiler import Profiler
from .result import (
    DynamicCaseResult,
    DynamicObservation,
    DynamicRun,
    DynamicRunResult,
    FixedCaseResult,
    FixedRun,
    FixedRunResult,
    TimingSummary,
    UnavailableRun,
)
from .types import Array, LibraryName, Runner

CASE_SEED_STRIDE = 1009
ROUND_BATCH_SEED_STRIDE = 7919


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

    def _numpy_batch(self, batch: tuple[Array, ...]) -> tuple[np.ndarray, ...]:
        return tuple(self.backend.to_numpy_array(item) for item in batch)

    def _validate_output(
        self,
        *,
        case: BenchmarkCase,
        batch: tuple[Array, ...],
        output: Array | tuple[Array, ...],
        library_name: str,
    ) -> None:
        expected = self.backend.to_numpy_output(
            case.reference(self._numpy_batch(batch))
        )
        got = self.backend.to_numpy_output(output)
        if len(expected) != len(got):
            raise ValueError(
                f"{library_name}/{case.name} output arity mismatch: "
                f"expected {len(expected)}, got {len(got)}"
            )

        uses_torch_backend = any(self.backend.is_torch_tensor(item) for item in batch)
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

    def run_fixed_case(
        self,
        *,
        case_spec: FixedCaseSpec,
        config: FixedTaskConfig,
        order_seed: int,
    ) -> FixedCaseResult:
        """Run one fixed-shape case across libraries."""
        case = case_spec.case
        runners = self._libraries_for_case(case)
        inputs = case_spec.inputs

        runs: dict[LibraryName, FixedRunResult] = {}
        available_libs: list[LibraryName] = []
        for lib_name in ("einf", "einops", "einx"):
            is_available, reason = self.available[lib_name]
            if not is_available:
                runs[lib_name] = UnavailableRun(reason=reason)
                continue
            available_libs.append(lib_name)

        if not available_libs:
            return FixedCaseResult(case=case, runs=runs, round_orders=[])

        cold_samples: dict[LibraryName, list[float]] = {
            name: [] for name in available_libs
        }
        warm_samples: dict[LibraryName, list[float]] = {
            name: [] for name in available_libs
        }
        warm_rounds: dict[LibraryName, list[TimingSummary]] = {
            name: [] for name in available_libs
        }
        validated: set[LibraryName] = set()
        round_orders = self._round_orders(
            library_names=available_libs,
            rounds=config.rounds,
            seed=order_seed,
        )

        for order in round_orders:
            for repeat_index in range(config.cold_repeats):
                repeat_order = self._rotate_order(order, offset=repeat_index)
                for lib_name in repeat_order:
                    cold_inputs = self.backend.clone_batch(inputs)
                    started = time.perf_counter()
                    runner = runners[lib_name]()
                    cold_output = runner(cold_inputs)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    self.backend.touch_output(cold_output)
                    cold_samples[lib_name].append(elapsed_ms)
                    if lib_name not in validated:
                        self._validate_output(
                            case=case,
                            batch=cold_inputs,
                            output=cold_output,
                            library_name=lib_name,
                        )
                        validated.add(lib_name)

            warm_runners = {
                lib_name: runners[lib_name]() for lib_name in available_libs
            }
            warm_inputs = {
                lib_name: self.backend.clone_batch(inputs)
                for lib_name in available_libs
            }

            for warmup_index in range(config.warmup):
                warmup_order = self._rotate_order(order, offset=warmup_index)
                for lib_name in warmup_order:
                    self.backend.touch_output(
                        warm_runners[lib_name](warm_inputs[lib_name])
                    )

            round_warm_samples: dict[LibraryName, list[float]] = {
                lib_name: [] for lib_name in available_libs
            }
            for repeat_index in range(config.warm_repeats):
                elapsed_ms_by_library: dict[LibraryName, float] = {
                    lib_name: 0.0 for lib_name in available_libs
                }
                for iteration_index in range(config.warm_iterations):
                    iteration_order = self._rotate_order(
                        order,
                        offset=repeat_index * config.warm_iterations + iteration_index,
                    )
                    for lib_name in iteration_order:
                        started = time.perf_counter()
                        warm_output = warm_runners[lib_name](warm_inputs[lib_name])
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        self.backend.touch_output(warm_output)
                        elapsed_ms_by_library[lib_name] += elapsed_ms
                for lib_name, elapsed_ms in elapsed_ms_by_library.items():
                    elapsed_per_call_ms = elapsed_ms / float(config.warm_iterations)
                    warm_samples[lib_name].append(elapsed_per_call_ms)
                    round_warm_samples[lib_name].append(elapsed_per_call_ms)
            for lib_name in available_libs:
                warm_rounds[lib_name].append(
                    self.profiler.summarize(round_warm_samples[lib_name])
                )

        for lib_name in available_libs:
            runs[lib_name] = FixedRun(
                cold=self.profiler.summarize(cold_samples[lib_name]),
                warm=self.profiler.summarize(warm_samples[lib_name]),
                warm_rounds=tuple(warm_rounds[lib_name]),
            )

        return FixedCaseResult(
            case=case,
            runs=runs,
            round_orders=round_orders,
        )

    def _make_dynamic_batches(
        self,
        *,
        case_spec: DynamicCaseSpec,
        count: int,
        seed: int,
    ) -> list[tuple[Array, ...]]:
        generator = TensorGenerator.from_seed(backend=self.backend, seed=seed)
        return [case_spec.batch_factory(generator) for _ in range(count)]

    def run_dynamic_case(
        self,
        *,
        case_spec: DynamicCaseSpec,
        config: DynamicTaskConfig,
        case_index: int,
    ) -> DynamicCaseResult:
        """Run one dynamic-shape case across libraries."""
        case = case_spec.case
        runners = self._libraries_for_case(case)

        runs: dict[LibraryName, DynamicRunResult] = {}
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
                runs=runs,
                round_orders=[],
                observations=(),
                comparisons=(),
            )

        round_batches = [
            self._make_dynamic_batches(
                case_spec=case_spec,
                count=config.batches,
                seed=(
                    config.seed
                    + case_index * CASE_SEED_STRIDE
                    + round_index * ROUND_BATCH_SEED_STRIDE
                ),
            )
            for round_index in range(config.rounds)
        ]

        round_orders = self._round_orders(
            library_names=available_libs,
            rounds=config.rounds,
            seed=config.round_order_seed + case_index * CASE_SEED_STRIDE,
        )
        runner_by_library: dict[LibraryName, Runner] = {
            lib_name: runners[lib_name]() for lib_name in available_libs
        }

        reference_batches = round_batches[0]
        checks = min(len(reference_batches), config.parity_checks)
        for lib_name in available_libs:
            runner = runner_by_library[lib_name]
            for index in range(checks):
                batch = self.backend.clone_batch(reference_batches[index])
                output = runner(batch)
                self._validate_output(
                    case=case,
                    batch=batch,
                    output=output,
                    library_name=lib_name,
                )

        samples_by_library: dict[LibraryName, list[float]] = {
            lib_name: [] for lib_name in available_libs
        }
        dynamic_rounds: dict[LibraryName, list[TimingSummary]] = {
            lib_name: [] for lib_name in available_libs
        }
        for round_index, batches in enumerate(round_batches):
            warmup_slice = batches[: config.warmup_batches]
            measure_slice = batches[config.warmup_batches :]
            round_samples = {lib_name: [] for lib_name in available_libs}

            for batch_index, canonical_batch in enumerate(warmup_slice):
                batch_order = self._rotate_order(
                    round_orders[round_index],
                    offset=batch_index,
                )
                for lib_name in batch_order:
                    batch = self.backend.clone_batch(canonical_batch)
                    self.backend.touch_output(runner_by_library[lib_name](batch))

            for repeat_index in range(config.repeats):
                for batch_index, canonical_batch in enumerate(measure_slice):
                    batch_order = self._rotate_order(
                        round_orders[round_index],
                        offset=repeat_index * len(measure_slice) + batch_index,
                    )
                    for lib_name in batch_order:
                        batch = self.backend.clone_batch(canonical_batch)
                        started = time.perf_counter()
                        output = runner_by_library[lib_name](batch)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        self.backend.touch_output(output)
                        samples_by_library[lib_name].append(elapsed_ms)
                        round_samples[lib_name].append(elapsed_ms)

            for lib_name in available_libs:
                dynamic_rounds[lib_name].append(
                    self.profiler.summarize(round_samples[lib_name])
                )

        for lib_name in available_libs:
            runs[lib_name] = DynamicRun(
                summary=self.profiler.summarize(samples_by_library[lib_name]),
                round_summaries=tuple(dynamic_rounds[lib_name]),
            )

        measured_batch_count = config.batches - config.warmup_batches
        observations: list[DynamicObservation] = []
        sample_index = 0
        # Rebuild the deterministic schedule after timing to keep records out of
        # the measurement loop.
        for round_index, round_order in enumerate(round_orders):
            for repeat_index in range(config.repeats):
                for measured_batch_index in range(measured_batch_count):
                    batch_order = self._rotate_order(
                        round_order,
                        offset=(
                            repeat_index * measured_batch_count + measured_batch_index
                        ),
                    )
                    for order_position, lib_name in enumerate(batch_order):
                        observations.append(
                            DynamicObservation(
                                round_index=round_index,
                                measured_batch_index=measured_batch_index,
                                repeat_index=repeat_index,
                                library=lib_name,
                                order_position=order_position,
                                latency_ms=samples_by_library[lib_name][sample_index],
                            )
                        )
                    sample_index += 1

        observation_tuple = tuple(observations)
        comparisons = compare_paired_observations(
            observations=observation_tuple,
            libraries=tuple(available_libs),
            baseline="einf",
            bootstrap_seed=config.seed + case_index * CASE_SEED_STRIDE,
        )
        return DynamicCaseResult(
            case=case,
            runs=runs,
            round_orders=round_orders,
            observations=observation_tuple,
            comparisons=comparisons,
        )
