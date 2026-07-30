from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

import numpy as np

from .result import DynamicObservation, PairedComparison
from .types import LibraryName

_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_CHUNK_SIZE = 256

_Coordinate = tuple[int, int, int]
_BatchCoordinate = tuple[int, int]


def _paired_batch_means(
    *,
    observations: Sequence[DynamicObservation],
    libraries: tuple[LibraryName, ...],
) -> tuple[dict[LibraryName, dict[_BatchCoordinate, float]], int]:
    expected_libraries = frozenset(libraries)
    latency_by_coordinate: dict[_Coordinate, dict[LibraryName, float]] = {}

    for observation in observations:
        coordinate = (
            observation.round_index,
            observation.measured_batch_index,
            observation.repeat_index,
        )
        latency_by_library = latency_by_coordinate.setdefault(coordinate, {})
        if observation.library in latency_by_library:
            raise ValueError(
                "duplicate dynamic observation for "
                f"coordinate={coordinate}, library={observation.library}"
            )
        latency_by_library[observation.library] = observation.latency_ms

    if not latency_by_coordinate:
        raise ValueError("paired comparison requires dynamic observations")

    latency_by_batch: dict[_BatchCoordinate, dict[LibraryName, dict[int, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for coordinate, latency_by_library in latency_by_coordinate.items():
        actual_libraries = frozenset(latency_by_library)
        if actual_libraries != expected_libraries:
            missing = sorted(expected_libraries - actual_libraries)
            unexpected = sorted(actual_libraries - expected_libraries)
            raise ValueError(
                "incomplete dynamic observation pairing for "
                f"coordinate={coordinate}: missing={missing}, "
                f"unexpected={unexpected}"
            )
        round_index, measured_batch_index, repeat_index = coordinate
        batch_coordinate = (round_index, measured_batch_index)
        for library in libraries:
            latency_by_batch[batch_coordinate][library][repeat_index] = (
                latency_by_library[library]
            )

    batch_means: dict[LibraryName, dict[_BatchCoordinate, float]] = {
        library: {} for library in libraries
    }
    expected_repeat_indices: frozenset[int] | None = None
    for batch_coordinate in sorted(latency_by_batch):
        latency_by_library = latency_by_batch[batch_coordinate]
        library_repeat_indices = {
            frozenset(latencies) for latencies in latency_by_library.values()
        }
        if len(library_repeat_indices) != 1:
            raise ValueError(
                "dynamic observation repeat identities differ for "
                f"batch={batch_coordinate}"
            )
        repeat_indices = library_repeat_indices.pop()
        if expected_repeat_indices is None:
            expected_repeat_indices = repeat_indices
        elif repeat_indices != expected_repeat_indices:
            raise ValueError(
                "dynamic observation repeat identities differ across measured batches"
            )
        for library in libraries:
            batch_means[library][batch_coordinate] = fmean(
                latency_by_library[library][repeat_index]
                for repeat_index in sorted(repeat_indices)
            )

    if expected_repeat_indices is None:
        raise ValueError("paired comparison requires measured batch observations")
    repeat_count = len(expected_repeat_indices)
    return batch_means, len(batch_means[libraries[0]]) * repeat_count


def _round_stratified_ratio_interval(
    *,
    baseline_by_batch: dict[_BatchCoordinate, float],
    competitor_by_batch: dict[_BatchCoordinate, float],
    seed: int,
) -> tuple[float, float]:
    values_by_round: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for batch_coordinate in sorted(baseline_by_batch):
        baseline_latency = baseline_by_batch[batch_coordinate]
        values_by_round[batch_coordinate[0]].append(
            (baseline_latency, competitor_by_batch[batch_coordinate])
        )

    round_arrays = tuple(
        np.asarray(values_by_round[round_index], dtype=np.float64)
        for round_index in sorted(values_by_round)
    )
    if any(len(round_values) < 2 for round_values in round_arrays):
        raise ValueError(
            "paired bootstrap requires at least two measured batches per round"
        )
    random_state = np.random.RandomState(seed)
    ratios = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)

    for chunk_start in range(0, _BOOTSTRAP_RESAMPLES, _BOOTSTRAP_CHUNK_SIZE):
        chunk_stop = min(
            chunk_start + _BOOTSTRAP_CHUNK_SIZE,
            _BOOTSTRAP_RESAMPLES,
        )
        chunk_size = chunk_stop - chunk_start
        baseline_sums = np.zeros(chunk_size, dtype=np.float64)
        competitor_sums = np.zeros(chunk_size, dtype=np.float64)
        observation_count = 0

        for round_values in round_arrays:
            batch_count = len(round_values)
            indices = random_state.randint(
                0,
                batch_count,
                size=(chunk_size, batch_count),
            )
            baseline_sums += round_values[indices, 0].sum(axis=1)
            competitor_sums += round_values[indices, 1].sum(axis=1)
            observation_count += batch_count

        ratios[chunk_start:chunk_stop] = (
            competitor_sums / float(observation_count)
        ) / (baseline_sums / float(observation_count))

    tail_probability = (1.0 - _CONFIDENCE_LEVEL) / 2.0
    low, high = np.percentile(
        ratios,
        [tail_probability * 100.0, (1.0 - tail_probability) * 100.0],
    )
    return float(low), float(high)


def compare_paired_observations(
    *,
    observations: Sequence[DynamicObservation],
    libraries: tuple[LibraryName, ...],
    baseline: LibraryName,
    bootstrap_seed: int,
) -> tuple[PairedComparison, ...]:
    """Compare available libraries using paired batch resampling units."""
    if baseline not in libraries:
        return ()
    if len(libraries) < 2:
        return ()

    batch_means, call_pair_count = _paired_batch_means(
        observations=observations,
        libraries=libraries,
    )
    baseline_by_batch = batch_means[baseline]
    paired_batch_count = len(baseline_by_batch)
    baseline_mean = fmean(baseline_by_batch.values())

    comparisons: list[PairedComparison] = []
    competitors: tuple[LibraryName, ...] = tuple(
        library for library in libraries if library != baseline
    )
    for competitor_index, competitor in enumerate(competitors):
        competitor_by_batch = batch_means[competitor]
        competitor_mean = fmean(competitor_by_batch.values())
        comparison_seed = bootstrap_seed + competitor_index
        interval_low, interval_high = _round_stratified_ratio_interval(
            baseline_by_batch=baseline_by_batch,
            competitor_by_batch=competitor_by_batch,
            seed=comparison_seed,
        )
        comparisons.append(
            PairedComparison(
                baseline=baseline,
                competitor=competitor,
                call_pair_count=call_pair_count,
                paired_batch_count=paired_batch_count,
                latency_ratio=competitor_mean / baseline_mean,
                confidence_level=_CONFIDENCE_LEVEL,
                confidence_interval_low=interval_low,
                confidence_interval_high=interval_high,
                bootstrap_resamples=_BOOTSTRAP_RESAMPLES,
                bootstrap_seed=comparison_seed,
            )
        )
    return tuple(comparisons)
