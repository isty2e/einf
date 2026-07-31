from collections import defaultdict
from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Protocol, TypeVar

import numpy as np

from .result import LibraryTimingObservation, PairedComparison
from .types import LibraryName

_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_CHUNK_SIZE = 256

_Coordinate = tuple[int, int, int]
_UnitCoordinate = tuple[int, int]
_ComparisonMemberT = TypeVar("_ComparisonMemberT", bound=str)


class _TimingObservation(Protocol):
    @property
    def round_index(self) -> int: ...

    @property
    def unit_index(self) -> int: ...

    @property
    def repeat_index(self) -> int: ...

    @property
    def latency_ms(self) -> float: ...


_TimingObservationT = TypeVar("_TimingObservationT", bound=_TimingObservation)


def _paired_unit_means(
    *,
    observations: Sequence[_TimingObservationT],
    members: tuple[_ComparisonMemberT, ...],
    member_of: Callable[[_TimingObservationT], _ComparisonMemberT],
) -> tuple[dict[_ComparisonMemberT, dict[_UnitCoordinate, float]], int]:
    expected_members = frozenset(members)
    latency_by_coordinate: dict[_Coordinate, dict[_ComparisonMemberT, float]] = {}

    for observation in observations:
        coordinate = (
            observation.round_index,
            observation.unit_index,
            observation.repeat_index,
        )
        member = member_of(observation)
        latency_by_member = latency_by_coordinate.setdefault(coordinate, {})
        if member in latency_by_member:
            raise ValueError(
                "duplicate paired timing observation for "
                f"coordinate={coordinate}, member={member}"
            )
        latency_by_member[member] = observation.latency_ms

    if not latency_by_coordinate:
        raise ValueError("paired comparison requires timing observations")

    latency_by_unit: dict[
        _UnitCoordinate, dict[_ComparisonMemberT, dict[int, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for coordinate, latency_by_member in latency_by_coordinate.items():
        actual_members = frozenset(latency_by_member)
        if actual_members != expected_members:
            missing = sorted(expected_members - actual_members)
            unexpected = sorted(actual_members - expected_members)
            raise ValueError(
                "incomplete paired timing observation for "
                f"coordinate={coordinate}: missing={missing}, "
                f"unexpected={unexpected}"
            )
        round_index, unit_index, repeat_index = coordinate
        unit_coordinate = (round_index, unit_index)
        for member in members:
            latency_by_unit[unit_coordinate][member][repeat_index] = latency_by_member[
                member
            ]

    unit_means: dict[_ComparisonMemberT, dict[_UnitCoordinate, float]] = {
        member: {} for member in members
    }
    expected_repeat_indices: frozenset[int] | None = None
    for unit_coordinate in sorted(latency_by_unit):
        latency_by_member = latency_by_unit[unit_coordinate]
        member_repeat_indices = {
            frozenset(latencies) for latencies in latency_by_member.values()
        }
        if len(member_repeat_indices) != 1:
            raise ValueError(
                f"paired timing repeat identities differ for unit={unit_coordinate}"
            )
        repeat_indices = member_repeat_indices.pop()
        if expected_repeat_indices is None:
            expected_repeat_indices = repeat_indices
        elif repeat_indices != expected_repeat_indices:
            raise ValueError(
                "paired timing repeat identities differ across measured units"
            )
        for member in members:
            unit_means[member][unit_coordinate] = fmean(
                latency_by_member[member][repeat_index]
                for repeat_index in sorted(repeat_indices)
            )

    if expected_repeat_indices is None:
        raise ValueError("paired comparison requires measured units")
    repeat_count = len(expected_repeat_indices)
    return unit_means, len(unit_means[members[0]]) * repeat_count


def _round_stratified_ratio_interval(
    *,
    baseline_by_unit: dict[_UnitCoordinate, float],
    competitor_by_unit: dict[_UnitCoordinate, float],
    seed: int,
) -> tuple[float, float]:
    values_by_round: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for unit_coordinate in sorted(baseline_by_unit):
        baseline_latency = baseline_by_unit[unit_coordinate]
        values_by_round[unit_coordinate[0]].append(
            (baseline_latency, competitor_by_unit[unit_coordinate])
        )

    round_arrays = tuple(
        np.asarray(values_by_round[round_index], dtype=np.float64)
        for round_index in sorted(values_by_round)
    )
    if any(len(round_values) < 2 for round_values in round_arrays):
        raise ValueError(
            "paired bootstrap requires at least two measured units per round"
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
            unit_count = len(round_values)
            indices = random_state.randint(
                0,
                unit_count,
                size=(chunk_size, unit_count),
            )
            baseline_sums += round_values[indices, 0].sum(axis=1)
            competitor_sums += round_values[indices, 1].sum(axis=1)
            observation_count += unit_count

        ratios[chunk_start:chunk_stop] = (
            competitor_sums / float(observation_count)
        ) / (baseline_sums / float(observation_count))

    tail_probability = (1.0 - _CONFIDENCE_LEVEL) / 2.0
    low, high = np.percentile(
        ratios,
        [tail_probability * 100.0, (1.0 - tail_probability) * 100.0],
    )
    return float(low), float(high)


def compare_paired_timings(
    *,
    observations: Sequence[_TimingObservationT],
    members: tuple[_ComparisonMemberT, ...],
    baseline: _ComparisonMemberT,
    member_of: Callable[[_TimingObservationT], _ComparisonMemberT],
    bootstrap_seed: int,
) -> tuple[PairedComparison[_ComparisonMemberT], ...]:
    """Compare timed members using paired resampling units."""
    if len(frozenset(members)) != len(members):
        raise ValueError("paired comparison members must be unique")
    if baseline not in members:
        return ()
    if len(members) < 2:
        return ()

    unit_means, call_pair_count = _paired_unit_means(
        observations=observations,
        members=members,
        member_of=member_of,
    )
    baseline_by_unit = unit_means[baseline]
    paired_unit_count = len(baseline_by_unit)
    baseline_mean = fmean(baseline_by_unit.values())

    comparisons: list[PairedComparison[_ComparisonMemberT]] = []
    competitors = tuple(member for member in members if member != baseline)
    for competitor_index, competitor in enumerate(competitors):
        competitor_by_unit = unit_means[competitor]
        competitor_mean = fmean(competitor_by_unit.values())
        comparison_seed = bootstrap_seed + competitor_index
        interval_low, interval_high = _round_stratified_ratio_interval(
            baseline_by_unit=baseline_by_unit,
            competitor_by_unit=competitor_by_unit,
            seed=comparison_seed,
        )
        comparisons.append(
            PairedComparison(
                baseline=baseline,
                competitor=competitor,
                call_pair_count=call_pair_count,
                paired_unit_count=paired_unit_count,
                latency_ratio=competitor_mean / baseline_mean,
                confidence_level=_CONFIDENCE_LEVEL,
                confidence_interval_low=interval_low,
                confidence_interval_high=interval_high,
                bootstrap_resamples=_BOOTSTRAP_RESAMPLES,
                bootstrap_seed=comparison_seed,
            )
        )
    return tuple(comparisons)


def compare_library_timings(
    *,
    observations: Sequence[LibraryTimingObservation],
    libraries: tuple[LibraryName, ...],
    baseline: LibraryName,
    bootstrap_seed: int,
) -> tuple[PairedComparison[LibraryName], ...]:
    """Compare available libraries using paired resampling units."""

    def library_of(observation: LibraryTimingObservation) -> LibraryName:
        return observation.library

    return compare_paired_timings(
        observations=observations,
        members=libraries,
        baseline=baseline,
        member_of=library_of,
        bootstrap_seed=bootstrap_seed,
    )
