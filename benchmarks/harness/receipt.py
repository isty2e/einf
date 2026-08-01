from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

from .backend import BackendSpec
from .result import DynamicInputUnit, PairedEvidence, TimingSummary

_ExecutionIdentity: TypeAlias = tuple[int, int, int]


def execution_target_payload(backend: BackendSpec) -> dict[str, str]:
    """Serialize the resolved benchmark execution target."""
    return {
        "backend": backend.name,
        "requested_device": backend.requested_device,
        "resolved_device": backend.resolved_device,
    }


def synchronized_measurement_contract_payload() -> dict[str, object]:
    """Serialize the shared synchronized-completion timing contract."""
    return {
        "phase": "steady",
        "clock": "host_perf_counter",
        "synchronization": "before_timer_start_and_before_timer_stop",
        "input_pairing": "same_prepared_tensor_objects_per_coordinate",
        "timed_region": [
            "library_call",
            "target_completion_wait",
        ],
        "excluded": [
            "runner_construction",
            "warmup",
            "input_preparation",
            "input_device_transfer",
            "output_validation",
            "output_device_to_host_transfer",
            "summary_and_serialization",
        ],
    }


def dynamic_input_units_payload(
    units: Sequence[DynamicInputUnit],
) -> list[dict[str, object]]:
    """Serialize exact dynamic input coordinates and their realized shapes."""
    return [
        {
            "round_index": unit.round_index,
            "unit_index": unit.unit_index,
            "stream_index": unit.stream_index,
            "seed": unit.seed,
            "input_shapes": [list(shape) for shape in unit.input_shapes],
        }
        for unit in units
    ]


def validation_coverage_payload(
    *,
    coordinate_source: Literal["measurement_observations", "realized_input_units"],
    coordinate_count: int,
    expected_executions_per_member_per_coordinate: int,
    execution_identities_by_member: Mapping[str, Sequence[_ExecutionIdentity]],
) -> dict[str, object]:
    """Serialize exhaustive timed-output validation coverage."""
    if coordinate_count < 0:
        raise ValueError("validation coordinate count must be non-negative")
    if expected_executions_per_member_per_coordinate < 1:
        raise ValueError("expected validation executions must be positive")
    if coordinate_count > 0 and not execution_identities_by_member:
        raise ValueError("validation coordinates require at least one member")

    expected_coordinates: frozenset[tuple[int, int]] | None = None
    expected_repeat_indices: frozenset[int] | None = None
    for member, identities in execution_identities_by_member.items():
        identity_set = frozenset(identities)
        if len(identity_set) != len(identities):
            raise ValueError(f"validation identities for {member!r} must be unique")
        coordinates = frozenset(
            (round_index, unit_index) for round_index, unit_index, _ in identity_set
        )
        if len(coordinates) != coordinate_count:
            raise ValueError(
                f"validation identities for {member!r} do not cover every coordinate"
            )
        repeats_by_coordinate: dict[tuple[int, int], set[int]] = {}
        for round_index, unit_index, repeat_index in identity_set:
            repeats_by_coordinate.setdefault((round_index, unit_index), set()).add(
                repeat_index
            )
        repeat_sets = frozenset(
            frozenset(repeat_indices)
            for repeat_indices in repeats_by_coordinate.values()
        )
        if len(repeat_sets) != 1:
            raise ValueError(
                f"validation identities for {member!r} have uneven repeat coverage"
            )
        repeat_indices = next(iter(repeat_sets), frozenset())
        if repeat_indices != frozenset(range(len(repeat_indices))):
            raise ValueError(
                f"validation identities for {member!r} have non-contiguous repeats"
            )
        if expected_coordinates is None:
            expected_coordinates = coordinates
            expected_repeat_indices = repeat_indices
        elif (
            coordinates != expected_coordinates
            or repeat_indices != expected_repeat_indices
        ):
            raise ValueError("validation coverage must match across members")

    executions_per_member_per_coordinate = len(expected_repeat_indices or ())
    if execution_identities_by_member and (
        executions_per_member_per_coordinate
        != expected_executions_per_member_per_coordinate
    ):
        raise ValueError(
            "validation execution coverage does not match benchmark configuration"
        )
    return {
        "scope": "all_measured_executions",
        "coordinate_source": coordinate_source,
        "coordinate_count": coordinate_count,
        "members": list(execution_identities_by_member),
        "executions_per_member_per_coordinate": (executions_per_member_per_coordinate),
        "phase": "immediately_after_each_timed_execution",
        "output_source": "timed_call_return_value",
        "output_check": "numerical_reference_match",
    }


def timing_summary_payload(summary: TimingSummary) -> dict[str, int | float]:
    """Serialize one timing summary."""
    return {
        "count": summary.count,
        "p25_ms": summary.p25_ms,
        "median_ms": summary.median_ms,
        "p75_ms": summary.p75_ms,
        "iqr_ms": summary.iqr_ms,
        "p95_ms": summary.p95_ms,
        "mean_ms": summary.mean_ms,
        "min_ms": summary.min_ms,
        "max_ms": summary.max_ms,
    }


def paired_evidence_payload(evidence: PairedEvidence) -> dict[str, object]:
    """Serialize one paired measurement phase."""
    return {
        "observations": [
            {
                "round_index": observation.round_index,
                "unit_index": observation.unit_index,
                "repeat_index": observation.repeat_index,
                "library": observation.library,
                "order_position": observation.order_position,
                "latency_ms": observation.latency_ms,
            }
            for observation in evidence.observations
        ],
        "comparisons": [
            {
                "baseline": comparison.baseline,
                "competitor": comparison.competitor,
                "call_pair_count": comparison.call_pair_count,
                "paired_unit_count": comparison.paired_unit_count,
                "latency_ratio": comparison.latency_ratio,
                "confidence_level": comparison.confidence_level,
                "confidence_interval": {
                    "low": comparison.confidence_interval_low,
                    "high": comparison.confidence_interval_high,
                },
                "bootstrap_resamples": comparison.bootstrap_resamples,
                "bootstrap_seed": comparison.bootstrap_seed,
            }
            for comparison in evidence.comparisons
        ],
    }
