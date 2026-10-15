from collections.abc import Mapping, Sequence
from typing import Literal

from .backend import BackendSpec
from .result import DynamicInputUnit, PairedEvidence, TimingSummary


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
    expected_rounds: int,
    expected_units_per_round: int,
    expected_executions_per_member_per_coordinate: int,
    expected_members: Sequence[str],
    execution_identities_by_member: Mapping[str, Sequence[tuple[int, int, int]]],
) -> dict[str, object]:
    """Admit complete configured execution coverage before serialization.

    Parameters
    ----------
    coordinate_source : {"measurement_observations", "realized_input_units"}
        Receipt field identifying the measured input coordinates.
    expected_rounds : int
        Positive configured number of measured rounds.
    expected_units_per_round : int
        Positive configured number of measured units, excluding warmup.
    expected_executions_per_member_per_coordinate : int
        Positive configured number of executions for each member and unit.
    expected_members : Sequence[str]
        Available libraries or strategies, in receipt display order.
    execution_identities_by_member : Mapping[str, Sequence[tuple[int, int, int]]]
        Every observed execution, grouped by member, as round/unit/repeat triples.

    Returns
    -------
    dict[str, object]
        Coverage metadata; all-unavailable cases retain empty coverage.

    Raises
    ------
    ValueError
        If dimensions, membership or execution identities do not match the
        configured measurement grid. Duplicate executions are invalid.
    """
    for name, value in (
        ("rounds", expected_rounds),
        ("units per round", expected_units_per_round),
        ("executions per coordinate", expected_executions_per_member_per_coordinate),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"expected validation {name} must be a positive integer")

    member_set = frozenset(expected_members)
    if len(member_set) != len(expected_members):
        raise ValueError("expected validation members must be unique")
    if frozenset(execution_identities_by_member) != member_set:
        raise ValueError("validation members must match available members")

    coordinate_count = expected_rounds * expected_units_per_round if member_set else 0
    execution_count = coordinate_count * expected_executions_per_member_per_coordinate
    for member, identities in execution_identities_by_member.items():
        identity_set = frozenset(identities)
        if len(identity_set) != len(identities):
            raise ValueError(f"validation identities for {member!r} must be unique")
        for round_index, unit_index, repeat_index in identity_set:
            if any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in (round_index, unit_index, repeat_index)
            ) or not (
                0 <= round_index < expected_rounds
                and 0 <= unit_index < expected_units_per_round
                and 0 <= repeat_index < expected_executions_per_member_per_coordinate
            ):
                raise ValueError(
                    f"validation identities for {member!r} are outside benchmark configuration"
                )
        # Unique bounded triples with full Cartesian cardinality cover the grid.
        if len(identity_set) != execution_count:
            raise ValueError(
                f"validation identities for {member!r} do not cover every configured execution"
            )

    return {
        "scope": "all_measured_executions",
        "coordinate_source": coordinate_source,
        "coordinate_count": coordinate_count,
        "members": list(expected_members),
        "executions_per_member_per_coordinate": (
            expected_executions_per_member_per_coordinate if member_set else 0
        ),
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
