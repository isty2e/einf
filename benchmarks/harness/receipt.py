from collections.abc import Sequence

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
            "parity_validation",
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
    coordinate_count: int,
    members: Sequence[str],
) -> dict[str, object]:
    """Serialize exhaustive out-of-timer numerical validation coverage."""
    return {
        "scope": "all_distinct_measured_coordinates",
        "coordinate_source": "realized_input_units",
        "coordinate_count": coordinate_count,
        "members": list(members),
        "executions_per_member_per_coordinate": 1,
        "phase": "after_measurement_before_receipt",
        "input_replay": "deterministic_seed_regeneration",
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
