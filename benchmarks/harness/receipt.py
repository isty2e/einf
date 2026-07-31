from pathlib import Path

from .backend import BackendSpec
from .result import PairedEvidence, TimingSummary


def _alternate_case(path: Path) -> Path | None:
    name = path.name
    for index, character in enumerate(name):
        if character.isalpha():
            replacement = (
                character.upper() if character.islower() else character.lower()
            )
            return path.with_name(name[:index] + replacement + name[index + 1 :])
    return None


def _filesystem_is_case_insensitive(path: Path) -> bool:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    for ancestor in (candidate, *candidate.parents):
        alternate = _alternate_case(ancestor)
        if alternate is None:
            continue
        try:
            return alternate.exists() and alternate.samefile(ancestor)
        except OSError:
            return False
    return False


def _paths_alias(first: Path, second: Path) -> bool:
    resolved_first = first.expanduser().resolve(strict=False)
    resolved_second = second.expanduser().resolve(strict=False)
    if resolved_first == resolved_second:
        return True
    try:
        if (
            resolved_first.exists()
            and resolved_second.exists()
            and resolved_first.samefile(resolved_second)
        ):
            return True
    except OSError:
        pass
    return str(resolved_first).casefold() == str(
        resolved_second
    ).casefold() and _filesystem_is_case_insensitive(resolved_first.parent)


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


def resolve_raw_output_path(
    *,
    output: Path | None,
    raw_output: Path | None,
) -> Path | None:
    """Resolve an explicit or report-derived raw receipt path."""
    if raw_output is not None:
        if output is not None and _paths_alias(raw_output, output):
            raise ValueError("--output and --raw-output must use different paths")
        return raw_output
    if output is None:
        return None
    derived_path = output.with_suffix(".json")
    if _paths_alias(derived_path, output):
        raise ValueError(
            "--output must have a non-JSON suffix when --raw-output is omitted"
        )
    return derived_path


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
