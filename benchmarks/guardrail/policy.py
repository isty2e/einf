"""Overhead benchmark guardrail helpers.

This module compares raw JSON outputs emitted by
`benchmarks/profile/overhead_breakdown.py` and reports regressions.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

MetricName = Literal["unpatched_call_ms", "instrumented_call_ms"]


def _positive_finite_latency(
    value: object,
    *,
    name: str,
    context: str,
) -> float:
    """Normalize one finite, positive latency value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context}: {name} must be numeric")

    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{context}: {name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{context}: {name} must be finite")
    if normalized <= 0.0:
        raise ValueError(f"{context}: {name} must be > 0")
    return normalized


def _required_latency_metric(
    case: Mapping[str, object],
    *,
    metric_name: MetricName,
    context: str,
) -> float:
    """Return one required finite, positive latency metric."""
    if metric_name not in case:
        raise TypeError(f"{context}: missing required metric {metric_name}")

    return _positive_finite_latency(
        case[metric_name],
        name=metric_name,
        context=context,
    )


class OverheadCaseDict(TypedDict):
    """One case entry in overhead raw JSON."""

    name: str
    call_repr: str
    loops: int
    unpatched_call_ms: float
    instrumented_call_ms: float
    stage_ms_per_call: dict[str, float]
    residual_ms_per_call: NotRequired[float]


class OverheadScenarioDict(TypedDict):
    """One scenario entry in overhead raw JSON."""

    scenario: str
    mode: str
    scale: str
    cases: list[OverheadCaseDict]


class OverheadMetaDict(TypedDict):
    """Top-level metadata section in overhead raw JSON."""

    backend: str
    python: str
    numpy: str
    torch: str
    einops: str
    einx: str
    stages: list[str]


class OverheadReportDict(TypedDict):
    """Top-level overhead raw JSON payload."""

    meta: OverheadMetaDict
    scenarios: list[OverheadScenarioDict]


@dataclass(frozen=True, slots=True)
class CaseMetric:
    """Comparable metric snapshot for one scenario/case."""

    scenario: str
    mode: str
    scale: str
    case_name: str
    unpatched_call_ms: float
    instrumented_call_ms: float

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Stable case identity key."""
        return (self.scenario, self.mode, self.scale, self.case_name)

    def metric_value(self, metric_name: MetricName) -> float:
        """Return selected metric value."""
        if metric_name == "unpatched_call_ms":
            return self.unpatched_call_ms
        return self.instrumented_call_ms


@dataclass(frozen=True, slots=True)
class RegressionFinding:
    """One benchmark regression finding."""

    key: tuple[str, str, str, str]
    metric: MetricName
    baseline_ms: float
    candidate_ms: float
    allowed_ms: float

    def __post_init__(self) -> None:
        """Enforce the positive latency invariant used by ratio arithmetic."""
        context = f"invalid regression finding {self.key!r}"
        object.__setattr__(
            self,
            "baseline_ms",
            _positive_finite_latency(
                self.baseline_ms,
                name="baseline_ms",
                context=context,
            ),
        )
        object.__setattr__(
            self,
            "candidate_ms",
            _positive_finite_latency(
                self.candidate_ms,
                name="candidate_ms",
                context=context,
            ),
        )
        object.__setattr__(
            self,
            "allowed_ms",
            _positive_finite_latency(
                self.allowed_ms,
                name="allowed_ms",
                context=context,
            ),
        )

    @property
    def ratio(self) -> float:
        """Relative slowdown ratio over baseline."""
        return (self.candidate_ms / self.baseline_ms) - 1.0


@dataclass(frozen=True, slots=True)
class RepeatedRegressionFinding:
    """One regression repeated across enough benchmark trials to gate."""

    key: tuple[str, str, str, str]
    metric: MetricName
    required_count: int
    findings: tuple[RegressionFinding, ...]

    @property
    def count(self) -> int:
        """Number of trial pairs that reported this regression."""
        return len(self.findings)

    @property
    def worst_ratio(self) -> float:
        """Largest observed slowdown ratio across failing trials."""
        return max((finding.ratio for finding in self.findings), default=0.0)


def load_overhead_report(path: Path) -> OverheadReportDict:
    """Load one overhead raw JSON report with schema checks."""
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise TypeError(f"invalid overhead report at {path}: expected object root")

    meta_raw = loaded.get("meta")
    scenarios_raw = loaded.get("scenarios")
    if not isinstance(meta_raw, dict):
        raise TypeError(f"invalid overhead report at {path}: missing object meta")
    if not isinstance(scenarios_raw, list):
        raise TypeError(f"invalid overhead report at {path}: missing list scenarios")

    stages_raw = meta_raw.get("stages")
    if not isinstance(stages_raw, list) or not all(
        isinstance(stage_name, str) for stage_name in stages_raw
    ):
        raise TypeError(
            f"invalid overhead report at {path}: meta.stages must be list[str]"
        )

    meta: OverheadMetaDict = {
        "backend": str(meta_raw.get("backend", "")),
        "python": str(meta_raw.get("python", "")),
        "numpy": str(meta_raw.get("numpy", "")),
        "torch": str(meta_raw.get("torch", "")),
        "einops": str(meta_raw.get("einops", "")),
        "einx": str(meta_raw.get("einx", "")),
        "stages": [str(stage_name) for stage_name in stages_raw],
    }

    scenarios: list[OverheadScenarioDict] = []
    seen_case_keys: set[tuple[str, str, str, str]] = set()
    for scenario_raw in scenarios_raw:
        if not isinstance(scenario_raw, dict):
            raise TypeError(
                f"invalid overhead report at {path}: each scenario must be object"
            )
        scenario_name = str(scenario_raw.get("scenario", ""))
        mode = str(scenario_raw.get("mode", ""))
        scale = str(scenario_raw.get("scale", ""))
        cases_raw = scenario_raw.get("cases")
        if not isinstance(cases_raw, list):
            raise TypeError(
                f"invalid overhead report at {path}: scenario.cases must be list"
            )

        cases: list[OverheadCaseDict] = []
        for case_raw in cases_raw:
            if not isinstance(case_raw, dict):
                raise TypeError(
                    f"invalid overhead report at {path}: each case must be object"
                )
            case_name = str(case_raw.get("name", ""))
            case_key = (scenario_name, mode, scale, case_name)
            if case_key in seen_case_keys:
                raise ValueError(f"duplicate overhead case key: {case_key!r}")
            seen_case_keys.add(case_key)
            case_context = f"invalid overhead report at {path}: case {case_name!r}"
            stage_ms_raw = case_raw.get("stage_ms_per_call")
            if not isinstance(stage_ms_raw, dict):
                raise TypeError(
                    f"invalid overhead report at {path}: case.stage_ms_per_call must be object"
                )

            stage_ms: dict[str, float] = {}
            for stage_name, stage_value in stage_ms_raw.items():
                if not isinstance(stage_name, str):
                    raise TypeError(
                        f"invalid overhead report at {path}: stage name must be string"
                    )
                if isinstance(stage_value, bool) or not isinstance(
                    stage_value, (int, float)
                ):
                    raise TypeError(
                        f"invalid overhead report at {path}: stage value must be numeric"
                    )
                stage_ms[stage_name] = float(stage_value)

            case = OverheadCaseDict(
                name=case_name,
                call_repr=str(case_raw.get("call_repr", "")),
                loops=int(case_raw.get("loops", 0)),
                unpatched_call_ms=_required_latency_metric(
                    case_raw,
                    metric_name="unpatched_call_ms",
                    context=case_context,
                ),
                instrumented_call_ms=_required_latency_metric(
                    case_raw,
                    metric_name="instrumented_call_ms",
                    context=case_context,
                ),
                stage_ms_per_call=stage_ms,
                residual_ms_per_call=float(case_raw.get("residual_ms_per_call", 0.0)),
            )
            cases.append(case)

        scenario = OverheadScenarioDict(
            scenario=scenario_name,
            mode=mode,
            scale=scale,
            cases=cases,
        )
        scenarios.append(scenario)

    return OverheadReportDict(meta=meta, scenarios=scenarios)


def collect_case_metrics(
    report: OverheadReportDict,
) -> dict[tuple[str, str, str, str], CaseMetric]:
    """Collect case metrics keyed by (scenario, mode, scale, case_name)."""
    metrics: dict[tuple[str, str, str, str], CaseMetric] = {}
    for scenario in report["scenarios"]:
        scenario_name = scenario["scenario"]
        mode = scenario["mode"]
        scale = scenario["scale"]
        for case in scenario["cases"]:
            key = (scenario_name, mode, scale, case["name"])
            context = f"invalid overhead case {key!r}"
            metric = CaseMetric(
                scenario=scenario_name,
                mode=mode,
                scale=scale,
                case_name=case["name"],
                unpatched_call_ms=_required_latency_metric(
                    case,
                    metric_name="unpatched_call_ms",
                    context=context,
                ),
                instrumented_call_ms=_required_latency_metric(
                    case,
                    metric_name="instrumented_call_ms",
                    context=context,
                ),
            )
            if metric.key in metrics:
                raise ValueError(f"duplicate overhead case key: {metric.key!r}")
            metrics[metric.key] = metric
    return metrics


def compare_overhead_reports(
    *,
    baseline: OverheadReportDict,
    candidate: OverheadReportDict,
    metric: MetricName,
    max_regression_ratio: float,
    fail_on_missing_cases: bool,
) -> tuple[list[RegressionFinding], list[tuple[str, str, str, str]]]:
    """Compare baseline/candidate overhead reports and return regressions."""
    baseline_metrics = collect_case_metrics(baseline)
    candidate_metrics = collect_case_metrics(candidate)
    missing_keys: list[tuple[str, str, str, str]] = []
    findings: list[RegressionFinding] = []

    for key, baseline_case in baseline_metrics.items():
        candidate_case = candidate_metrics.get(key)
        if candidate_case is None:
            missing_keys.append(key)
            continue

        baseline_value = baseline_case.metric_value(metric)
        candidate_value = candidate_case.metric_value(metric)

        allowed_value = baseline_value * (1.0 + max_regression_ratio)
        if candidate_value > allowed_value:
            findings.append(
                RegressionFinding(
                    key=key,
                    metric=metric,
                    baseline_ms=baseline_value,
                    candidate_ms=candidate_value,
                    allowed_ms=allowed_value,
                )
            )

    if not fail_on_missing_cases:
        missing_keys = []

    return findings, missing_keys


def compare_overhead_report_trials(
    *,
    report_pairs: tuple[tuple[OverheadReportDict, OverheadReportDict], ...],
    metric: MetricName,
    max_regression_ratio: float,
    min_regression_count: int,
    fail_on_missing_cases: bool,
) -> tuple[list[RepeatedRegressionFinding], list[tuple[str, str, str, str]]]:
    """Compare repeated baseline/candidate trials and keep repeated regressions."""
    if min_regression_count < 1:
        raise ValueError("min_regression_count must be >= 1")
    if min_regression_count > len(report_pairs):
        raise ValueError("min_regression_count cannot exceed report pair count")

    findings_by_key: dict[tuple[str, str, str, str], list[RegressionFinding]] = {}
    missing_keys: set[tuple[str, str, str, str]] = set()
    for baseline, candidate in report_pairs:
        findings, missing = compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric=metric,
            max_regression_ratio=max_regression_ratio,
            fail_on_missing_cases=fail_on_missing_cases,
        )
        for finding in findings:
            findings_by_key.setdefault(finding.key, []).append(finding)
        missing_keys.update(missing)

    repeated = [
        RepeatedRegressionFinding(
            key=key,
            metric=metric,
            required_count=min_regression_count,
            findings=tuple(findings),
        )
        for key, findings in findings_by_key.items()
        if len(findings) >= min_regression_count
    ]
    repeated.sort(key=lambda finding: (*finding.key, finding.metric))
    return repeated, sorted(missing_keys)


def render_findings(
    *,
    regressions: list[RegressionFinding],
    missing_keys: list[tuple[str, str, str, str]],
) -> str:
    """Render guardrail findings in plain text."""
    lines: list[str] = []
    if not regressions and not missing_keys:
        return "No guardrail violations."

    if regressions:
        lines.append("Regressions:")
        for finding in regressions:
            scenario, mode, scale, case_name = finding.key
            lines.append(
                f"- {scenario}/{mode}/{scale}/{case_name}: "
                f"{finding.metric} {finding.candidate_ms:.6f}ms "
                f"> allowed {finding.allowed_ms:.6f}ms "
                f"(baseline {finding.baseline_ms:.6f}ms, "
                f"+{finding.ratio * 100.0:.2f}%)"
            )

    if missing_keys:
        lines.append("Missing cases:")
        for scenario, mode, scale, case_name in missing_keys:
            lines.append(f"- {scenario}/{mode}/{scale}/{case_name}")

    return "\n".join(lines)


def render_trial_findings(
    *,
    regressions: list[RepeatedRegressionFinding],
    missing_keys: list[tuple[str, str, str, str]],
) -> str:
    """Render repeated-trial guardrail findings in plain text."""
    lines: list[str] = []
    if not regressions and not missing_keys:
        return "No guardrail violations."

    if regressions:
        lines.append("Repeated regressions:")
        for finding in regressions:
            scenario, mode, scale, case_name = finding.key
            lines.append(
                f"- {scenario}/{mode}/{scale}/{case_name}: "
                f"{finding.metric} failed {finding.count}/{finding.required_count} "
                f"required trials (worst +{finding.worst_ratio * 100.0:.2f}%)"
            )
            for trial_index, trial_finding in enumerate(finding.findings, start=1):
                lines.append(
                    f"  trial {trial_index}: {trial_finding.candidate_ms:.6f}ms "
                    f"> allowed {trial_finding.allowed_ms:.6f}ms "
                    f"(baseline {trial_finding.baseline_ms:.6f}ms, "
                    f"+{trial_finding.ratio * 100.0:.2f}%)"
                )

    if missing_keys:
        lines.append("Missing cases:")
        for scenario, mode, scale, case_name in missing_keys:
            lines.append(f"- {scenario}/{mode}/{scale}/{case_name}")

    return "\n".join(lines)


__all__ = [
    "CaseMetric",
    "MetricName",
    "OverheadReportDict",
    "RegressionFinding",
    "RepeatedRegressionFinding",
    "collect_case_metrics",
    "compare_overhead_report_trials",
    "compare_overhead_reports",
    "load_overhead_report",
    "render_findings",
    "render_trial_findings",
]
