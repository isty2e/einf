import json
from pathlib import Path
from typing import cast

import pytest

from benchmarks.guardrail.policy import (
    MetricName,
    OverheadReportDict,
    collect_case_metrics,
    compare_overhead_report_trials,
    compare_overhead_reports,
    load_overhead_report,
    render_findings,
    render_trial_findings,
)


def _report(*, call_ms: float) -> OverheadReportDict:
    return OverheadReportDict(
        meta={
            "backend": "numpy",
            "python": "3.11",
            "numpy": "1.26",
            "torch": "not-installed",
            "einops": "not-installed",
            "einx": "not-installed",
            "stages": ["__call__", "solve", "runner_resolve", "fusion", "kernel"],
        },
        scenarios=[
            {
                "scenario": "fixed_medium",
                "mode": "fixed",
                "scale": "medium",
                "cases": [
                    {
                        "name": "rearrange_flatten",
                        "call_repr": "rearrange(...)",
                        "loops": 100,
                        "unpatched_call_ms": call_ms,
                        "instrumented_call_ms": call_ms,
                        "residual_ms_per_call": call_ms * 0.1,
                        "stage_ms_per_call": {
                            "solve": call_ms * 0.1,
                            "runner_resolve": 0.0,
                            "fusion": 0.0,
                            "kernel": call_ms * 0.8,
                        },
                    }
                ],
            }
        ],
    )


def _set_metric(
    report: OverheadReportDict,
    *,
    metric_name: MetricName,
    value: object,
) -> None:
    case = report["scenarios"][0]["cases"][0]
    case[metric_name] = cast(float, value)


def _delete_metric(
    report: OverheadReportDict,
    *,
    metric_name: MetricName,
) -> None:
    case = report["scenarios"][0]["cases"][0]
    case_mapping = cast(dict[str, object], case)
    del case_mapping[metric_name]


def _write_report(
    tmp_path: Path,
    *,
    report: OverheadReportDict,
) -> Path:
    path = tmp_path / "overhead.json"
    path.write_text(json.dumps(report))
    return path


def test_collect_case_metrics_builds_stable_keys() -> None:
    metrics = collect_case_metrics(_report(call_ms=1.0))
    key = ("fixed_medium", "fixed", "medium", "rearrange_flatten")
    assert key in metrics
    assert metrics[key].instrumented_call_ms == 1.0


@pytest.mark.parametrize(
    "metric_name",
    ("unpatched_call_ms", "instrumented_call_ms"),
)
def test_load_overhead_report_requires_each_latency_metric(
    tmp_path: Path,
    metric_name: MetricName,
) -> None:
    report = _report(call_ms=1.0)
    _delete_metric(report, metric_name=metric_name)

    with pytest.raises(TypeError, match=f"missing required metric {metric_name}"):
        load_overhead_report(_write_report(tmp_path, report=report))


def test_load_overhead_report_rejects_boolean_latency(tmp_path: Path) -> None:
    report = _report(call_ms=1.0)
    _set_metric(
        report,
        metric_name="instrumented_call_ms",
        value=True,
    )

    with pytest.raises(TypeError, match="instrumented_call_ms must be numeric"):
        load_overhead_report(_write_report(tmp_path, report=report))


@pytest.mark.parametrize("report_side", ("baseline", "candidate"))
@pytest.mark.parametrize(
    "metric_name",
    ("unpatched_call_ms", "instrumented_call_ms"),
)
def test_compare_overhead_reports_requires_each_latency_metric(
    report_side: str,
    metric_name: MetricName,
) -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.0)
    invalid_report = baseline if report_side == "baseline" else candidate
    _delete_metric(invalid_report, metric_name=metric_name)

    with pytest.raises(TypeError, match=f"missing required metric {metric_name}"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize("report_side", ("baseline", "candidate"))
@pytest.mark.parametrize(
    "metric_name",
    ("unpatched_call_ms", "instrumented_call_ms"),
)
def test_compare_overhead_reports_rejects_boolean_latency(
    report_side: str,
    metric_name: MetricName,
) -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.0)
    invalid_report = baseline if report_side == "baseline" else candidate
    _set_metric(
        invalid_report,
        metric_name=metric_name,
        value=True,
    )

    with pytest.raises(TypeError, match=f"{metric_name} must be numeric"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize("report_side", ("baseline", "candidate"))
@pytest.mark.parametrize(
    "metric_name",
    ("unpatched_call_ms", "instrumented_call_ms"),
)
@pytest.mark.parametrize(
    ("invalid_value", "expected_message"),
    (
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (float("-inf"), "must be finite"),
        (10**1000, "must be finite"),
        (0.0, "must be > 0"),
        (-1.0, "must be > 0"),
    ),
)
def test_compare_overhead_reports_rejects_invalid_latency(
    report_side: str,
    metric_name: MetricName,
    invalid_value: float,
    expected_message: str,
) -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.0)
    invalid_report = baseline if report_side == "baseline" else candidate
    _set_metric(
        invalid_report,
        metric_name=metric_name,
        value=invalid_value,
    )

    with pytest.raises(ValueError, match=expected_message):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize("report_side", ("baseline", "candidate"))
def test_compare_overhead_reports_rejects_duplicate_case_keys(
    report_side: str,
) -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.0)
    duplicate_report = baseline if report_side == "baseline" else candidate
    duplicate_case = duplicate_report["scenarios"][0]["cases"][0].copy()
    duplicate_report["scenarios"][0]["cases"].append(duplicate_case)

    with pytest.raises(ValueError, match="duplicate overhead case key"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


def test_compare_overhead_reports_flags_regression() -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.2)
    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        fail_on_missing_cases=True,
    )
    assert len(regressions) == 1
    assert not missing_keys
    assert regressions[0].key == (
        "fixed_medium",
        "fixed",
        "medium",
        "rearrange_flatten",
    )


def test_compare_overhead_reports_allows_small_drift() -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.04)
    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric="instrumented_call_ms",
        max_regression_ratio=0.05,
        fail_on_missing_cases=True,
    )
    assert not regressions
    assert not missing_keys


def test_compare_overhead_reports_flags_missing_cases() -> None:
    baseline = _report(call_ms=1.0)
    candidate: OverheadReportDict = {
        "meta": baseline["meta"],
        "scenarios": [],
    }
    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric="instrumented_call_ms",
        max_regression_ratio=0.05,
        fail_on_missing_cases=True,
    )
    assert not regressions
    assert missing_keys == [("fixed_medium", "fixed", "medium", "rearrange_flatten")]
    text = render_findings(regressions=regressions, missing_keys=missing_keys)
    assert "Missing cases:" in text


def test_compare_overhead_report_trials_requires_repeated_regressions() -> None:
    first_baseline = _report(call_ms=1.0)
    first_candidate = _report(call_ms=1.2)
    second_baseline = _report(call_ms=1.0)
    second_candidate = _report(call_ms=1.03)

    regressions, missing_keys = compare_overhead_report_trials(
        report_pairs=(
            (first_baseline, first_candidate),
            (second_baseline, second_candidate),
        ),
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        min_regression_count=2,
        fail_on_missing_cases=True,
    )

    assert not regressions
    assert not missing_keys


def test_compare_overhead_report_trials_flags_repeated_regressions() -> None:
    regressions, missing_keys = compare_overhead_report_trials(
        report_pairs=(
            (_report(call_ms=1.0), _report(call_ms=1.2)),
            (_report(call_ms=2.0), _report(call_ms=2.4)),
        ),
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        min_regression_count=2,
        fail_on_missing_cases=True,
    )

    assert len(regressions) == 1
    assert not missing_keys
    assert regressions[0].key == (
        "fixed_medium",
        "fixed",
        "medium",
        "rearrange_flatten",
    )
    text = render_trial_findings(regressions=regressions, missing_keys=missing_keys)
    assert "Repeated regressions:" in text
    assert "failed 2/2 required trials" in text


def test_compare_overhead_report_trials_flags_missing_cases() -> None:
    empty_candidate: OverheadReportDict = {
        "meta": _report(call_ms=1.0)["meta"],
        "scenarios": [],
    }

    regressions, missing_keys = compare_overhead_report_trials(
        report_pairs=((_report(call_ms=1.0), empty_candidate),),
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        min_regression_count=1,
        fail_on_missing_cases=True,
    )

    assert not regressions
    assert missing_keys == [("fixed_medium", "fixed", "medium", "rearrange_flatten")]
    text = render_trial_findings(regressions=regressions, missing_keys=missing_keys)
    assert "Missing cases:" in text
