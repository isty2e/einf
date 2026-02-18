from benchmarks.guardrails import (
    OverheadReportDict,
    collect_case_metrics,
    compare_overhead_reports,
    render_findings,
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
            "stages": ["__call__", "solve", "kernel"],
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
                        "stage_ms_per_call": {
                            "solve": call_ms * 0.1,
                            "kernel": call_ms * 0.8,
                        },
                    }
                ],
            }
        ],
    )


def test_collect_case_metrics_builds_stable_keys() -> None:
    metrics = collect_case_metrics(_report(call_ms=1.0))
    key = ("fixed_medium", "fixed", "medium", "rearrange_flatten")
    assert key in metrics
    assert metrics[key].instrumented_call_ms == 1.0


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
