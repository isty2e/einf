#!/usr/bin/env python3
"""Compare two overhead raw JSON reports and fail on regressions."""

import argparse
from pathlib import Path

from benchmarks.guardrail.policy import (
    MetricName,
    compare_overhead_reports,
    load_overhead_report,
    render_findings,
)


def main() -> int:
    """Run guardrail comparison for overhead benchmark reports."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline/candidate overhead reports and fail on configured regressions."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline raw JSON report.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate raw JSON report.",
    )
    parser.add_argument(
        "--metric",
        choices=("instrumented_call_ms", "unpatched_call_ms"),
        default="instrumented_call_ms",
        help="Latency metric used for regression checks.",
    )
    parser.add_argument(
        "--max-regression-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed relative slowdown ratio (e.g. 0.10 == +10%%).",
    )
    parser.add_argument(
        "--allow-missing-cases",
        action="store_true",
        help="Do not fail when baseline keys are missing in candidate report.",
    )
    args = parser.parse_args()

    if args.max_regression_ratio < 0.0:
        raise ValueError("max-regression-ratio must be >= 0.0")

    baseline = load_overhead_report(args.baseline)
    candidate = load_overhead_report(args.candidate)
    if args.metric == "instrumented_call_ms":
        metric_name: MetricName = "instrumented_call_ms"
    else:
        metric_name = "unpatched_call_ms"
    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric=metric_name,
        max_regression_ratio=args.max_regression_ratio,
        fail_on_missing_cases=not args.allow_missing_cases,
    )
    findings_text = render_findings(
        regressions=regressions,
        missing_keys=missing_keys,
    )
    print(findings_text)
    if regressions or missing_keys:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
