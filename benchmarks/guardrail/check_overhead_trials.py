#!/usr/bin/env python3
"""Compare repeated overhead raw JSON pairs and fail on repeated regressions."""

import argparse
from pathlib import Path

from benchmarks.guardrail.policy import (
    MetricName,
    compare_overhead_report_trials,
    load_overhead_report,
    render_trial_findings,
)


def main() -> int:
    """Run repeated-trial overhead guardrail comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare repeated baseline/candidate overhead reports and fail only "
            "when the same case regresses across enough trial pairs."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        type=Path,
        required=True,
        help="Baseline and candidate raw JSON reports for one trial pair.",
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
        "--min-regression-count",
        type=int,
        default=None,
        help=(
            "Number of trial pairs where the same case must regress before "
            "failing. Defaults to all supplied pairs."
        ),
    )
    parser.add_argument(
        "--allow-missing-cases",
        action="store_true",
        help="Do not fail when baseline keys are missing in candidate reports.",
    )
    args = parser.parse_args()

    if args.max_regression_ratio < 0.0:
        raise ValueError("max-regression-ratio must be >= 0.0")

    report_pairs = tuple(
        (load_overhead_report(baseline), load_overhead_report(candidate))
        for baseline, candidate in args.pair
    )
    min_regression_count = (
        len(report_pairs)
        if args.min_regression_count is None
        else args.min_regression_count
    )
    if args.metric == "instrumented_call_ms":
        metric_name: MetricName = "instrumented_call_ms"
    else:
        metric_name = "unpatched_call_ms"

    regressions, missing_keys = compare_overhead_report_trials(
        report_pairs=report_pairs,
        metric=metric_name,
        max_regression_ratio=args.max_regression_ratio,
        min_regression_count=min_regression_count,
        fail_on_missing_cases=not args.allow_missing_cases,
    )
    findings_text = render_trial_findings(
        regressions=regressions,
        missing_keys=missing_keys,
    )
    print(findings_text)
    if regressions or missing_keys:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
