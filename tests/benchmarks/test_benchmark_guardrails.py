import json
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

import benchmarks.guardrail.check_overhead as check_overhead_module
import benchmarks.guardrail.check_overhead_trials as check_overhead_trials_module
from benchmarks.guardrail.policy import (
    OVERHEAD_REPORT_SCHEMA_VERSION,
    MetricName,
    OverheadCgroupCpuHierarchyDict,
    OverheadReportDict,
    RegressionFinding,
    collect_case_metrics,
    compare_overhead_report_trials,
    compare_overhead_reports,
    load_overhead_report,
    render_findings,
    render_trial_findings,
)

_ALIAS_ERROR = "baseline and candidate reports must use different files"
_TRIAL_ALIAS_ERROR = "trial reports must use distinct files"


def _report(
    *,
    call_ms: float,
    source_revision: str = "a" * 40,
    source_content_sha256: str | None = None,
    seed: int = 20260215,
) -> OverheadReportDict:
    stages = ["__call__", "solve", "runner_resolve", "fusion", "kernel"]
    return OverheadReportDict(
        schema_version=OVERHEAD_REPORT_SCHEMA_VERSION,
        meta={
            "capture_id": str(uuid4()),
            "harness_source_sha256": "0" * 64,
            "subject_source": {
                "kind": "git_checkout",
                "distribution_version": "0.2.0.dev1",
                "git_revision": source_revision,
                "git_dirty": False,
                "content_sha256": (
                    sha256(source_revision.encode()).hexdigest()
                    if source_content_sha256 is None
                    else source_content_sha256
                ),
            },
            "execution_target": {
                "backend": "numpy",
                "requested_device": "cpu",
                "resolved_device": "cpu",
            },
            "host": {
                "system": "Darwin",
                "release": "25.5.0",
                "machine": "arm64",
                "cpu_model": "Apple M1 Pro",
                "logical_cpu_count": 10,
            },
            "execution_resources": {
                "cpu_allocation": {
                    "process_cpu_affinity": None,
                    "cgroup_cpu_hierarchy": None,
                },
                "native_threadpools": [
                    {
                        "user_api": "blas",
                        "internal_api": "openblas",
                        "prefix": "libopenblas",
                        "num_threads": 10,
                        "version": "0.3.30",
                        "threading_layer": "openmp",
                        "architecture": "VORTEX",
                    }
                ],
            },
            "environment": {
                "python": {
                    "implementation_name": "cpython",
                    "implementation_version": "3.11.0-final.0",
                    "language_version": "3.11.0",
                    "build": "3.11.0 (test build)",
                    "cache_tag": "cpython-311",
                    "abi_flags": "",
                    "optimize": 0,
                    "debug": 0,
                    "py_debug": False,
                    "hash_seed": 0,
                    "hash_witness": (123, 456),
                },
                "numpy": "1.26",
                "array_api_compat": "1.12",
                "opt_einsum": "3.4",
            },
            "seed": seed,
            "stages": stages,
            "resolved_stage_targets": {
                stage: [f"einf.{stage}"] for stage in stages if stage != "__call__"
            },
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


def _replace_nested_value(
    report: OverheadReportDict,
    *,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    current: object = report
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list):
                raise TypeError("test path expected a list")
            current = cast(list[object], current)[part]
        else:
            if not isinstance(current, dict):
                raise TypeError("test path expected an object")
            current = cast(dict[str, object], current)[part]

    final_part = path[-1]
    if isinstance(final_part, int):
        if not isinstance(current, list):
            raise TypeError("test path expected a list")
        cast(list[object], current)[final_part] = value
    else:
        if not isinstance(current, dict):
            raise TypeError("test path expected an object")
        cast(dict[str, object], current)[final_part] = value


def _assert_single_guardrail_rejects_alias(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline: Path,
    candidate: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
    )

    with pytest.raises(ValueError, match=_ALIAS_ERROR):
        check_overhead_module.main()


def _write_invalid_report(path: Path) -> Path:
    path.write_text("{not-json", encoding="utf-8")
    return path


def test_single_guardrail_rejects_same_report_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _write_invalid_report(tmp_path / "report.json")

    _assert_single_guardrail_rejects_alias(
        monkeypatch,
        baseline=report,
        candidate=report,
    )


def test_single_guardrail_rejects_resolved_path_alias_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _write_invalid_report(tmp_path / "report.json")
    (tmp_path / "nested").mkdir()

    _assert_single_guardrail_rejects_alias(
        monkeypatch,
        baseline=report,
        candidate=tmp_path / "nested" / ".." / "report.json",
    )


def test_single_guardrail_rejects_symlink_alias_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _write_invalid_report(tmp_path / "report.json")
    alias = tmp_path / "report-symlink.json"
    try:
        alias.symlink_to(report)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    _assert_single_guardrail_rejects_alias(
        monkeypatch,
        baseline=report,
        candidate=alias,
    )


def test_single_guardrail_rejects_hardlink_alias_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _write_invalid_report(tmp_path / "report.json")
    alias = tmp_path / "report-hardlink.json"
    try:
        alias.hardlink_to(report)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")

    _assert_single_guardrail_rejects_alias(
        monkeypatch,
        baseline=report,
        candidate=alias,
    )


def test_single_guardrail_rejects_case_alias_on_insensitive_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _write_invalid_report(tmp_path / "report.JSON")
    alias = tmp_path / "report.json"
    try:
        same_entry = alias.samefile(report)
    except FileNotFoundError:
        same_entry = False
    if not same_entry:
        pytest.skip("requires a case-insensitive filesystem")

    _assert_single_guardrail_rejects_alias(
        monkeypatch,
        baseline=report,
        candidate=alias,
    )


def test_single_guardrail_allows_distinct_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report(call_ms=1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_report(call_ms=1.0)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
    )

    assert check_overhead_module.main() == 0


@pytest.mark.parametrize("identity_error_type", (OSError, FileNotFoundError))
def test_single_guardrail_fails_closed_when_identity_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity_error_type: type[OSError],
) -> None:
    report = _write_invalid_report(tmp_path / "report.json")
    alias = tmp_path / "report-hardlink.json"
    try:
        alias.hardlink_to(report)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")

    def raise_identity_error(_first: Path, _second: Path) -> bool:
        raise identity_error_type("simulated filesystem identity failure")

    monkeypatch.setattr(Path, "samefile", raise_identity_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead",
            "--baseline",
            str(report),
            "--candidate",
            str(alias),
        ],
    )

    with pytest.raises(
        identity_error_type,
        match="simulated filesystem identity failure",
    ):
        check_overhead_module.main()


def test_single_guardrail_reports_missing_normalized_path_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead",
            "--baseline",
            str(missing),
            "--candidate",
            str(nested / ".." / "missing.json"),
        ],
    )

    with pytest.raises(FileNotFoundError):
        check_overhead_module.main()


@pytest.mark.parametrize("parent_name", ("数据", "123"))
def test_single_guardrail_allows_case_distinct_reports_on_sensitive_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_name: str,
) -> None:
    report_directory = tmp_path / parent_name
    report_directory.mkdir()
    baseline = report_directory / "report.JSON"
    candidate = report_directory / "report.json"
    baseline.write_text(json.dumps(_report(call_ms=1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_report(call_ms=1.0)), encoding="utf-8")
    if baseline.samefile(candidate):
        pytest.skip("requires a case-sensitive filesystem")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
    )

    assert check_overhead_module.main() == 0


def test_trial_guardrail_rejects_alias_across_pair_roles_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared_report = _write_invalid_report(tmp_path / "shared.json")
    first_candidate = _write_invalid_report(tmp_path / "first-candidate.json")
    second_baseline = _write_invalid_report(tmp_path / "second-baseline.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead-trials",
            "--pair",
            str(shared_report),
            str(first_candidate),
            "--pair",
            str(second_baseline),
            str(shared_report),
        ],
    )

    with pytest.raises(ValueError, match=_TRIAL_ALIAS_ERROR):
        check_overhead_trials_module.main()


@pytest.mark.parametrize("reused_role", ("baseline", "candidate"))
def test_trial_guardrail_rejects_reused_report_within_role_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reused_role: str,
) -> None:
    first_baseline = _write_invalid_report(tmp_path / "first-baseline.json")
    first_candidate = _write_invalid_report(tmp_path / "first-candidate.json")
    second_baseline = (
        first_baseline
        if reused_role == "baseline"
        else _write_invalid_report(tmp_path / "second-baseline.json")
    )
    second_candidate = (
        first_candidate
        if reused_role == "candidate"
        else _write_invalid_report(tmp_path / "second-candidate.json")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead-trials",
            "--pair",
            str(first_baseline),
            str(first_candidate),
            "--pair",
            str(second_baseline),
            str(second_candidate),
        ],
    )

    with pytest.raises(ValueError, match=_TRIAL_ALIAS_ERROR):
        check_overhead_trials_module.main()


def test_trial_guardrail_allows_distinct_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_paths = tuple(tmp_path / f"report-{index}.json" for index in range(4))
    for report_path in report_paths:
        report_path.write_text(json.dumps(_report(call_ms=1.0)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-overhead-trials",
            "--pair",
            str(report_paths[0]),
            str(report_paths[1]),
            "--pair",
            str(report_paths[2]),
            str(report_paths[3]),
        ],
    )

    assert check_overhead_trials_module.main() == 0


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


def test_load_overhead_report_requires_schema_version(tmp_path: Path) -> None:
    report = _report(call_ms=1.0)
    del cast(dict[str, object], report)["schema_version"]

    with pytest.raises(TypeError, match="schema_version must be an integer"):
        load_overhead_report(_write_report(tmp_path, report=report))


def test_load_overhead_report_requires_execution_resources(tmp_path: Path) -> None:
    report = _report(call_ms=1.0)
    del cast(dict[str, object], report["meta"])["execution_resources"]

    with pytest.raises(TypeError, match="execution_resources"):
        load_overhead_report(_write_report(tmp_path, report=report))


def test_load_overhead_report_preserves_cgroup_control_hierarchy(
    tmp_path: Path,
) -> None:
    report = _report(call_ms=1.0)
    hierarchy = OverheadCgroupCpuHierarchyDict(
        version=2,
        child_to_root=[
            {"bandwidth_limit": None, "weight": 25},
            {
                "bandwidth_limit": {
                    "quota_us": 50_000,
                    "period_us": 100_000,
                    "burst_us": 0,
                },
                "weight": 100,
            },
            {
                "bandwidth_limit": {
                    "quota_us": 50_000,
                    "period_us": 100_000,
                    "burst_us": 0,
                },
                "weight": 100,
            },
        ],
    )
    report["meta"]["execution_resources"]["cpu_allocation"]["cgroup_cpu_hierarchy"] = (
        hierarchy
    )

    loaded = load_overhead_report(_write_report(tmp_path, report=report))

    assert (
        loaded["meta"]["execution_resources"]["cpu_allocation"]["cgroup_cpu_hierarchy"]
        == hierarchy
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            (
                "meta",
                "execution_resources",
                "cpu_allocation",
                "cgroup_cpu_hierarchy",
                "child_to_root",
            ),
            [],
            "must not be empty",
        ),
        (
            (
                "meta",
                "execution_resources",
                "cpu_allocation",
                "cgroup_cpu_hierarchy",
                "child_to_root",
                0,
                "weight",
            ),
            10_001,
            "weight is invalid",
        ),
        (
            (
                "meta",
                "execution_resources",
                "cpu_allocation",
                "cgroup_cpu_hierarchy",
                "child_to_root",
                0,
                "bandwidth_limit",
                "quota_us",
            ),
            0,
            "quota and period must be positive",
        ),
        (
            ("meta", "environment", "python", "hash_seed"),
            4_294_967_296,
            "outside the valid range",
        ),
        (
            ("meta", "environment", "python", "abi_flags"),
            None,
            "abi_flags must be a string",
        ),
        (
            ("meta", "environment", "python", "hash_witness"),
            [123],
            "hash_witness must contain two integers",
        ),
    ),
)
def test_load_overhead_report_rejects_invalid_runtime_fingerprint(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    report = _report(call_ms=1.0)
    report["meta"]["execution_resources"]["cpu_allocation"]["cgroup_cpu_hierarchy"] = {
        "version": 2,
        "child_to_root": [
            {
                "bandwidth_limit": {
                    "quota_us": 50_000,
                    "period_us": 100_000,
                    "burst_us": 0,
                },
                "weight": 100,
            }
        ],
    }
    _replace_nested_value(report, path=path, value=value)

    with pytest.raises((TypeError, ValueError), match=message):
        load_overhead_report(_write_report(tmp_path, report=report))


def test_load_overhead_report_rejects_unsupported_schema(tmp_path: Path) -> None:
    report = _report(call_ms=1.0)
    unsupported_version = OVERHEAD_REPORT_SCHEMA_VERSION + 1
    cast(dict[str, object], report)["schema_version"] = unsupported_version

    with pytest.raises(
        ValueError,
        match=f"unsupported schema_version {unsupported_version}",
    ):
        load_overhead_report(_write_report(tmp_path, report=report))


@pytest.mark.parametrize("field_name", ("bandwidth_limit", "weight"))
def test_load_overhead_report_requires_each_cgroup_level_field(
    tmp_path: Path,
    field_name: str,
) -> None:
    report = _report(call_ms=1.0)
    hierarchy = OverheadCgroupCpuHierarchyDict(
        version=2,
        child_to_root=[{"bandwidth_limit": None, "weight": 100}],
    )
    report["meta"]["execution_resources"]["cpu_allocation"]["cgroup_cpu_hierarchy"] = (
        hierarchy
    )
    del cast(dict[str, object], hierarchy["child_to_root"][0])[field_name]

    with pytest.raises(TypeError, match=f"{field_name} is required"):
        load_overhead_report(_write_report(tmp_path, report=report))


@pytest.mark.parametrize("field_name", ("call_repr", "loops"))
def test_load_overhead_report_requires_case_configuration(
    tmp_path: Path,
    field_name: str,
) -> None:
    report = _report(call_ms=1.0)
    case = cast(dict[str, object], report["scenarios"][0]["cases"][0])
    del case[field_name]

    with pytest.raises(TypeError, match=field_name):
        load_overhead_report(_write_report(tmp_path, report=report))


@pytest.mark.parametrize("loops", (True, 0, -1))
def test_load_overhead_report_rejects_invalid_loop_count(
    tmp_path: Path,
    loops: object,
) -> None:
    report = _report(call_ms=1.0)
    case = cast(dict[str, object], report["scenarios"][0]["cases"][0])
    case["loops"] = loops

    with pytest.raises((TypeError, ValueError), match="loops"):
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


def test_load_overhead_report_rejects_duplicate_case_keys(tmp_path: Path) -> None:
    report = _report(call_ms=1.0)
    duplicate_case = report["scenarios"][0]["cases"][0].copy()
    report["scenarios"][0]["cases"].append(duplicate_case)

    with pytest.raises(ValueError, match="duplicate overhead case key"):
        load_overhead_report(_write_report(tmp_path, report=report))


def test_regression_finding_requires_positive_baseline() -> None:
    with pytest.raises(ValueError, match="baseline_ms must be > 0"):
        RegressionFinding(
            key=("fixed_medium", "fixed", "medium", "rearrange_flatten"),
            metric="instrumented_call_ms",
            baseline_ms=0.0,
            candidate_ms=1.2,
            allowed_ms=1.1,
        )


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
    candidate_meta = deepcopy(baseline["meta"])
    candidate_meta["capture_id"] = str(uuid4())
    candidate: OverheadReportDict = {
        "schema_version": baseline["schema_version"],
        "meta": candidate_meta,
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
    complete_report = _report(call_ms=1.0)
    candidate_meta = deepcopy(complete_report["meta"])
    candidate_meta["capture_id"] = str(uuid4())
    empty_candidate: OverheadReportDict = {
        "schema_version": complete_report["schema_version"],
        "meta": candidate_meta,
        "scenarios": [],
    }

    regressions, missing_keys = compare_overhead_report_trials(
        report_pairs=((complete_report, empty_candidate),),
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        min_regression_count=1,
        fail_on_missing_cases=True,
    )

    assert not regressions
    assert missing_keys == [("fixed_medium", "fixed", "medium", "rearrange_flatten")]
    text = render_trial_findings(regressions=regressions, missing_keys=missing_keys)
    assert "Missing cases:" in text


def test_compare_overhead_reports_rejects_empty_baseline() -> None:
    complete_report = _report(call_ms=1.0)
    empty_baseline: OverheadReportDict = {
        "schema_version": complete_report["schema_version"],
        "meta": deepcopy(complete_report["meta"]),
        "scenarios": [],
    }

    with pytest.raises(ValueError, match="baseline.*at least one case"):
        compare_overhead_reports(
            baseline=empty_baseline,
            candidate=complete_report,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


def test_compare_overhead_reports_allows_different_subject_sources() -> None:
    baseline = _report(call_ms=1.0, source_revision="a" * 40)
    candidate = _report(call_ms=1.0, source_revision="b" * 40)

    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric="instrumented_call_ms",
        max_regression_ratio=0.10,
        fail_on_missing_cases=True,
    )

    assert not regressions
    assert not missing_keys


def test_compare_overhead_reports_rejects_duplicate_capture() -> None:
    baseline = _report(call_ms=1.0)
    candidate = _report(call_ms=1.0)
    candidate["meta"]["capture_id"] = baseline["meta"]["capture_id"]

    with pytest.raises(ValueError, match="distinct captures"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize(
    ("path", "value", "expected_axis"),
    (
        (("schema_version",), OVERHEAD_REPORT_SCHEMA_VERSION + 1, "schema"),
        (("meta", "harness_source_sha256"), "1" * 64, "harness_source"),
        (("meta", "execution_target", "backend"), "torch", "execution_target"),
        (
            ("meta", "execution_target", "resolved_device"),
            "cuda:0",
            "execution_target",
        ),
        (("meta", "host", "cpu_model"), "Apple M4 Max", "host"),
        (
            (
                "meta",
                "execution_resources",
                "cpu_allocation",
                "process_cpu_affinity",
            ),
            [0],
            "execution_resources",
        ),
        (
            (
                "meta",
                "execution_resources",
                "cpu_allocation",
                "cgroup_cpu_hierarchy",
            ),
            {
                "version": 2,
                "child_to_root": [
                    {"bandwidth_limit": None, "weight": 50},
                    {"bandwidth_limit": None, "weight": 100},
                ],
            },
            "execution_resources",
        ),
        (
            (
                "meta",
                "execution_resources",
                "native_threadpools",
                0,
                "num_threads",
            ),
            1,
            "execution_resources",
        ),
        (("meta", "environment", "numpy"), "2.0", "environment"),
        (
            ("meta", "environment", "python", "cache_tag"),
            "cpython-312",
            "environment",
        ),
        (
            ("meta", "environment", "python", "hash_witness"),
            (789, 1011),
            "environment",
        ),
        (("meta", "seed"), 7, "configuration"),
        (("meta", "stages"), ["__call__", "kernel"], "instrumentation"),
        (
            ("meta", "resolved_stage_targets", "kernel"),
            ["einf.different_kernel"],
            "instrumentation",
        ),
        (
            ("scenarios", 0, "cases", 0, "call_repr"),
            "different(...) ",
            "configuration",
        ),
        (("scenarios", 0, "cases", 0, "loops"), 101, "configuration"),
    ),
)
def test_compare_overhead_reports_rejects_incompatible_experiments_before_metrics(
    path: tuple[str | int, ...],
    value: object,
    expected_axis: str,
) -> None:
    baseline = _report(call_ms=1.0)
    candidate = deepcopy(baseline)
    candidate["meta"]["capture_id"] = str(uuid4())
    _replace_nested_value(candidate, path=path, value=value)
    _set_metric(
        baseline,
        metric_name="instrumented_call_ms",
        value=float("nan"),
    )

    with pytest.raises(ValueError, match=expected_axis):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


def test_compare_overhead_reports_preserves_cgroup_quota_positions() -> None:
    baseline = _report(call_ms=1.0)
    candidate = deepcopy(baseline)
    candidate["meta"]["capture_id"] = str(uuid4())
    baseline["meta"]["execution_resources"]["cpu_allocation"][
        "cgroup_cpu_hierarchy"
    ] = {
        "version": 2,
        "child_to_root": [
            {
                "bandwidth_limit": {
                    "quota_us": 50_000,
                    "period_us": 100_000,
                    "burst_us": 0,
                },
                "weight": 100,
            },
            {"bandwidth_limit": None, "weight": 100},
        ],
    }
    candidate["meta"]["execution_resources"]["cpu_allocation"][
        "cgroup_cpu_hierarchy"
    ] = {
        "version": 2,
        "child_to_root": [
            {"bandwidth_limit": None, "weight": 100},
            {
                "bandwidth_limit": {
                    "quota_us": 50_000,
                    "period_us": 100_000,
                    "burst_us": 0,
                },
                "weight": 100,
            },
        ],
    }
    _set_metric(
        baseline,
        metric_name="instrumented_call_ms",
        value=float("nan"),
    )

    with pytest.raises(ValueError, match="execution_resources"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


def test_unpatched_comparison_ignores_instrumentation_target_changes() -> None:
    baseline = _report(call_ms=1.0)
    candidate = deepcopy(baseline)
    candidate["meta"]["capture_id"] = str(uuid4())
    candidate["meta"]["resolved_stage_targets"]["kernel"] = ["einf.moved_kernel"]

    regressions, missing_keys = compare_overhead_reports(
        baseline=baseline,
        candidate=candidate,
        metric="unpatched_call_ms",
        max_regression_ratio=0.10,
        fail_on_missing_cases=True,
    )

    assert not regressions
    assert not missing_keys


def test_torch_comparison_rejects_effective_thread_count_changes() -> None:
    baseline = _report(call_ms=1.0)
    candidate = deepcopy(baseline)
    candidate["meta"]["capture_id"] = str(uuid4())
    for report in (baseline, candidate):
        report["meta"]["execution_target"]["backend"] = "torch"
        report["meta"]["environment"]["torch"] = "2.6"
        report["meta"]["execution_resources"]["torch_threads"] = {
            "intra_op": 8,
            "inter_op": 2,
        }
    candidate_torch_threads = candidate["meta"]["execution_resources"].get(
        "torch_threads"
    )
    assert candidate_torch_threads is not None
    candidate_torch_threads["intra_op"] = 4

    with pytest.raises(ValueError, match="execution_resources"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="unpatched_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


def test_torch_comparison_rejects_native_runtime_changes() -> None:
    baseline = _report(call_ms=1.0)
    candidate = deepcopy(baseline)
    candidate["meta"]["capture_id"] = str(uuid4())
    for report in (baseline, candidate):
        report["meta"]["execution_target"]["backend"] = "torch"
        report["meta"]["environment"]["torch"] = "2.6"
        report["meta"]["execution_resources"]["torch_threads"] = {
            "intra_op": 8,
            "inter_op": 2,
        }
    candidate["meta"]["execution_resources"]["native_threadpools"][0][
        "internal_api"
    ] = "mkl"

    with pytest.raises(ValueError, match="execution_resources"):
        compare_overhead_reports(
            baseline=baseline,
            candidate=candidate,
            metric="unpatched_call_ms",
            max_regression_ratio=0.10,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize("drifting_role", ("baseline", "candidate"))
def test_compare_overhead_report_trials_requires_stable_role_sources(
    drifting_role: str,
) -> None:
    first_baseline = _report(call_ms=1.0, source_revision="a" * 40)
    first_candidate = _report(call_ms=1.0, source_revision="b" * 40)
    second_baseline = _report(
        call_ms=1.0,
        source_revision="c" * 40 if drifting_role == "baseline" else "a" * 40,
    )
    second_candidate = _report(
        call_ms=1.0,
        source_revision="c" * 40 if drifting_role == "candidate" else "b" * 40,
    )

    with pytest.raises(ValueError, match=f"{drifting_role} trial reports"):
        compare_overhead_report_trials(
            report_pairs=(
                (first_baseline, first_candidate),
                (second_baseline, second_candidate),
            ),
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            min_regression_count=2,
            fail_on_missing_cases=True,
        )


def test_compare_overhead_report_trials_distinguishes_dirty_source_content() -> None:
    with pytest.raises(ValueError, match="baseline trial reports"):
        compare_overhead_report_trials(
            report_pairs=(
                (
                    _report(
                        call_ms=1.0,
                        source_revision="a" * 40,
                        source_content_sha256="1" * 64,
                    ),
                    _report(call_ms=1.0, source_revision="b" * 40),
                ),
                (
                    _report(
                        call_ms=1.0,
                        source_revision="a" * 40,
                        source_content_sha256="2" * 64,
                    ),
                    _report(call_ms=1.0, source_revision="b" * 40),
                ),
            ),
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            min_regression_count=2,
            fail_on_missing_cases=True,
        )


@pytest.mark.parametrize("drifting_role", ("baseline", "candidate"))
def test_compare_overhead_report_trials_requires_stable_case_coverage(
    drifting_role: str,
) -> None:
    first_baseline = _report(call_ms=1.0, source_revision="a" * 40)
    first_candidate = _report(call_ms=1.0, source_revision="b" * 40)
    second_baseline = _report(call_ms=1.0, source_revision="a" * 40)
    second_candidate = _report(call_ms=1.0, source_revision="b" * 40)
    drifting_report = (
        second_baseline if drifting_role == "baseline" else second_candidate
    )
    drifting_report["scenarios"] = []

    with pytest.raises(
        ValueError,
        match=f"{drifting_role} trial reports use different case configurations",
    ):
        compare_overhead_report_trials(
            report_pairs=(
                (first_baseline, first_candidate),
                (second_baseline, second_candidate),
            ),
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            min_regression_count=2,
            fail_on_missing_cases=True,
        )


def test_compare_overhead_report_trials_rejects_cross_trial_configuration_drift() -> (
    None
):
    with pytest.raises(ValueError, match="configuration"):
        compare_overhead_report_trials(
            report_pairs=(
                (
                    _report(call_ms=1.0, source_revision="a" * 40, seed=1),
                    _report(call_ms=1.0, source_revision="b" * 40, seed=1),
                ),
                (
                    _report(call_ms=1.0, source_revision="a" * 40, seed=2),
                    _report(call_ms=1.0, source_revision="b" * 40, seed=2),
                ),
            ),
            metric="instrumented_call_ms",
            max_regression_ratio=0.10,
            min_regression_count=2,
            fail_on_missing_cases=True,
        )
