import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.guardrail.policy import (
    OVERHEAD_REPORT_SCHEMA_VERSION,
    OverheadCpuAllocationDict,
    OverheadExecutionResourcesDict,
    OverheadPythonRuntimeDict,
    load_overhead_report,
)
from benchmarks.profile import overhead_breakdown
from benchmarks.profile.overhead_breakdown import (
    STAGE_TARGETS,
    STAGES,
    CaseResult,
    OverheadCase,
    ScenarioResult,
    _profile_case,
    _resolve_target,
    _to_json,
)


def _resolved_targets() -> dict[str, tuple[str, ...]]:
    return {stage: (f"einf.{stage}",) for stage in STAGES if stage != "__call__"}


def _execution_resources() -> OverheadExecutionResourcesDict:
    return {
        "cpu_allocation": {
            "process_cpu_affinity": None,
            "cgroup_cpu_bandwidth_limits": [],
            "cgroup_cpu_weight_hierarchy": None,
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
    }


def _python_runtime() -> OverheadPythonRuntimeDict:
    return {
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
    }


def test_overhead_stage_targets_resolve() -> None:
    for targets in STAGE_TARGETS.values():
        for target in targets:
            _resolve_target(target)


def test_expected_einf_source_root_is_checked_before_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "overhead_breakdown",
            "--expect-einf-source-root",
            str(tmp_path),
        ],
    )

    def reject_source(checkout_root: Path) -> None:
        assert checkout_root == tmp_path
        raise RuntimeError("wrong source")

    monkeypatch.setattr(
        overhead_breakdown,
        "require_einf_source_root",
        reject_source,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_resolved_stage_target_names",
        lambda: pytest.fail("profiling preflight must not run"),
    )

    with pytest.raises(RuntimeError, match="wrong source"):
        overhead_breakdown.main()


def test_receipt_sources_must_remain_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def require_subject_content(expected: str) -> None:
        if expected != "subject-after":
            raise RuntimeError("imported einf source changed during measurement")

    monkeypatch.setattr(
        overhead_breakdown,
        "_harness_source_sha256",
        lambda: "harness-after",
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "require_stable_einf_source_content",
        require_subject_content,
    )

    overhead_breakdown._require_stable_receipt_sources(
        harness_source_sha256="harness-after",
        subject_content_sha256="subject-after",
    )
    with pytest.raises(RuntimeError, match="profiler source changed"):
        overhead_breakdown._require_stable_receipt_sources(
            harness_source_sha256="harness-before",
            subject_content_sha256="subject-after",
        )
    with pytest.raises(RuntimeError, match="einf source changed"):
        overhead_breakdown._require_stable_receipt_sources(
            harness_source_sha256="harness-after",
            subject_content_sha256="subject-before",
        )


def test_receipt_requires_fixed_hash_seed_before_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["overhead_breakdown", "--receipt", str(tmp_path / "report.json")],
    )
    monkeypatch.setattr(overhead_breakdown, "_resolved_stage_target_names", dict)
    monkeypatch.setattr(overhead_breakdown, "_harness_source_sha256", lambda: "0" * 64)
    monkeypatch.setattr(
        overhead_breakdown,
        "_subject_source_metadata",
        lambda: {
            "kind": "git_checkout",
            "distribution_version": "0.2.0.dev1",
            "git_revision": "a" * 40,
            "git_dirty": False,
            "content_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_cpu_allocation_metadata",
        lambda: _execution_resources()["cpu_allocation"],
    )

    def execution_resources(
        _backend: object,
        *,
        cpu_allocation: OverheadCpuAllocationDict,
    ) -> OverheadExecutionResourcesDict:
        assert cpu_allocation == _execution_resources()["cpu_allocation"]
        return _execution_resources()

    monkeypatch.setattr(
        overhead_breakdown,
        "_execution_resources_metadata",
        execution_resources,
    )

    def reject_uncontrolled_seed() -> OverheadPythonRuntimeDict:
        raise RuntimeError("receipt capture requires a fixed PYTHONHASHSEED")

    monkeypatch.setattr(
        overhead_breakdown,
        "_python_runtime_metadata",
        reject_uncontrolled_seed,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_profile_scenario",
        lambda **_kwargs: pytest.fail("profiling must not start"),
    )

    with pytest.raises(RuntimeError, match="requires a fixed PYTHONHASHSEED"):
        overhead_breakdown.main()


def test_execution_resources_capture_affinity_and_effective_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overhead_breakdown,
        "_process_cpu_affinity",
        lambda: [1, 3],
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "threadpool_info",
        lambda: [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "num_threads": 2,
                "version": "0.3.30",
                "threading_layer": "openmp",
                "architecture": "VORTEX",
            },
            {
                "user_api": "openmp",
                "internal_api": "openmp",
                "prefix": "libomp",
                "num_threads": 8,
                "version": None,
                "threading_layer": None,
                "architecture": None,
            },
        ],
    )

    class Torch:
        @staticmethod
        def get_num_threads() -> int:
            return 4

        @staticmethod
        def get_num_interop_threads() -> int:
            return 2

    monkeypatch.setattr(overhead_breakdown, "torch", Torch)

    cpu_allocation = OverheadCpuAllocationDict(
        process_cpu_affinity=[1, 3],
        cgroup_cpu_bandwidth_limits=[],
        cgroup_cpu_weight_hierarchy=None,
    )
    resources = overhead_breakdown._execution_resources_metadata(
        "torch",
        cpu_allocation=cpu_allocation,
    )
    numpy_resources = overhead_breakdown._execution_resources_metadata(
        "numpy",
        cpu_allocation=cpu_allocation,
    )

    assert resources["cpu_allocation"]["process_cpu_affinity"] == [1, 3]
    assert resources["native_threadpools"] == []
    assert resources.get("torch_threads") == {"intra_op": 4, "inter_op": 2}
    assert [pool["user_api"] for pool in numpy_resources["native_threadpools"]] == [
        "blas"
    ]
    assert "torch_threads" not in numpy_resources


def test_cgroup_v2_controls_include_ancestor_quota_and_ordered_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "self.cgroup"
    membership_path.write_text("0::/parent/child\n")
    mount_point = tmp_path / "cgroup2"
    child = mount_point / "parent" / "child"
    child.mkdir(parents=True)
    mountinfo_path = tmp_path / "mountinfo"
    mountinfo_path.write_text(
        f"36 25 0:32 / {mount_point} rw,nosuid,nodev,noexec,relatime - "
        "cgroup2 cgroup rw\n"
    )
    for directory, cpu_max, burst, weight in (
        (mount_point, "max 100000", "0", None),
        (mount_point / "parent", "200000 100000", "10000", "100"),
        (child, "max 50000", "0", "100"),
    ):
        directory.mkdir(exist_ok=True)
        (directory / "cpu.max").write_text(cpu_max)
        (directory / "cpu.max.burst").write_text(burst)
        if weight is not None:
            (directory / "cpu.weight").write_text(weight)

    monkeypatch.setattr(overhead_breakdown.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MEMBERSHIP_PATH",
        membership_path,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MOUNTINFO_PATH",
        mountinfo_path,
    )

    assert overhead_breakdown._cgroup_cpu_controls() == (
        [{"quota_us": 200_000, "period_us": 100_000, "burst_us": 10_000}],
        {"version": 2, "child_to_root": [100, 100]},
    )


def test_cgroup_v1_controls_resolve_namespaced_mount_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "self.cgroup"
    membership_path.write_text("2:cpu,cpuacct:/\n")
    mount_point = tmp_path / "cpu"
    mount_point.mkdir()
    (mount_point / "cpu.cfs_quota_us").write_text("50000")
    (mount_point / "cpu.cfs_period_us").write_text("100000")
    (mount_point / "cpu.shares").write_text("1024")
    mountinfo_path = tmp_path / "mountinfo"
    mountinfo_path.write_text(
        f"36 25 0:32 /docker/container {mount_point} rw,relatime - "
        "cgroup cgroup rw,cpu,cpuacct\n"
    )

    monkeypatch.setattr(overhead_breakdown.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MEMBERSHIP_PATH",
        membership_path,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MOUNTINFO_PATH",
        mountinfo_path,
    )

    assert overhead_breakdown._cgroup_cpu_controls() == (
        [{"quota_us": 50_000, "period_us": 100_000, "burst_us": 0}],
        {"version": 1, "child_to_root": [1024]},
    )


def test_cgroup_cpu_bandwidth_fails_closed_on_malformed_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "self.cgroup"
    membership_path.write_text("0::/\n")
    mount_point = tmp_path / "cgroup2"
    mount_point.mkdir()
    (mount_point / "cpu.max").write_text("invalid")
    mountinfo_path = tmp_path / "mountinfo"
    mountinfo_path.write_text(
        f"36 25 0:32 / {mount_point} rw,relatime - cgroup2 cgroup rw\n"
    )

    monkeypatch.setattr(overhead_breakdown.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MEMBERSHIP_PATH",
        membership_path,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_CGROUP_MOUNTINFO_PATH",
        mountinfo_path,
    )

    with pytest.raises(RuntimeError, match="cpu.max must contain quota and period"):
        overhead_breakdown._cgroup_cpu_controls()


def test_cgroup_cpu_weight_enforces_controller_range(tmp_path: Path) -> None:
    (tmp_path / "cpu.weight").write_text("0")
    assert overhead_breakdown._cgroup_cpu_weight(tmp_path, version=2) == 0

    (tmp_path / "cpu.weight").write_text("10001")
    with pytest.raises(RuntimeError, match="above 10000"):
        overhead_breakdown._cgroup_cpu_weight(tmp_path, version=2)

    (tmp_path / "cpu.shares").write_text("1")
    with pytest.raises(RuntimeError, match="below 2"):
        overhead_breakdown._cgroup_cpu_weight(tmp_path, version=1)


def test_receipt_rejects_execution_resource_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _execution_resources()
    changed = _execution_resources()
    changed["cpu_allocation"]["cgroup_cpu_weight_hierarchy"] = {
        "version": 2,
        "child_to_root": [50, 100],
    }
    monkeypatch.setattr(
        overhead_breakdown,
        "_cpu_allocation_metadata",
        lambda: changed["cpu_allocation"],
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_native_threadpool_metadata",
        lambda _backend: changed["native_threadpools"],
    )

    with pytest.raises(RuntimeError, match="execution resources changed"):
        overhead_breakdown._require_stable_execution_resources(
            expected,
            backend="numpy",
        )


def test_affinity_read_failure_is_not_treated_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_affinity(_pid: int) -> set[int]:
        raise OSError("unavailable")

    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getaffinity",
        fail_affinity,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="cannot determine process CPU affinity"):
        overhead_breakdown._process_cpu_affinity()


@pytest.mark.parametrize(
    ("seed", "hash_randomization"),
    (("0", 0), ("7", 1), ("4294967295", 1)),
)
def test_fixed_python_hash_seed_accepts_applied_decimal_seed(
    monkeypatch: pytest.MonkeyPatch,
    seed: str,
    hash_randomization: int,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    monkeypatch.setattr(
        overhead_breakdown.sys,
        "flags",
        SimpleNamespace(
            ignore_environment=0,
            hash_randomization=hash_randomization,
        ),
    )

    assert overhead_breakdown._fixed_python_hash_seed() == int(seed)


@pytest.mark.parametrize(
    ("seed", "ignore_environment", "hash_randomization", "message"),
    (
        (None, 0, 1, "requires a fixed"),
        ("random", 0, 1, "requires a fixed"),
        ("-1", 0, 1, "unsigned decimal"),
        ("4294967296", 0, 1, "outside"),
        ("7", 1, 1, "environment variables disabled"),
        ("0", 0, 1, "does not match"),
    ),
)
def test_fixed_python_hash_seed_rejects_uncontrolled_runtime(
    monkeypatch: pytest.MonkeyPatch,
    seed: str | None,
    ignore_environment: int,
    hash_randomization: int,
    message: str,
) -> None:
    if seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
    monkeypatch.setattr(
        overhead_breakdown.sys,
        "flags",
        SimpleNamespace(
            ignore_environment=ignore_environment,
            hash_randomization=hash_randomization,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        overhead_breakdown._fixed_python_hash_seed()


def test_python_runtime_metadata_records_build_abi_and_process_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(overhead_breakdown, "_fixed_python_hash_seed", lambda: 7)

    runtime = overhead_breakdown._python_runtime_metadata()

    assert runtime["implementation_name"] == overhead_breakdown.sys.implementation.name
    assert runtime["language_version"] == overhead_breakdown.platform.python_version()
    assert runtime["build"] == overhead_breakdown.sys.version
    assert runtime["cache_tag"] == overhead_breakdown.sys.implementation.cache_tag
    assert runtime["optimize"] == overhead_breakdown.sys.flags.optimize
    assert runtime["debug"] == overhead_breakdown.sys.flags.debug
    assert runtime["hash_seed"] == 7
    assert len(runtime["hash_witness"]) == 2


def test_receipt_rejects_python_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _python_runtime()
    changed = _python_runtime()
    changed["hash_seed"] = 7
    monkeypatch.setattr(
        overhead_breakdown,
        "_python_runtime_metadata",
        lambda: changed,
    )

    with pytest.raises(RuntimeError, match="runtime settings changed"):
        overhead_breakdown._require_stable_python_runtime(expected)


def test_overhead_json_emits_residual_field(tmp_path: Path) -> None:
    result = (
        ScenarioResult(
            scenario="fixed-medium",
            mode="fixed",
            scale="medium",
            cases=(
                CaseResult(
                    name="rearrange_flatten",
                    call_repr="rearrange(...)",
                    loops=100,
                    unpatched_call_ms=1.0,
                    instrumented_call_ms=1.1,
                    stage_ms_per_call={
                        "input_shape": 0.05,
                        "solve": 0.1,
                        "normalize_context": 0.0,
                        "backend_checks": 0.0,
                        "plan_select": 0.0,
                        "step_specialize": 0.0,
                        "runner_resolve": 0.1,
                        "fusion": 0.0,
                        "step_run": 0.05,
                        "primitive": 0.1,
                        "kernel": 0.6,
                    },
                    residual_ms_per_call=0.2,
                ),
            ),
        ),
    )

    payload = _to_json(
        result,
        backend="numpy",
        seed=20260215,
        resolved_stage_targets=_resolved_targets(),
        harness_source_sha256="0" * 64,
        subject_source={
            "kind": "git_checkout",
            "distribution_version": "0.2.0.dev1",
            "git_revision": "a" * 40,
            "git_dirty": False,
            "content_sha256": "1" * 64,
        },
        execution_resources=_execution_resources(),
        python_runtime=_python_runtime(),
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_overhead_report(path)
    assert loaded["schema_version"] == OVERHEAD_REPORT_SCHEMA_VERSION
    assert loaded["meta"]["execution_target"] == {
        "backend": "numpy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
    }
    assert loaded["meta"]["seed"] == 20260215
    assert len(loaded["meta"]["harness_source_sha256"]) == 64
    assert len(loaded["meta"]["subject_source"]["content_sha256"]) == 64
    assert loaded["meta"]["environment"]["python"] == _python_runtime()
    assert "torch" not in loaded["meta"]["environment"]
    case = loaded["scenarios"][0]["cases"][0]
    assert case.get("residual_ms_per_call") == 0.2


def test_guardrail_loader_accepts_residual_field(tmp_path: Path) -> None:
    payload = {
        "schema_version": OVERHEAD_REPORT_SCHEMA_VERSION,
        "meta": {
            "capture_id": "00000000-0000-4000-8000-000000000000",
            "harness_source_sha256": "0" * 64,
            "subject_source": {
                "kind": "git_checkout",
                "distribution_version": "0.2.0.dev1",
                "git_revision": "a" * 40,
                "git_dirty": False,
                "content_sha256": "1" * 64,
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
            "execution_resources": _execution_resources(),
            "environment": {
                "python": _python_runtime(),
                "numpy": "1.26",
                "array_api_compat": "1.12",
                "opt_einsum": "3.4",
            },
            "seed": 20260215,
            "stages": ["__call__", "solve", "runner_resolve", "fusion", "kernel"],
            "resolved_stage_targets": {
                "solve": ["einf.solve"],
                "runner_resolve": ["einf.runner_resolve"],
                "fusion": ["einf.fusion"],
                "kernel": ["einf.kernel"],
            },
        },
        "scenarios": [
            {
                "scenario": "fixed-medium",
                "mode": "fixed",
                "scale": "medium",
                "cases": [
                    {
                        "name": "rearrange_flatten",
                        "call_repr": "rearrange(...)",
                        "loops": 100,
                        "unpatched_call_ms": 1.0,
                        "instrumented_call_ms": 1.1,
                        "residual_ms_per_call": 0.2,
                        "stage_ms_per_call": {
                            "solve": 0.1,
                            "runner_resolve": 0.1,
                            "fusion": 0.0,
                            "kernel": 0.7,
                        },
                    }
                ],
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_overhead_report(path)
    case = loaded["scenarios"][0]["cases"][0]
    assert case.get("residual_ms_per_call") == 0.2


def test_profile_case_builds_fresh_invoke_for_each_timing_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_tokens: list[int] = []
    call_tokens: list[int] = []

    def build_invoke():
        token = len(build_tokens) + 1
        build_tokens.append(token)

        def invoke() -> np.ndarray:
            call_tokens.append(token)
            return np.zeros((1,), dtype=np.float32)

        return invoke

    def fake_measure_call_median_ms(
        *,
        invoke,
        loops: int,
        warmup: int,
        repeats: int,
    ) -> float:
        _ = loops, warmup, repeats
        invoke()
        return 1.0

    def fake_measure_stage_ms_per_call(
        *,
        invoke,
        loops: int,
        warmup: int,
    ) -> tuple[dict[str, float], float, float]:
        _ = loops, warmup
        invoke()
        return {}, 1.1, 0.1

    monkeypatch.setattr(
        "benchmarks.profile.overhead_breakdown._measure_call_median_ms",
        fake_measure_call_median_ms,
    )
    monkeypatch.setattr(
        "benchmarks.profile.overhead_breakdown._measure_stage_ms_per_call",
        fake_measure_stage_ms_per_call,
    )

    case = OverheadCase(
        name="fake",
        call_repr="fake()",
        build_invoke=build_invoke,
        loops=10,
    )

    result = _profile_case(case)

    assert result.unpatched_call_ms == 1.0
    assert result.instrumented_call_ms == 1.1
    assert result.residual_ms_per_call == 0.1
    assert build_tokens == [1, 2]
    assert call_tokens == [1, 2]
