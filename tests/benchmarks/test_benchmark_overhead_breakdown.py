import json
import subprocess
import sys
from importlib.metadata import Distribution
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import pytest

from benchmarks.guardrail.experiment import (
    CgroupCpuHierarchyFingerprint,
    CgroupV1CpuLevel,
    CgroupV2CpuLevel,
    CpuAllocationFingerprint,
    CpuBandwidthLimit,
    CpuUtilizationClamp,
    DependencyBuildFingerprint,
    ExecutionResourcesFingerprint,
    NativeRuntimeEnvironmentFingerprint,
    ProcessSchedulingFingerprint,
)
from benchmarks.guardrail.policy import load_overhead_report
from benchmarks.guardrail.receipt import (
    OVERHEAD_REPORT_SCHEMA_VERSION,
    OverheadDependencyBuildDict,
    OverheadEnvironmentDict,
    OverheadExecutionResourcesDict,
    OverheadPythonRuntimeDict,
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


def _dependency_build(
    version: str,
    *,
    digest_character: str,
) -> OverheadDependencyBuildDict:
    return OverheadDependencyBuildDict(
        version=version,
        record_sha256=digest_character * 64,
    )


def _execution_resources() -> OverheadExecutionResourcesDict:
    return {
        "cpu_allocation": {
            "process_cpu_affinity": None,
            "process_scheduling": {
                "scheduler_policy": None,
                "scheduler_priority": None,
                "nice_value": 0,
            },
            "cgroup_cpu_hierarchy": None,
        },
        "native_runtime_environment": {},
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


def _environment() -> OverheadEnvironmentDict:
    return OverheadEnvironmentDict(
        python=_python_runtime(),
        numpy=_dependency_build("1.26", digest_character="1"),
        array_api_compat=_dependency_build("1.12", digest_character="2"),
        opt_einsum=_dependency_build("3.4", digest_character="3"),
    )


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
    cpu_allocation = CpuAllocationFingerprint(
        process_cpu_affinity=None,
        process_scheduling=ProcessSchedulingFingerprint(
            scheduler_policy=None,
            scheduler_priority=None,
            nice_value=0,
        ),
        cgroup_cpu_hierarchy=None,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_cpu_allocation_fingerprint",
        lambda: cpu_allocation,
    )

    def execution_resources(
        _backend: object,
        *,
        cpu_allocation: CpuAllocationFingerprint,
    ) -> ExecutionResourcesFingerprint:
        assert cpu_allocation == CpuAllocationFingerprint(
            process_cpu_affinity=None,
            process_scheduling=ProcessSchedulingFingerprint(
                scheduler_policy=None,
                scheduler_priority=None,
                nice_value=0,
            ),
            cgroup_cpu_hierarchy=None,
        )
        return ExecutionResourcesFingerprint.from_mapping(
            _execution_resources(),
            backend="numpy",
            context="test resources",
        )

    monkeypatch.setattr(
        overhead_breakdown,
        "_execution_resources_fingerprint",
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

    cpu_allocation = CpuAllocationFingerprint(
        process_cpu_affinity=(1, 3),
        process_scheduling=ProcessSchedulingFingerprint(
            scheduler_policy="SCHED_OTHER",
            scheduler_priority=0,
            nice_value=0,
        ),
        cgroup_cpu_hierarchy=None,
    )
    resources = overhead_breakdown._execution_resources_fingerprint(
        "torch",
        cpu_allocation=cpu_allocation,
    )
    numpy_resources = overhead_breakdown._execution_resources_fingerprint(
        "numpy",
        cpu_allocation=cpu_allocation,
    )

    assert resources.cpu_allocation.process_cpu_affinity == (1, 3)
    assert [pool.user_api for pool in resources.native_threadpools] == [
        "blas",
        "openmp",
    ]
    assert resources.torch_threads == (4, 2)
    assert [pool.user_api for pool in numpy_resources.native_threadpools] == ["blas"]
    assert numpy_resources.torch_threads is None


def test_cgroup_v2_controls_preserve_child_to_root_levels(
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
    for directory, cpu_max, burst, weight, idle, uclamp in (
        (mount_point, "max 100000", "0", None, None, None),
        (
            mount_point / "parent",
            "200000 100000",
            "10000",
            "100",
            "0",
            ("0", "100"),
        ),
        (
            child,
            "200000 100000",
            "10000",
            "0",
            "1",
            ("12.500", "max"),
        ),
    ):
        directory.mkdir(exist_ok=True)
        (directory / "cpu.max").write_text(cpu_max)
        (directory / "cpu.max.burst").write_text(burst)
        if weight is not None:
            (directory / "cpu.weight").write_text(weight)
        if idle is not None:
            (directory / "cpu.idle").write_text(idle)
        if uclamp is not None:
            (directory / "cpu.uclamp.min").write_text(uclamp[0])
            (directory / "cpu.uclamp.max").write_text(uclamp[1])

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

    limit = CpuBandwidthLimit(
        quota_us=200_000,
        period_us=100_000,
        burst_us=10_000,
    )
    assert overhead_breakdown._cgroup_cpu_controls() == (
        CgroupCpuHierarchyFingerprint(
            version=2,
            child_to_root=(
                CgroupV2CpuLevel(
                    bandwidth_limit=limit,
                    weight=0,
                    idle=True,
                    utilization_clamp=CpuUtilizationClamp(
                        minimum_percent="12.5",
                        maximum_percent="max",
                    ),
                ),
                CgroupV2CpuLevel(
                    bandwidth_limit=limit,
                    weight=100,
                    idle=False,
                    utilization_clamp=CpuUtilizationClamp(
                        minimum_percent="0",
                        maximum_percent="100",
                    ),
                ),
                CgroupV2CpuLevel(
                    bandwidth_limit=None,
                    weight=None,
                    idle=None,
                    utilization_clamp=None,
                ),
            ),
        )
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
        CgroupCpuHierarchyFingerprint(
            version=1,
            child_to_root=(
                CgroupV1CpuLevel(
                    bandwidth_limit=CpuBandwidthLimit(
                        quota_us=50_000,
                        period_us=100_000,
                        burst_us=0,
                    ),
                    shares=1024,
                ),
            ),
        )
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


def test_cgroup_cpu_weight_and_idle_are_validated_together(tmp_path: Path) -> None:
    (tmp_path / "cpu.weight").write_text("0")
    assert (
        overhead_breakdown._bounded_optional_control_integer(
            tmp_path,
            control_name="cpu.weight",
            minimum=0,
            maximum=10_000,
        )
        == 0
    )
    CgroupV2CpuLevel(
        bandwidth_limit=None,
        weight=0,
        idle=True,
        utilization_clamp=None,
    )
    with pytest.raises(ValueError, match="non-idle"):
        CgroupV2CpuLevel(
            bandwidth_limit=None,
            weight=0,
            idle=False,
            utilization_clamp=None,
        )
    with pytest.raises(ValueError, match="idle cgroup"):
        CgroupV2CpuLevel(
            bandwidth_limit=None,
            weight=100,
            idle=True,
            utilization_clamp=None,
        )

    (tmp_path / "cpu.weight").write_text("10001")
    with pytest.raises(RuntimeError, match="above 10000"):
        overhead_breakdown._bounded_optional_control_integer(
            tmp_path,
            control_name="cpu.weight",
            minimum=0,
            maximum=10_000,
        )

    (tmp_path / "cpu.shares").write_text("1")
    with pytest.raises(RuntimeError, match="below 2"):
        overhead_breakdown._bounded_optional_control_integer(
            tmp_path,
            control_name="cpu.shares",
            minimum=2,
            maximum=262_144,
        )


def test_cgroup_v2_utilization_clamp_requires_both_controls(tmp_path: Path) -> None:
    (tmp_path / "cpu.uclamp.min").write_text("10")

    with pytest.raises(RuntimeError, match="must both be available"):
        overhead_breakdown._v2_cpu_utilization_clamp(tmp_path)


def test_cgroup_control_capture_rejects_unclassified_cpu_controls(
    tmp_path: Path,
) -> None:
    (tmp_path / "cpu.future_control").write_text("1")

    with pytest.raises(RuntimeError, match="cpu.future_control"):
        overhead_breakdown._reject_unknown_cgroup_cpu_controls(tmp_path, version=2)


def test_receipt_rejects_execution_resource_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ExecutionResourcesFingerprint.from_mapping(
        _execution_resources(),
        backend="numpy",
        context="expected resources",
    )
    changed = ExecutionResourcesFingerprint(
        backend="numpy",
        cpu_allocation=CpuAllocationFingerprint(
            process_cpu_affinity=None,
            process_scheduling=ProcessSchedulingFingerprint(
                scheduler_policy=None,
                scheduler_priority=None,
                nice_value=0,
            ),
            cgroup_cpu_hierarchy=CgroupCpuHierarchyFingerprint(
                version=2,
                child_to_root=(
                    CgroupV2CpuLevel(
                        bandwidth_limit=None,
                        weight=0,
                        idle=True,
                        utilization_clamp=None,
                    ),
                    CgroupV2CpuLevel(
                        bandwidth_limit=None,
                        weight=100,
                        idle=False,
                        utilization_clamp=None,
                    ),
                ),
            ),
        ),
        native_runtime_environment=NativeRuntimeEnvironmentFingerprint(variables=()),
        native_threadpools=expected.native_threadpools,
        torch_threads=None,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_execution_resources_fingerprint",
        lambda _backend, *, cpu_allocation: changed,
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


def test_process_scheduling_capture_preserves_default_policy_and_niceness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getscheduler",
        lambda _pid: 3,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "SCHED_OTHER",
        3,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getparam",
        lambda _pid: SimpleNamespace(sched_priority=0),
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "PRIO_PROCESS",
        0,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "getpriority",
        lambda _scope, _pid: 19,
        raising=False,
    )

    assert overhead_breakdown._process_scheduling_fingerprint() == (
        ProcessSchedulingFingerprint(
            scheduler_policy="SCHED_OTHER",
            scheduler_priority=0,
            nice_value=19,
        )
    )


def test_process_scheduling_capture_rejects_nondefault_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getscheduler",
        lambda _pid: 6,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getparam",
        lambda _pid: SimpleNamespace(sched_priority=0),
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "SCHED_OTHER",
        0,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="default SCHED_OTHER"):
        overhead_breakdown._process_scheduling_fingerprint()


def test_cpu_allocation_rejects_invalid_direct_scheduler_state() -> None:
    with pytest.raises(TypeError, match="ProcessSchedulingFingerprint"):
        CpuAllocationFingerprint(
            process_cpu_affinity=None,
            process_scheduling=cast(ProcessSchedulingFingerprint, None),
            cgroup_cpu_hierarchy=None,
        )


def test_process_scheduling_rejects_nonzero_default_policy_priority() -> None:
    with pytest.raises(ValueError, match="priority must be zero"):
        ProcessSchedulingFingerprint(
            scheduler_policy="SCHED_OTHER",
            scheduler_priority=1,
            nice_value=0,
        )


def test_process_scheduling_rejects_nondefault_direct_policy() -> None:
    with pytest.raises(ValueError, match="only the SCHED_OTHER"):
        ProcessSchedulingFingerprint(
            scheduler_policy=cast(Literal["SCHED_OTHER"], "SCHED_BATCH"),
            scheduler_priority=0,
            nice_value=0,
        )


def test_cgroup_hierarchy_rejects_invalid_direct_version() -> None:
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        CgroupCpuHierarchyFingerprint(
            version=cast(Literal[1, 2], 3),
            child_to_root=(
                CgroupV2CpuLevel(
                    bandwidth_limit=None,
                    weight=100,
                    idle=False,
                    utilization_clamp=None,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("failing_api", "message"),
    (
        ("sched_getscheduler", "cannot determine process scheduler state"),
        ("sched_getparam", "cannot determine process scheduler state"),
        ("getpriority", "cannot determine process niceness"),
    ),
)
def test_process_scheduling_read_failure_is_not_treated_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    failing_api: str,
    message: str,
) -> None:
    def fail(*_arguments: object) -> int:
        raise OSError("unavailable")

    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getscheduler",
        fail if failing_api == "sched_getscheduler" else lambda _pid: 0,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "SCHED_OTHER",
        0,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "sched_getparam",
        (
            fail
            if failing_api == "sched_getparam"
            else lambda _pid: SimpleNamespace(sched_priority=0)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "PRIO_PROCESS",
        0,
        raising=False,
    )
    monkeypatch.setattr(
        overhead_breakdown.os,
        "getpriority",
        fail if failing_api == "getpriority" else lambda _scope, _pid: 0,
        raising=False,
    )

    with pytest.raises(RuntimeError, match=message):
        overhead_breakdown._process_scheduling_fingerprint()


def test_numpy_resource_capture_does_not_request_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_import(_name: str) -> object:
        raise AssertionError("NumPy resource capture must not import Torch")

    monkeypatch.setattr(overhead_breakdown, "torch", None)
    monkeypatch.setattr(overhead_breakdown.importlib, "import_module", reject_import)
    monkeypatch.setattr(overhead_breakdown, "threadpool_info", list)
    cpu_allocation = CpuAllocationFingerprint.from_mapping(
        _execution_resources()["cpu_allocation"],
        context="test allocation",
    )

    resources = overhead_breakdown._execution_resources_fingerprint(
        "numpy",
        cpu_allocation=cpu_allocation,
    )

    assert resources.backend == "numpy"
    assert overhead_breakdown.torch is None


def test_numpy_profiler_import_does_not_load_torch() -> None:
    subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; "
                "import benchmarks.profile.overhead_breakdown; "
                "raise SystemExit(int('torch' in sys.modules))"
            ),
        ),
        check=True,
        capture_output=True,
        text=True,
    )


def test_dependency_build_identity_distinguishes_same_version_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SimpleNamespace(version="1.0", read_text=lambda _name: "first RECORD")
    second = SimpleNamespace(version="1.0", read_text=lambda _name: "second RECORD")
    installed = iter((first, second))
    monkeypatch.setattr(
        overhead_breakdown,
        "distribution",
        lambda _name: next(installed),
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_require_distribution_owns_import",
        lambda _name, _distribution: None,
    )

    first_fingerprint = overhead_breakdown._dependency_build_fingerprint("numpy")
    second_fingerprint = overhead_breakdown._dependency_build_fingerprint("numpy")

    assert first_fingerprint.version == second_fingerprint.version
    assert first_fingerprint.record_sha256 != second_fingerprint.record_sha256


def test_dependency_build_identity_requires_record_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = SimpleNamespace(version="1.0", read_text=lambda _name: None)
    monkeypatch.setattr(
        overhead_breakdown,
        "distribution",
        lambda _name: installed,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_require_distribution_owns_import",
        lambda _name, _distribution: None,
    )

    with pytest.raises(RuntimeError, match="no installed RECORD"):
        overhead_breakdown._dependency_build_fingerprint("numpy")


def test_dependency_build_identity_rejects_shadowed_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installed_package = tmp_path / "installed" / "numpy"
    imported_package = tmp_path / "shadowed" / "numpy"
    installed_package.mkdir(parents=True)
    imported_package.mkdir(parents=True)
    imported_init = imported_package / "__init__.py"
    imported_init.write_text("", encoding="utf-8")
    installed_distribution = cast(
        Distribution,
        SimpleNamespace(
            locate_file=lambda _name: installed_package,
        ),
    )
    monkeypatch.setattr(
        overhead_breakdown.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(imported_init)),
    )

    with pytest.raises(RuntimeError, match="does not belong"):
        overhead_breakdown._require_distribution_owns_import(
            "numpy",
            installed_distribution,
        )


@pytest.mark.parametrize(
    ("version", "record_sha256", "message"),
    (
        (cast(str, 1), "0" * 64, "version must be a non-empty string"),
        ("1.0", cast(str, None), "RECORD identity must be a string"),
        ("1.0", "not-a-digest", "lowercase SHA-256"),
    ),
)
def test_dependency_build_rejects_invalid_direct_state(
    version: str,
    record_sha256: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DependencyBuildFingerprint(
            version=version,
            record_sha256=record_sha256,
        )


def test_native_runtime_environment_captures_supported_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_DYNAMIC", "TRUE")
    monkeypatch.setenv("KMP_AFFINITY", "compact")
    monkeypatch.setenv("UNRELATED_SETTING", "ignored")

    fingerprint = overhead_breakdown._native_runtime_environment_fingerprint()

    assert ("OMP_DYNAMIC", "TRUE") in fingerprint.variables
    assert ("KMP_AFFINITY", "compact") in fingerprint.variables
    assert all(name != "UNRELATED_SETTING" for name, _value in fingerprint.variables)


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


def test_receipt_rejects_runtime_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _environment()
    changed = _environment()
    changed["python"]["hash_seed"] = 7
    monkeypatch.setattr(
        overhead_breakdown,
        "_environment_metadata",
        lambda _backend, *, python_runtime: changed,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_python_runtime_metadata",
        lambda: expected["python"],
    )

    with pytest.raises(RuntimeError, match="runtime environment changed"):
        overhead_breakdown._require_stable_environment(expected, backend="numpy")


def test_receipt_rejects_dependency_build_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _environment()
    changed = _environment()
    changed["numpy"]["record_sha256"] = "f" * 64
    monkeypatch.setattr(
        overhead_breakdown,
        "_environment_metadata",
        lambda _backend, *, python_runtime: changed,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_python_runtime_metadata",
        lambda: expected["python"],
    )

    with pytest.raises(RuntimeError, match="runtime environment changed"):
        overhead_breakdown._require_stable_environment(expected, backend="numpy")


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
        environment=_environment(),
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
            "environment": _environment(),
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
