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
from uuid import UUID

MetricName = Literal["unpatched_call_ms", "instrumented_call_ms"]
OVERHEAD_REPORT_SCHEMA_VERSION = 6


def _required_string(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> str:
    """Return one required non-empty string field."""
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}: {name} must be a non-empty string")
    return value


def _required_integer(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> int:
    """Return one required integer field, excluding booleans."""
    value = mapping.get(name)
    if type(value) is not int:
        raise TypeError(f"{context}: {name} must be an integer")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> Mapping[str, object]:
    """Return one required object field."""
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"{context}: {name} must be an object")
    return value


def _required_sha256(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> str:
    value = _required_string(mapping, name=name, context=context)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{context}: {name} must be lowercase SHA-256")
    return value


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


class OverheadSourceDict(TypedDict):
    """Source identity of the measured einf subject."""

    kind: str
    distribution_version: str
    git_revision: str | None
    git_dirty: bool | None
    content_sha256: str


class OverheadExecutionTargetDict(TypedDict):
    """Resolved backend and device used for one overhead capture."""

    backend: str
    requested_device: str
    resolved_device: str


class OverheadPythonRuntimeDict(TypedDict):
    """Python implementation and process settings affecting measured overhead."""

    implementation_name: str
    implementation_version: str
    language_version: str
    build: str
    cache_tag: str | None
    abi_flags: str
    optimize: int
    debug: int
    py_debug: bool | None
    hash_seed: int
    hash_witness: tuple[int, int]


class OverheadEnvironmentDict(TypedDict):
    """Runtime and dependency versions that can affect overhead measurements."""

    python: OverheadPythonRuntimeDict
    numpy: str
    array_api_compat: str
    opt_einsum: str
    torch: NotRequired[str]


class OverheadHostDict(TypedDict):
    """Host hardware identity required for CPU latency comparison."""

    system: str
    release: str
    machine: str
    cpu_model: str
    logical_cpu_count: int


class OverheadNativeThreadPoolDict(TypedDict):
    """One effective native thread pool visible to the benchmark process."""

    user_api: str
    internal_api: str
    prefix: str
    num_threads: int
    version: str | None
    threading_layer: str | None
    architecture: str | None


class OverheadTorchThreadsDict(TypedDict):
    """Effective PyTorch thread counts for one capture."""

    intra_op: int
    inter_op: int


class OverheadCpuBandwidthLimitDict(TypedDict):
    """One finite cgroup CPU bandwidth constraint."""

    quota_us: int
    period_us: int
    burst_us: int


class OverheadCgroupCpuLevelDict(TypedDict):
    """CPU controls at one visible cgroup level."""

    bandwidth_limit: OverheadCpuBandwidthLimitDict | None
    weight: int | None


class OverheadCgroupCpuHierarchyDict(TypedDict):
    """CPU controls from the process cgroup toward the visible root."""

    version: int
    child_to_root: list[OverheadCgroupCpuLevelDict]


class OverheadCpuAllocationDict(TypedDict):
    """CPU scheduling capacity assigned to the benchmark process."""

    process_cpu_affinity: list[int] | None
    cgroup_cpu_hierarchy: OverheadCgroupCpuHierarchyDict | None


class OverheadExecutionResourcesDict(TypedDict):
    """Process-level CPU allocation and effective backend thread settings."""

    cpu_allocation: OverheadCpuAllocationDict
    native_threadpools: list[OverheadNativeThreadPoolDict]
    torch_threads: NotRequired[OverheadTorchThreadsDict]


class OverheadMetaDict(TypedDict):
    """Top-level metadata section in overhead raw JSON."""

    capture_id: str
    harness_source_sha256: str
    subject_source: OverheadSourceDict
    execution_target: OverheadExecutionTargetDict
    host: OverheadHostDict
    execution_resources: OverheadExecutionResourcesDict
    environment: OverheadEnvironmentDict
    seed: int
    stages: list[str]
    resolved_stage_targets: dict[str, list[str]]


class OverheadReportDict(TypedDict):
    """Top-level overhead raw JSON payload."""

    schema_version: int
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

    context = f"invalid overhead report at {path}"
    schema_version = _required_integer(
        loaded,
        name="schema_version",
        context=context,
    )
    if schema_version != OVERHEAD_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: unsupported schema_version {schema_version}; "
            f"expected {OVERHEAD_REPORT_SCHEMA_VERSION}"
        )

    meta_raw = loaded.get("meta")
    scenarios_raw = loaded.get("scenarios")
    if not isinstance(meta_raw, dict):
        raise TypeError(f"invalid overhead report at {path}: missing object meta")
    if not isinstance(scenarios_raw, list):
        raise TypeError(f"invalid overhead report at {path}: missing list scenarios")

    capture_id = _required_string(meta_raw, name="capture_id", context=context)
    try:
        parsed_capture_id = UUID(capture_id)
    except ValueError as error:
        raise ValueError(f"{context}: capture_id must be a UUID") from error
    if str(parsed_capture_id) != capture_id:
        raise ValueError(f"{context}: capture_id must use canonical UUID form")

    harness_source_sha256 = _required_sha256(
        meta_raw,
        name="harness_source_sha256",
        context=context,
    )

    source_raw = _required_mapping(
        meta_raw,
        name="subject_source",
        context=context,
    )
    source_kind = _required_string(
        source_raw,
        name="kind",
        context=f"{context}: meta.subject_source",
    )
    if source_kind not in ("git_checkout", "installed_distribution"):
        raise ValueError(f"{context}: unsupported subject source kind {source_kind!r}")
    distribution_version = _required_string(
        source_raw,
        name="distribution_version",
        context=f"{context}: meta.subject_source",
    )
    git_revision_raw = source_raw.get("git_revision")
    if git_revision_raw is not None and (
        not isinstance(git_revision_raw, str) or not git_revision_raw
    ):
        raise TypeError(f"{context}: subject git_revision must be string or null")
    git_dirty_raw = source_raw.get("git_dirty")
    if git_dirty_raw is not None and type(git_dirty_raw) is not bool:
        raise TypeError(f"{context}: subject git_dirty must be boolean or null")
    if source_kind == "git_checkout" and (
        git_revision_raw is None or git_dirty_raw is None
    ):
        raise ValueError(
            f"{context}: git checkout source requires revision and dirty state"
        )
    if source_kind == "installed_distribution" and (
        git_revision_raw is not None or git_dirty_raw is not None
    ):
        raise ValueError(
            f"{context}: installed distribution source cannot include git state"
        )
    content_sha256 = _required_sha256(
        source_raw,
        name="content_sha256",
        context=f"{context}: meta.subject_source",
    )

    target_raw = _required_mapping(
        meta_raw,
        name="execution_target",
        context=context,
    )
    backend = _required_string(
        target_raw,
        name="backend",
        context=f"{context}: meta.execution_target",
    )
    if backend not in ("numpy", "torch"):
        raise ValueError(f"{context}: unsupported execution backend {backend!r}")
    execution_target = OverheadExecutionTargetDict(
        backend=backend,
        requested_device=_required_string(
            target_raw,
            name="requested_device",
            context=f"{context}: meta.execution_target",
        ),
        resolved_device=_required_string(
            target_raw,
            name="resolved_device",
            context=f"{context}: meta.execution_target",
        ),
    )

    host_raw = _required_mapping(meta_raw, name="host", context=context)
    logical_cpu_count = _required_integer(
        host_raw,
        name="logical_cpu_count",
        context=f"{context}: meta.host",
    )
    if logical_cpu_count < 1:
        raise ValueError(f"{context}: host logical_cpu_count must be positive")
    host = OverheadHostDict(
        system=_required_string(
            host_raw,
            name="system",
            context=f"{context}: meta.host",
        ),
        release=_required_string(
            host_raw,
            name="release",
            context=f"{context}: meta.host",
        ),
        machine=_required_string(
            host_raw,
            name="machine",
            context=f"{context}: meta.host",
        ),
        cpu_model=_required_string(
            host_raw,
            name="cpu_model",
            context=f"{context}: meta.host",
        ),
        logical_cpu_count=logical_cpu_count,
    )

    resources_raw = _required_mapping(
        meta_raw,
        name="execution_resources",
        context=context,
    )
    allocation_raw = _required_mapping(
        resources_raw,
        name="cpu_allocation",
        context=f"{context}: meta.execution_resources",
    )
    affinity_raw = allocation_raw.get("process_cpu_affinity")
    if affinity_raw is None:
        process_cpu_affinity = None
    elif not isinstance(affinity_raw, list) or not all(
        type(cpu_index) is int and cpu_index >= 0 for cpu_index in affinity_raw
    ):
        raise TypeError(f"{context}: process_cpu_affinity must be list[int] or null")
    elif not affinity_raw or affinity_raw != sorted(set(affinity_raw)):
        raise ValueError(
            f"{context}: process_cpu_affinity must be non-empty, sorted, and unique"
        )
    else:
        process_cpu_affinity = list(affinity_raw)

    if "cgroup_cpu_hierarchy" not in allocation_raw:
        raise TypeError(f"{context}: cgroup CPU hierarchy is required")
    hierarchy_raw = allocation_raw.get("cgroup_cpu_hierarchy")
    if hierarchy_raw is None:
        cgroup_cpu_hierarchy = None
    elif not isinstance(hierarchy_raw, dict):
        raise TypeError(f"{context}: cgroup CPU hierarchy must be object or null")
    else:
        hierarchy_context = f"{context}: cgroup CPU hierarchy"
        cgroup_version = _required_integer(
            hierarchy_raw,
            name="version",
            context=hierarchy_context,
        )
        if cgroup_version not in (1, 2):
            raise ValueError(f"{hierarchy_context}: version must be 1 or 2")
        levels_raw = hierarchy_raw.get("child_to_root")
        if not isinstance(levels_raw, list):
            raise TypeError(f"{hierarchy_context}: child_to_root must be a list")
        if not levels_raw:
            raise ValueError(f"{hierarchy_context}: child_to_root must not be empty")

        minimum_weight, maximum_weight = (
            (2, 262_144) if cgroup_version == 1 else (1, 10_000)
        )
        levels: list[OverheadCgroupCpuLevelDict] = []
        for index, level_raw in enumerate(levels_raw):
            level_context = f"{hierarchy_context}: child_to_root[{index}]"
            if not isinstance(level_raw, dict):
                raise TypeError(f"{level_context} must be an object")
            if "bandwidth_limit" not in level_raw:
                raise TypeError(f"{level_context}: bandwidth_limit is required")
            if "weight" not in level_raw:
                raise TypeError(f"{level_context}: weight is required")

            limit_raw = level_raw.get("bandwidth_limit")
            if limit_raw is None:
                bandwidth_limit = None
            elif not isinstance(limit_raw, dict):
                raise TypeError(
                    f"{level_context}: bandwidth_limit must be object or null"
                )
            else:
                quota_us = _required_integer(
                    limit_raw,
                    name="quota_us",
                    context=level_context,
                )
                period_us = _required_integer(
                    limit_raw,
                    name="period_us",
                    context=level_context,
                )
                burst_us = _required_integer(
                    limit_raw,
                    name="burst_us",
                    context=level_context,
                )
                if quota_us < 1 or period_us < 1 or burst_us < 0:
                    raise ValueError(
                        f"{level_context}: quota and period must be positive "
                        "and burst non-negative"
                    )
                bandwidth_limit = OverheadCpuBandwidthLimitDict(
                    quota_us=quota_us,
                    period_us=period_us,
                    burst_us=burst_us,
                )

            weight_raw = level_raw.get("weight")
            if weight_raw is None:
                weight = None
            elif type(weight_raw) is not int or not (
                minimum_weight <= weight_raw <= maximum_weight
            ):
                raise TypeError(f"{level_context}: weight is invalid")
            else:
                weight = weight_raw
            levels.append(
                OverheadCgroupCpuLevelDict(
                    bandwidth_limit=bandwidth_limit,
                    weight=weight,
                )
            )

        cgroup_cpu_hierarchy = OverheadCgroupCpuHierarchyDict(
            version=cgroup_version,
            child_to_root=levels,
        )

    native_threadpools_raw = resources_raw.get("native_threadpools")
    if not isinstance(native_threadpools_raw, list):
        raise TypeError(f"{context}: native_threadpools must be a list")
    native_threadpools: list[OverheadNativeThreadPoolDict] = []
    for index, threadpool_raw in enumerate(native_threadpools_raw):
        threadpool_context = f"{context}: native_threadpools[{index}]"
        if not isinstance(threadpool_raw, dict):
            raise TypeError(f"{threadpool_context} must be an object")
        identity = (
            _required_string(
                threadpool_raw,
                name="user_api",
                context=threadpool_context,
            ),
            _required_string(
                threadpool_raw,
                name="internal_api",
                context=threadpool_context,
            ),
            _required_string(
                threadpool_raw,
                name="prefix",
                context=threadpool_context,
            ),
        )
        num_threads = _required_integer(
            threadpool_raw,
            name="num_threads",
            context=threadpool_context,
        )
        if num_threads < 1:
            raise ValueError(f"{threadpool_context}: num_threads must be positive")

        optional_values: dict[str, str | None] = {}
        for field_name in ("version", "threading_layer", "architecture"):
            field_value = threadpool_raw.get(field_name)
            if field_value is not None and (
                not isinstance(field_value, str) or not field_value
            ):
                raise TypeError(
                    f"{threadpool_context}: {field_name} must be string or null"
                )
            optional_values[field_name] = field_value
        native_threadpools.append(
            OverheadNativeThreadPoolDict(
                user_api=identity[0],
                internal_api=identity[1],
                prefix=identity[2],
                num_threads=num_threads,
                version=optional_values["version"],
                threading_layer=optional_values["threading_layer"],
                architecture=optional_values["architecture"],
            )
        )
    native_threadpools.sort(
        key=lambda threadpool: (
            threadpool["user_api"],
            threadpool["internal_api"],
            threadpool["prefix"],
            threadpool["version"] or "",
            threadpool["threading_layer"] or "",
            threadpool["architecture"] or "",
            threadpool["num_threads"],
        )
    )

    execution_resources = OverheadExecutionResourcesDict(
        cpu_allocation=OverheadCpuAllocationDict(
            process_cpu_affinity=process_cpu_affinity,
            cgroup_cpu_hierarchy=cgroup_cpu_hierarchy,
        ),
        native_threadpools=native_threadpools,
    )
    if backend == "torch":
        torch_threads_raw = _required_mapping(
            resources_raw,
            name="torch_threads",
            context=context,
        )
        intra_op = _required_integer(
            torch_threads_raw,
            name="intra_op",
            context=f"{context}: meta.execution_resources.torch_threads",
        )
        inter_op = _required_integer(
            torch_threads_raw,
            name="inter_op",
            context=f"{context}: meta.execution_resources.torch_threads",
        )
        if intra_op < 1 or inter_op < 1:
            raise ValueError(f"{context}: torch thread counts must be positive")
        execution_resources["torch_threads"] = OverheadTorchThreadsDict(
            intra_op=intra_op,
            inter_op=inter_op,
        )
    elif "torch_threads" in resources_raw:
        raise ValueError(
            f"{context}: NumPy execution resources cannot include torch_threads"
        )

    environment_raw = _required_mapping(
        meta_raw,
        name="environment",
        context=context,
    )
    python_raw = _required_mapping(
        environment_raw,
        name="python",
        context=f"{context}: meta.environment",
    )
    python_context = f"{context}: meta.environment.python"
    cache_tag = python_raw.get("cache_tag")
    if cache_tag is not None and (not isinstance(cache_tag, str) or not cache_tag):
        raise TypeError(f"{python_context}: cache_tag must be string or null")
    abi_flags = python_raw.get("abi_flags")
    if not isinstance(abi_flags, str):
        raise TypeError(f"{python_context}: abi_flags must be a string")
    optimize = _required_integer(
        python_raw,
        name="optimize",
        context=python_context,
    )
    debug = _required_integer(
        python_raw,
        name="debug",
        context=python_context,
    )
    if optimize not in (0, 1, 2):
        raise ValueError(f"{python_context}: optimize must be 0, 1, or 2")
    if debug not in (0, 1):
        raise ValueError(f"{python_context}: debug must be 0 or 1")
    py_debug = python_raw.get("py_debug")
    if py_debug is not None and type(py_debug) is not bool:
        raise TypeError(f"{python_context}: py_debug must be boolean or null")
    hash_seed = _required_integer(
        python_raw,
        name="hash_seed",
        context=python_context,
    )
    if not 0 <= hash_seed <= 4_294_967_295:
        raise ValueError(f"{python_context}: hash_seed is outside the valid range")
    hash_witness_raw = python_raw.get("hash_witness")
    if (
        not isinstance(hash_witness_raw, list)
        or len(hash_witness_raw) != 2
        or not all(type(value) is int for value in hash_witness_raw)
    ):
        raise TypeError(f"{python_context}: hash_witness must contain two integers")
    python_runtime = OverheadPythonRuntimeDict(
        implementation_name=_required_string(
            python_raw,
            name="implementation_name",
            context=python_context,
        ),
        implementation_version=_required_string(
            python_raw,
            name="implementation_version",
            context=python_context,
        ),
        language_version=_required_string(
            python_raw,
            name="language_version",
            context=python_context,
        ),
        build=_required_string(
            python_raw,
            name="build",
            context=python_context,
        ),
        cache_tag=cache_tag,
        abi_flags=abi_flags,
        optimize=optimize,
        debug=debug,
        py_debug=py_debug,
        hash_seed=hash_seed,
        hash_witness=(hash_witness_raw[0], hash_witness_raw[1]),
    )
    environment = OverheadEnvironmentDict(
        python=python_runtime,
        numpy=_required_string(
            environment_raw,
            name="numpy",
            context=f"{context}: meta.environment",
        ),
        array_api_compat=_required_string(
            environment_raw,
            name="array_api_compat",
            context=f"{context}: meta.environment",
        ),
        opt_einsum=_required_string(
            environment_raw,
            name="opt_einsum",
            context=f"{context}: meta.environment",
        ),
    )
    if backend == "numpy":
        if "torch" in environment_raw:
            raise ValueError(f"{context}: NumPy environment cannot include torch")
    else:
        environment["torch"] = _required_string(
            environment_raw,
            name="torch",
            context=f"{context}: meta.environment",
        )

    seed = _required_integer(meta_raw, name="seed", context=context)
    stages_raw = meta_raw.get("stages")
    if not isinstance(stages_raw, list) or not all(
        isinstance(stage_name, str) and stage_name for stage_name in stages_raw
    ):
        raise TypeError(
            f"invalid overhead report at {path}: meta.stages must be list[str]"
        )
    if len(set(stages_raw)) != len(stages_raw):
        raise ValueError(f"{context}: meta.stages must be unique")

    resolved_targets_raw = _required_mapping(
        meta_raw,
        name="resolved_stage_targets",
        context=context,
    )
    expected_instrumented_stages = set(stages_raw) - {"__call__"}
    if set(resolved_targets_raw) != expected_instrumented_stages:
        raise ValueError(
            f"{context}: resolved_stage_targets must cover every instrumented stage"
        )
    resolved_stage_targets: dict[str, list[str]] = {}
    for stage, targets_raw in resolved_targets_raw.items():
        if (
            not isinstance(stage, str)
            or not isinstance(targets_raw, list)
            or not all(isinstance(target, str) and target for target in targets_raw)
        ):
            raise TypeError(
                f"{context}: resolved_stage_targets must map stages to list[str]"
            )
        if not targets_raw or len(set(targets_raw)) != len(targets_raw):
            raise ValueError(
                f"{context}: resolved targets for stage {stage!r} must be non-empty and unique"
            )
        resolved_stage_targets[stage] = list(targets_raw)

    meta: OverheadMetaDict = {
        "capture_id": capture_id,
        "harness_source_sha256": harness_source_sha256,
        "subject_source": OverheadSourceDict(
            kind=source_kind,
            distribution_version=distribution_version,
            git_revision=git_revision_raw,
            git_dirty=git_dirty_raw,
            content_sha256=content_sha256,
        ),
        "execution_target": execution_target,
        "host": host,
        "execution_resources": execution_resources,
        "environment": environment,
        "seed": seed,
        "stages": list(stages_raw),
        "resolved_stage_targets": resolved_stage_targets,
    }

    scenarios: list[OverheadScenarioDict] = []
    seen_case_keys: set[tuple[str, str, str, str]] = set()
    for scenario_raw in scenarios_raw:
        if not isinstance(scenario_raw, dict):
            raise TypeError(
                f"invalid overhead report at {path}: each scenario must be object"
            )
        scenario_name = _required_string(
            scenario_raw,
            name="scenario",
            context=context,
        )
        mode = _required_string(scenario_raw, name="mode", context=context)
        scale = _required_string(scenario_raw, name="scale", context=context)
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
            case_name = _required_string(case_raw, name="name", context=context)
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

            loops = _required_integer(
                case_raw,
                name="loops",
                context=case_context,
            )
            if loops < 1:
                raise ValueError(f"{case_context}: loops must be positive")
            case = OverheadCaseDict(
                name=case_name,
                call_repr=_required_string(
                    case_raw,
                    name="call_repr",
                    context=case_context,
                ),
                loops=loops,
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

    return OverheadReportDict(
        schema_version=schema_version,
        meta=meta,
        scenarios=scenarios,
    )


def _experiment_axes(
    report: OverheadReportDict,
    *,
    metric: MetricName,
) -> dict[str, object]:
    """Return controlled report-wide conditions in diagnostic order."""
    meta = report["meta"]
    target = meta["execution_target"]
    environment = meta["environment"]
    host = meta["host"]
    resources = meta["execution_resources"]
    allocation = resources["cpu_allocation"]
    torch_threads = resources.get("torch_threads")
    axes: dict[str, object] = {
        "schema": report["schema_version"],
        "harness_source": meta["harness_source_sha256"],
        "execution_target": (
            target["backend"],
            target["requested_device"],
            target["resolved_device"],
        ),
        "host": (
            host["system"],
            host["release"],
            host["machine"],
            host["cpu_model"],
            host["logical_cpu_count"],
        ),
        "execution_resources": (
            (
                None
                if allocation["process_cpu_affinity"] is None
                else tuple(allocation["process_cpu_affinity"])
            ),
            (
                None
                if allocation["cgroup_cpu_hierarchy"] is None
                else (
                    allocation["cgroup_cpu_hierarchy"]["version"],
                    tuple(
                        (
                            (
                                None
                                if level["bandwidth_limit"] is None
                                else (
                                    level["bandwidth_limit"]["quota_us"],
                                    level["bandwidth_limit"]["period_us"],
                                    level["bandwidth_limit"]["burst_us"],
                                )
                            ),
                            level["weight"],
                        )
                        for level in allocation["cgroup_cpu_hierarchy"]["child_to_root"]
                    ),
                )
            ),
            tuple(
                (
                    threadpool["user_api"],
                    threadpool["internal_api"],
                    threadpool["prefix"],
                    threadpool["num_threads"],
                    threadpool["version"],
                    threadpool["threading_layer"],
                    threadpool["architecture"],
                )
                for threadpool in resources["native_threadpools"]
            ),
            (
                None
                if torch_threads is None
                else (torch_threads["intra_op"], torch_threads["inter_op"])
            ),
        ),
        "environment": (
            tuple(sorted(environment["python"].items())),
            environment["numpy"],
            environment["array_api_compat"],
            environment["opt_einsum"],
            environment.get("torch"),
        ),
        "configuration": meta["seed"],
    }
    if metric == "instrumented_call_ms":
        axes["instrumentation"] = (
            tuple(meta["stages"]),
            tuple(
                (stage, tuple(targets))
                for stage, targets in sorted(meta["resolved_stage_targets"].items())
            ),
        )
    return axes


def _case_configurations(
    report: OverheadReportDict,
) -> dict[tuple[str, str, str, str], tuple[str, int]]:
    """Return configuration for each comparable case identity."""
    return {
        (
            scenario["scenario"],
            scenario["mode"],
            scenario["scale"],
            case["name"],
        ): (case["call_repr"], case["loops"])
        for scenario in report["scenarios"]
        for case in scenario["cases"]
    }


def _subject_source_identity(
    report: OverheadReportDict,
) -> tuple[str, str]:
    source = report["meta"]["subject_source"]
    return (
        source["kind"],
        source["content_sha256"],
    )


def _require_compatible_experiments(
    reports: tuple[OverheadReportDict, ...],
    *,
    metric: MetricName,
) -> None:
    if not reports:
        return
    expected_axes = _experiment_axes(reports[0], metric=metric)
    known_case_configurations: dict[tuple[str, str, str, str], tuple[str, int]] = {}
    mismatches: set[str] = set()
    capture_ids = [report["meta"]["capture_id"] for report in reports]
    if len(set(capture_ids)) != len(capture_ids):
        raise ValueError("overhead reports must come from distinct captures")
    for report in reports:
        axes = _experiment_axes(report, metric=metric)
        mismatches.update(
            axis for axis, expected in expected_axes.items() if axes[axis] != expected
        )
        case_configurations = _case_configurations(report)
        for key, configuration in case_configurations.items():
            existing = known_case_configurations.setdefault(key, configuration)
            if existing != configuration:
                mismatches.add("configuration")
    if mismatches:
        ordered = tuple(axis for axis in expected_axes if axis in mismatches)
        raise ValueError(
            "overhead reports are not comparable: mismatched experiment axes: "
            + ", ".join(ordered)
        )


def _require_stable_trial_sources(
    report_pairs: tuple[tuple[OverheadReportDict, OverheadReportDict], ...],
) -> None:
    if not report_pairs:
        return
    expected_baseline = _subject_source_identity(report_pairs[0][0])
    expected_candidate = _subject_source_identity(report_pairs[0][1])
    for baseline, candidate in report_pairs[1:]:
        if _subject_source_identity(baseline) != expected_baseline:
            raise ValueError("baseline trial reports use different subject sources")
        if _subject_source_identity(candidate) != expected_candidate:
            raise ValueError("candidate trial reports use different subject sources")


def _require_stable_trial_case_configurations(
    report_pairs: tuple[tuple[OverheadReportDict, OverheadReportDict], ...],
) -> None:
    if not report_pairs:
        return
    expected_baseline = _case_configurations(report_pairs[0][0])
    expected_candidate = _case_configurations(report_pairs[0][1])
    if not expected_baseline:
        raise ValueError("baseline trial reports must contain at least one case")
    for baseline, candidate in report_pairs[1:]:
        if _case_configurations(baseline) != expected_baseline:
            raise ValueError("baseline trial reports use different case configurations")
        if _case_configurations(candidate) != expected_candidate:
            raise ValueError(
                "candidate trial reports use different case configurations"
            )


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
    if not _case_configurations(baseline):
        raise ValueError("baseline overhead report must contain at least one case")
    _require_compatible_experiments((baseline, candidate), metric=metric)
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

    _require_compatible_experiments(
        tuple(report for report_pair in report_pairs for report in report_pair),
        metric=metric,
    )
    _require_stable_trial_sources(report_pairs)
    _require_stable_trial_case_configurations(report_pairs)

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
    "OVERHEAD_REPORT_SCHEMA_VERSION",
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
