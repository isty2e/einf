"""Serialized schema for overhead benchmark receipts."""

from typing import Literal, NotRequired, TypedDict

OVERHEAD_REPORT_SCHEMA_VERSION = 7
BackendName = Literal["numpy", "torch"]


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


class OverheadDependencyBuildDict(TypedDict):
    """Installed dependency version and distribution-manifest identity."""

    version: str
    record_sha256: str


class OverheadEnvironmentDict(TypedDict):
    """Runtime and dependency builds that can affect overhead measurements."""

    python: OverheadPythonRuntimeDict
    numpy: OverheadDependencyBuildDict
    array_api_compat: OverheadDependencyBuildDict
    opt_einsum: OverheadDependencyBuildDict
    torch: NotRequired[OverheadDependencyBuildDict]


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


class OverheadCpuUtilizationClampDict(TypedDict):
    """One cgroup v2 utilization clamp pair."""

    minimum_percent: str
    maximum_percent: str


class OverheadCgroupV1CpuLevelDict(TypedDict):
    """Fair-scheduler controls at one cgroup v1 level."""

    bandwidth_limit: OverheadCpuBandwidthLimitDict | None
    shares: int | None


class OverheadCgroupV2CpuLevelDict(TypedDict):
    """Scheduler controls at one cgroup v2 level."""

    bandwidth_limit: OverheadCpuBandwidthLimitDict | None
    weight: int | None
    idle: bool | None
    utilization_clamp: OverheadCpuUtilizationClampDict | None


class OverheadCgroupV1CpuHierarchyDict(TypedDict):
    """Cgroup v1 CPU controls from the process cgroup to the visible root."""

    version: Literal[1]
    child_to_root: list[OverheadCgroupV1CpuLevelDict]


class OverheadCgroupV2CpuHierarchyDict(TypedDict):
    """Cgroup v2 CPU controls from the process cgroup to the visible root."""

    version: Literal[2]
    child_to_root: list[OverheadCgroupV2CpuLevelDict]


OverheadCgroupCpuHierarchyDict = (
    OverheadCgroupV1CpuHierarchyDict | OverheadCgroupV2CpuHierarchyDict
)


class OverheadProcessSchedulingDict(TypedDict):
    """Scheduling policy and priority of the benchmark task."""

    scheduler_policy: Literal["SCHED_OTHER"] | None
    scheduler_priority: int | None
    nice_value: int | None


class OverheadCpuAllocationDict(TypedDict):
    """CPU scheduling context of the benchmark task and its cgroup."""

    process_cpu_affinity: list[int] | None
    process_scheduling: OverheadProcessSchedulingDict
    cgroup_cpu_hierarchy: OverheadCgroupCpuHierarchyDict | None


class OverheadExecutionResourcesDict(TypedDict):
    """Process-level CPU allocation and effective backend thread settings."""

    cpu_allocation: OverheadCpuAllocationDict
    native_runtime_environment: dict[str, str]
    native_threadpools: list[OverheadNativeThreadPoolDict]
    torch_threads: NotRequired[OverheadTorchThreadsDict]


__all__ = [
    "OVERHEAD_REPORT_SCHEMA_VERSION",
    "BackendName",
    "OverheadCgroupCpuHierarchyDict",
    "OverheadCgroupV1CpuHierarchyDict",
    "OverheadCgroupV1CpuLevelDict",
    "OverheadCgroupV2CpuHierarchyDict",
    "OverheadCgroupV2CpuLevelDict",
    "OverheadCpuAllocationDict",
    "OverheadCpuBandwidthLimitDict",
    "OverheadCpuUtilizationClampDict",
    "OverheadDependencyBuildDict",
    "OverheadEnvironmentDict",
    "OverheadExecutionResourcesDict",
    "OverheadExecutionTargetDict",
    "OverheadHostDict",
    "OverheadNativeThreadPoolDict",
    "OverheadProcessSchedulingDict",
    "OverheadPythonRuntimeDict",
    "OverheadSourceDict",
    "OverheadTorchThreadsDict",
]
