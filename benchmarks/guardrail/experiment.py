"""Canonical experiment identity for overhead benchmark receipts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, cast

from .receipt import (
    BackendName,
    OverheadCgroupCpuHierarchyDict,
    OverheadCgroupV1CpuHierarchyDict,
    OverheadCgroupV1CpuLevelDict,
    OverheadCgroupV2CpuHierarchyDict,
    OverheadCgroupV2CpuLevelDict,
    OverheadCpuAllocationDict,
    OverheadCpuBandwidthLimitDict,
    OverheadCpuUtilizationClampDict,
    OverheadDependencyBuildDict,
    OverheadEnvironmentDict,
    OverheadExecutionResourcesDict,
    OverheadExecutionTargetDict,
    OverheadHostDict,
    OverheadNativeThreadPoolDict,
    OverheadProcessSchedulingDict,
    OverheadTorchThreadsDict,
)

_NATIVE_RUNTIME_ENVIRONMENT_PREFIXES = (
    "BLIS_",
    "GOMP_",
    "GOTO_",
    "KMP_",
    "MKL_",
    "OMP_",
    "OPENBLAS_",
    "VECLIB_",
)


def _required_integer(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> int:
    value = mapping.get(name)
    if type(value) is not int:
        raise TypeError(f"{context}: {name} must be an integer")
    return value


def _required_string(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}: {name} must be a non-empty string")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    *,
    name: str,
    context: str,
) -> Mapping[str, object]:
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
        raise ValueError(f"{context}: {name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CpuBandwidthLimit:
    """Canonical finite cgroup CPU bandwidth constraint."""

    quota_us: int
    period_us: int
    burst_us: int

    def __post_init__(self) -> None:
        if self.quota_us < 1 or self.period_us < 1 or self.burst_us < 0:
            raise ValueError(
                "CPU bandwidth quota and period must be positive and burst non-negative"
            )
        if self.burst_us > self.quota_us:
            raise ValueError("CPU bandwidth burst cannot exceed quota")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "CpuBandwidthLimit | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{context}: bandwidth_limit must be object or null")
        return cls(
            quota_us=_required_integer(value, name="quota_us", context=context),
            period_us=_required_integer(value, name="period_us", context=context),
            burst_us=_required_integer(value, name="burst_us", context=context),
        )

    def to_receipt(self) -> OverheadCpuBandwidthLimitDict:
        """Return the canonical receipt representation."""
        return OverheadCpuBandwidthLimitDict(
            quota_us=self.quota_us,
            period_us=self.period_us,
            burst_us=self.burst_us,
        )


def _canonical_percentage(value: str, *, allow_max: bool, context: str) -> str:
    if allow_max and value == "max":
        return value
    try:
        percentage = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{context} must be a percentage or max") from error
    if not percentage.is_finite() or not Decimal(0) <= percentage <= Decimal(100):
        raise ValueError(f"{context} must be between 0 and 100")
    rendered = format(percentage.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


@dataclass(frozen=True, slots=True)
class CpuUtilizationClamp:
    """Canonical cgroup v2 utilization clamp pair."""

    minimum_percent: str
    maximum_percent: str

    def __post_init__(self) -> None:
        minimum = _canonical_percentage(
            self.minimum_percent,
            allow_max=False,
            context="cpu.uclamp.min",
        )
        maximum = _canonical_percentage(
            self.maximum_percent,
            allow_max=True,
            context="cpu.uclamp.max",
        )
        if maximum != "max" and Decimal(minimum) > Decimal(maximum):
            raise ValueError("cpu.uclamp.min cannot exceed cpu.uclamp.max")
        object.__setattr__(self, "minimum_percent", minimum)
        object.__setattr__(self, "maximum_percent", maximum)

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "CpuUtilizationClamp | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{context}: utilization_clamp must be object or null")
        return cls(
            minimum_percent=_required_string(
                value,
                name="minimum_percent",
                context=context,
            ),
            maximum_percent=_required_string(
                value,
                name="maximum_percent",
                context=context,
            ),
        )

    def to_receipt(self) -> OverheadCpuUtilizationClampDict:
        """Return the canonical receipt representation."""
        return OverheadCpuUtilizationClampDict(
            minimum_percent=self.minimum_percent,
            maximum_percent=self.maximum_percent,
        )


@dataclass(frozen=True, slots=True)
class CgroupV1CpuLevel:
    """Canonical fair-scheduler state at one cgroup v1 level."""

    bandwidth_limit: CpuBandwidthLimit | None
    shares: int | None

    def __post_init__(self) -> None:
        if self.shares is not None and not 2 <= self.shares <= 262_144:
            raise ValueError("cgroup v1 cpu.shares must be between 2 and 262144")

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> "CgroupV1CpuLevel":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        if "bandwidth_limit" not in value or "shares" not in value:
            raise TypeError(f"{context} requires bandwidth_limit and shares fields")
        shares = value.get("shares")
        if shares is not None and type(shares) is not int:
            raise TypeError(f"{context}: shares must be integer or null")
        return cls(
            bandwidth_limit=CpuBandwidthLimit.from_mapping(
                value.get("bandwidth_limit"),
                context=context,
            ),
            shares=shares,
        )

    def to_receipt(self) -> OverheadCgroupV1CpuLevelDict:
        """Return the canonical receipt representation."""
        return OverheadCgroupV1CpuLevelDict(
            bandwidth_limit=(
                None
                if self.bandwidth_limit is None
                else self.bandwidth_limit.to_receipt()
            ),
            shares=self.shares,
        )


@dataclass(frozen=True, slots=True)
class CgroupV2CpuLevel:
    """Canonical scheduler state at one cgroup v2 level."""

    bandwidth_limit: CpuBandwidthLimit | None
    weight: int | None
    idle: bool | None
    utilization_clamp: CpuUtilizationClamp | None

    def __post_init__(self) -> None:
        if self.idle is True:
            if self.weight != 0:
                raise ValueError("idle cgroup v2 levels must report cpu.weight 0")
            return
        if self.weight is not None and not 1 <= self.weight <= 10_000:
            raise ValueError(
                "non-idle cgroup v2 cpu.weight must be between 1 and 10000"
            )

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> "CgroupV2CpuLevel":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        required_fields = (
            "bandwidth_limit",
            "weight",
            "idle",
            "utilization_clamp",
        )
        missing = tuple(field for field in required_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        weight = value.get("weight")
        if weight is not None and type(weight) is not int:
            raise TypeError(f"{context}: weight must be integer or null")
        idle = value.get("idle")
        if idle is not None and type(idle) is not bool:
            raise TypeError(f"{context}: idle must be boolean or null")
        return cls(
            bandwidth_limit=CpuBandwidthLimit.from_mapping(
                value.get("bandwidth_limit"),
                context=context,
            ),
            weight=weight,
            idle=idle,
            utilization_clamp=CpuUtilizationClamp.from_mapping(
                value.get("utilization_clamp"),
                context=context,
            ),
        )

    def to_receipt(self) -> OverheadCgroupV2CpuLevelDict:
        """Return the canonical receipt representation."""
        return OverheadCgroupV2CpuLevelDict(
            bandwidth_limit=(
                None
                if self.bandwidth_limit is None
                else self.bandwidth_limit.to_receipt()
            ),
            weight=self.weight,
            idle=self.idle,
            utilization_clamp=(
                None
                if self.utilization_clamp is None
                else self.utilization_clamp.to_receipt()
            ),
        )


CgroupCpuLevel = CgroupV1CpuLevel | CgroupV2CpuLevel


@dataclass(frozen=True, slots=True)
class CgroupCpuHierarchyFingerprint:
    """Canonical cgroup CPU hierarchy and its controller version."""

    version: Literal[1, 2]
    child_to_root: tuple[CgroupCpuLevel, ...]

    def __post_init__(self) -> None:
        if self.version not in (1, 2):
            raise ValueError("cgroup CPU hierarchy version must be 1 or 2")
        if not self.child_to_root:
            raise ValueError("cgroup CPU hierarchy must contain at least one level")
        expected_type = CgroupV1CpuLevel if self.version == 1 else CgroupV2CpuLevel
        if not all(type(level) is expected_type for level in self.child_to_root):
            raise TypeError("cgroup CPU hierarchy levels must match its version")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "CgroupCpuHierarchyFingerprint | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be object or null")
        version = _required_integer(value, name="version", context=context)
        if version not in (1, 2):
            raise ValueError(f"{context}: version must be 1 or 2")
        levels_raw = value.get("child_to_root")
        if not isinstance(levels_raw, list):
            raise TypeError(f"{context}: child_to_root must be a list")
        level_type = CgroupV1CpuLevel if version == 1 else CgroupV2CpuLevel
        levels = tuple(
            level_type.from_mapping(
                level,
                context=f"{context}: child_to_root[{index}]",
            )
            for index, level in enumerate(levels_raw)
        )
        return cls(version=version, child_to_root=levels)

    def to_receipt(self) -> OverheadCgroupCpuHierarchyDict:
        """Return the version-specific canonical receipt representation."""
        if self.version == 1:
            levels_v1 = cast(tuple[CgroupV1CpuLevel, ...], self.child_to_root)
            levels = [level.to_receipt() for level in levels_v1]
            return OverheadCgroupV1CpuHierarchyDict(
                version=1,
                child_to_root=levels,
            )
        canonical_levels_v2 = cast(tuple[CgroupV2CpuLevel, ...], self.child_to_root)
        levels_v2 = [level.to_receipt() for level in canonical_levels_v2]
        return OverheadCgroupV2CpuHierarchyDict(
            version=2,
            child_to_root=levels_v2,
        )


@dataclass(frozen=True, slots=True)
class ProcessSchedulingFingerprint:
    """Canonical scheduling policy and priority of the benchmark task."""

    scheduler_policy: Literal["SCHED_OTHER"] | None
    scheduler_priority: int | None
    nice_value: int | None

    def __post_init__(self) -> None:
        if (self.scheduler_policy is None) != (self.scheduler_priority is None):
            raise ValueError("scheduler policy and priority must be available together")
        if self.scheduler_policy not in (None, "SCHED_OTHER"):
            raise ValueError("only the SCHED_OTHER scheduler policy is supported")
        for name in ("scheduler_priority", "nice_value"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"process scheduling {name} must be integer or null")
        if self.scheduler_policy is not None and self.scheduler_priority != 0:
            raise ValueError("SCHED_OTHER scheduler priority must be zero")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "ProcessSchedulingFingerprint":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        required_fields = (
            "scheduler_policy",
            "scheduler_priority",
            "nice_value",
        )
        missing = tuple(field for field in required_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        policy = value["scheduler_policy"]
        if policy is not None and not isinstance(policy, str):
            raise TypeError(f"{context}: scheduler_policy must be string or null")
        priority = value["scheduler_priority"]
        nice_value = value["nice_value"]
        for field, raw in (
            ("scheduler_priority", priority),
            ("nice_value", nice_value),
        ):
            if raw is not None and type(raw) is not int:
                raise TypeError(f"{context}: {field} must be integer or null")
        return cls(
            scheduler_policy=cast(Literal["SCHED_OTHER"] | None, policy),
            scheduler_priority=cast(int | None, priority),
            nice_value=cast(int | None, nice_value),
        )

    def to_receipt(self) -> OverheadProcessSchedulingDict:
        """Return the canonical receipt representation."""
        return OverheadProcessSchedulingDict(
            scheduler_policy=self.scheduler_policy,
            scheduler_priority=self.scheduler_priority,
            nice_value=self.nice_value,
        )


@dataclass(frozen=True, slots=True)
class CpuAllocationFingerprint:
    """Canonical CPU scheduling allocation for one capture."""

    process_cpu_affinity: tuple[int, ...] | None
    process_scheduling: ProcessSchedulingFingerprint
    cgroup_cpu_hierarchy: CgroupCpuHierarchyFingerprint | None

    def __post_init__(self) -> None:
        if type(self.process_scheduling) is not ProcessSchedulingFingerprint:
            raise TypeError("process scheduling must be a ProcessSchedulingFingerprint")
        if (
            self.cgroup_cpu_hierarchy is not None
            and type(self.cgroup_cpu_hierarchy) is not CgroupCpuHierarchyFingerprint
        ):
            raise TypeError(
                "cgroup CPU hierarchy must be a CgroupCpuHierarchyFingerprint or null"
            )
        affinity = self.process_cpu_affinity
        if affinity is not None and (
            not affinity
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or affinity != tuple(sorted(set(affinity)))
        ):
            raise ValueError(
                "process CPU affinity must be non-empty, sorted, and unique"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        context: str,
    ) -> "CpuAllocationFingerprint":
        required_fields = (
            "process_cpu_affinity",
            "process_scheduling",
            "cgroup_cpu_hierarchy",
        )
        missing = tuple(field for field in required_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        affinity_raw = value.get("process_cpu_affinity")
        if affinity_raw is None:
            affinity = None
        elif not isinstance(affinity_raw, list) or not all(
            type(cpu) is int for cpu in affinity_raw
        ):
            raise TypeError(
                f"{context}: process_cpu_affinity must be list[int] or null"
            )
        else:
            affinity = tuple(affinity_raw)
        return cls(
            process_cpu_affinity=affinity,
            process_scheduling=ProcessSchedulingFingerprint.from_mapping(
                value.get("process_scheduling"),
                context=f"{context}: process_scheduling",
            ),
            cgroup_cpu_hierarchy=CgroupCpuHierarchyFingerprint.from_mapping(
                value.get("cgroup_cpu_hierarchy"),
                context=f"{context}: cgroup_cpu_hierarchy",
            ),
        )

    def to_receipt(self) -> OverheadCpuAllocationDict:
        """Return the canonical receipt representation."""
        return OverheadCpuAllocationDict(
            process_cpu_affinity=(
                None
                if self.process_cpu_affinity is None
                else list(self.process_cpu_affinity)
            ),
            process_scheduling=self.process_scheduling.to_receipt(),
            cgroup_cpu_hierarchy=(
                None
                if self.cgroup_cpu_hierarchy is None
                else self.cgroup_cpu_hierarchy.to_receipt()
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeThreadPoolFingerprint:
    """Canonical native thread-pool identity and effective size."""

    user_api: str
    internal_api: str
    prefix: str
    num_threads: int
    version: str | None
    threading_layer: str | None
    architecture: str | None

    def __post_init__(self) -> None:
        for name in ("user_api", "internal_api", "prefix"):
            if not getattr(self, name):
                raise ValueError(f"native thread pool {name} must not be empty")
        if self.num_threads < 1:
            raise ValueError("native thread pool num_threads must be positive")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "NativeThreadPoolFingerprint":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        nullable_fields = ("version", "threading_layer", "architecture")
        missing = tuple(field for field in nullable_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        nullable: dict[str, str | None] = {}
        for name in nullable_fields:
            field = value.get(name)
            if field is not None and (not isinstance(field, str) or not field):
                raise TypeError(f"{context}: {name} must be string or null")
            nullable[name] = field
        return cls(
            user_api=_required_string(value, name="user_api", context=context),
            internal_api=_required_string(
                value,
                name="internal_api",
                context=context,
            ),
            prefix=_required_string(value, name="prefix", context=context),
            num_threads=_required_integer(
                value,
                name="num_threads",
                context=context,
            ),
            version=nullable["version"],
            threading_layer=nullable["threading_layer"],
            architecture=nullable["architecture"],
        )

    def to_receipt(self) -> OverheadNativeThreadPoolDict:
        """Return the canonical receipt representation."""
        return OverheadNativeThreadPoolDict(
            user_api=self.user_api,
            internal_api=self.internal_api,
            prefix=self.prefix,
            num_threads=self.num_threads,
            version=self.version,
            threading_layer=self.threading_layer,
            architecture=self.architecture,
        )

    @property
    def canonical_order(self) -> tuple[str, str, str, str, str, str, int]:
        """Return the stable ordering key used in experiment fingerprints."""
        return (
            self.user_api,
            self.internal_api,
            self.prefix,
            self.version or "",
            self.threading_layer or "",
            self.architecture or "",
            self.num_threads,
        )


@dataclass(frozen=True, slots=True)
class NativeRuntimeEnvironmentFingerprint:
    """Canonical native-runtime environment visible to this process."""

    variables: tuple[tuple[str, str], ...]

    @staticmethod
    def recognizes(name: str) -> bool:
        """Return whether an environment variable belongs to a native runtime."""
        return name.startswith(_NATIVE_RUNTIME_ENVIRONMENT_PREFIXES)

    def __post_init__(self) -> None:
        if type(self.variables) is not tuple:
            raise TypeError("native runtime environment variables must be a tuple")
        for variable in self.variables:
            if type(variable) is not tuple or len(variable) != 2:
                raise TypeError(
                    "native runtime environment entries must be name-value pairs"
                )
            name, value = variable
            if type(name) is not str or type(value) is not str:
                raise TypeError(
                    "native runtime environment names and values must be strings"
                )
        names = tuple(name for name, _value in self.variables)
        if names != tuple(sorted(set(names))):
            raise ValueError(
                "native runtime environment names must be sorted and unique"
            )
        for name, _value in self.variables:
            if not self.recognizes(name):
                raise ValueError(f"unsupported native runtime variable {name!r}")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "NativeRuntimeEnvironmentFingerprint":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        variables: list[tuple[str, str]] = []
        for name, raw_value in value.items():
            if not isinstance(name, str) or not isinstance(raw_value, str):
                raise TypeError(f"{context} must map strings to strings")
            variables.append((name, raw_value))
        return cls(variables=tuple(sorted(variables)))

    def to_receipt(self) -> dict[str, str]:
        """Return the canonical receipt representation."""
        return dict(self.variables)


@dataclass(frozen=True, slots=True)
class ExecutionResourcesFingerprint:
    """Canonical process resources that affect overhead latency."""

    backend: BackendName
    cpu_allocation: CpuAllocationFingerprint
    native_runtime_environment: NativeRuntimeEnvironmentFingerprint
    native_threadpools: tuple[NativeThreadPoolFingerprint, ...]
    torch_threads: tuple[int, int] | None

    def __post_init__(self) -> None:
        if self.backend not in ("numpy", "torch"):
            raise ValueError(f"unsupported execution backend {self.backend!r}")
        if type(self.cpu_allocation) is not CpuAllocationFingerprint:
            raise TypeError("CPU allocation must be a CpuAllocationFingerprint")
        if (
            type(self.native_runtime_environment)
            is not NativeRuntimeEnvironmentFingerprint
        ):
            raise TypeError(
                "native runtime environment must be a "
                "NativeRuntimeEnvironmentFingerprint"
            )
        if type(self.native_threadpools) is not tuple or any(
            type(pool) is not NativeThreadPoolFingerprint
            for pool in self.native_threadpools
        ):
            raise TypeError(
                "native thread pools must be a tuple of NativeThreadPoolFingerprint"
            )
        if self.backend == "numpy" and self.torch_threads is not None:
            raise ValueError("NumPy execution resources cannot include torch threads")
        if self.backend == "torch" and self.torch_threads is None:
            raise ValueError("Torch execution resources require torch threads")
        if self.torch_threads is not None and (
            type(self.torch_threads) is not tuple
            or len(self.torch_threads) != 2
            or any(
                type(thread_count) is not int or thread_count < 1
                for thread_count in self.torch_threads
            )
        ):
            raise ValueError("torch thread settings must contain two positive integers")
        object.__setattr__(
            self,
            "native_threadpools",
            tuple(
                sorted(
                    self.native_threadpools,
                    key=lambda pool: pool.canonical_order,
                )
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        backend: BackendName,
        context: str,
    ) -> "ExecutionResourcesFingerprint":
        required_fields = (
            "cpu_allocation",
            "native_runtime_environment",
            "native_threadpools",
        )
        missing = tuple(field for field in required_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        allocation_raw = _required_mapping(
            value,
            name="cpu_allocation",
            context=context,
        )
        native_runtime_environment_raw = _required_mapping(
            value,
            name="native_runtime_environment",
            context=context,
        )
        threadpools_raw = value.get("native_threadpools")
        if not isinstance(threadpools_raw, list):
            raise TypeError(f"{context}: native_threadpools must be a list")
        threadpools = tuple(
            NativeThreadPoolFingerprint.from_mapping(
                threadpool,
                context=f"{context}: native_threadpools[{index}]",
            )
            for index, threadpool in enumerate(threadpools_raw)
        )
        torch_raw = value.get("torch_threads")
        if backend == "torch":
            if not isinstance(torch_raw, dict):
                raise TypeError(f"{context}: torch_threads must be an object")
            torch_threads = (
                _required_integer(torch_raw, name="intra_op", context=context),
                _required_integer(torch_raw, name="inter_op", context=context),
            )
        else:
            if "torch_threads" in value:
                raise ValueError(
                    f"{context}: numpy resources cannot include torch_threads"
                )
            torch_threads = None
        return cls(
            backend=backend,
            cpu_allocation=CpuAllocationFingerprint.from_mapping(
                allocation_raw,
                context=f"{context}: cpu_allocation",
            ),
            native_runtime_environment=NativeRuntimeEnvironmentFingerprint.from_mapping(
                native_runtime_environment_raw,
                context=f"{context}: native_runtime_environment",
            ),
            native_threadpools=threadpools,
            torch_threads=torch_threads,
        )

    def to_receipt(self) -> OverheadExecutionResourcesDict:
        """Return the canonical receipt representation."""
        receipt = OverheadExecutionResourcesDict(
            cpu_allocation=self.cpu_allocation.to_receipt(),
            native_runtime_environment=self.native_runtime_environment.to_receipt(),
            native_threadpools=[pool.to_receipt() for pool in self.native_threadpools],
        )
        if self.torch_threads is not None:
            receipt["torch_threads"] = OverheadTorchThreadsDict(
                intra_op=self.torch_threads[0],
                inter_op=self.torch_threads[1],
            )
        return receipt


@dataclass(frozen=True, slots=True)
class DependencyBuildFingerprint:
    """Canonical installed-distribution build identity."""

    version: str
    record_sha256: str

    def __post_init__(self) -> None:
        if type(self.version) is not str or not self.version:
            raise TypeError("dependency version must be a non-empty string")
        if type(self.record_sha256) is not str:
            raise TypeError("dependency RECORD identity must be a string")
        if len(self.record_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.record_sha256
        ):
            raise ValueError("dependency RECORD identity must be a lowercase SHA-256")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> "DependencyBuildFingerprint":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        required_fields = ("version", "record_sha256")
        missing = tuple(field for field in required_fields if field not in value)
        if missing:
            raise TypeError(f"{context} missing required fields: {', '.join(missing)}")
        return cls(
            version=_required_string(value, name="version", context=context),
            record_sha256=_required_sha256(
                value,
                name="record_sha256",
                context=context,
            ),
        )

    def to_receipt(self) -> OverheadDependencyBuildDict:
        """Return the canonical receipt representation."""
        return OverheadDependencyBuildDict(
            version=self.version,
            record_sha256=self.record_sha256,
        )


@dataclass(frozen=True, slots=True)
class DependencyBuildsFingerprint:
    """Canonical dependency builds selected for one benchmark backend."""

    numpy: DependencyBuildFingerprint
    array_api_compat: DependencyBuildFingerprint
    opt_einsum: DependencyBuildFingerprint
    torch: DependencyBuildFingerprint | None

    def __post_init__(self) -> None:
        for name in ("numpy", "array_api_compat", "opt_einsum"):
            if type(getattr(self, name)) is not DependencyBuildFingerprint:
                raise TypeError(
                    f"{name} dependency must be a DependencyBuildFingerprint"
                )
        if (
            self.torch is not None
            and type(self.torch) is not DependencyBuildFingerprint
        ):
            raise TypeError(
                "torch dependency must be a DependencyBuildFingerprint or null"
            )

    @classmethod
    def from_environment(
        cls,
        environment: OverheadEnvironmentDict,
    ) -> "DependencyBuildsFingerprint":
        """Normalize receipt dependency records into one named build set."""
        torch_dependency = environment.get("torch")
        return cls(
            numpy=DependencyBuildFingerprint.from_mapping(
                environment["numpy"],
                context="normalized NumPy dependency",
            ),
            array_api_compat=DependencyBuildFingerprint.from_mapping(
                environment["array_api_compat"],
                context="normalized array-api-compat dependency",
            ),
            opt_einsum=DependencyBuildFingerprint.from_mapping(
                environment["opt_einsum"],
                context="normalized opt_einsum dependency",
            ),
            torch=(
                None
                if torch_dependency is None
                else DependencyBuildFingerprint.from_mapping(
                    torch_dependency,
                    context="normalized Torch dependency",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class CaseConfigurationFingerprint:
    """Canonical execution form for one comparable overhead case."""

    scenario: str
    mode: str
    scale: str
    case_name: str
    call_repr: str
    loops: int

    def __post_init__(self) -> None:
        if not all(
            (
                self.scenario,
                self.mode,
                self.scale,
                self.case_name,
                self.call_repr,
            )
        ):
            raise ValueError("case configuration strings must not be empty")
        if self.loops < 1:
            raise ValueError("case configuration loops must be positive")

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Return the identity shared by equivalent case configurations."""
        return self.scenario, self.mode, self.scale, self.case_name

    @property
    def execution_form(self) -> tuple[str, int]:
        """Return the settings that must match for this case identity."""
        return self.call_repr, self.loops


@dataclass(frozen=True, slots=True)
class ExperimentFingerprint:
    """Canonical identity used to decide whether overhead reports are comparable."""

    schema_version: int
    harness_source_sha256: str
    execution_target: tuple[str, str, str]
    host: tuple[str, str, str, str, int]
    execution_resources: ExecutionResourcesFingerprint
    python_runtime: tuple[
        str,
        str,
        str,
        str,
        str | None,
        str,
        int,
        int,
        bool | None,
        int,
        tuple[int, int],
    ]
    dependency_builds: DependencyBuildsFingerprint
    seed: int
    case_configurations: tuple[CaseConfigurationFingerprint, ...]
    instrumentation: (
        tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]] | None
    )

    def __post_init__(self) -> None:
        if self.execution_target[0] != self.execution_resources.backend:
            raise ValueError(
                "execution target and resource fingerprint backends must match"
            )
        if type(self.dependency_builds) is not DependencyBuildsFingerprint:
            raise TypeError("dependency builds must be a DependencyBuildsFingerprint")
        has_torch_build = self.dependency_builds.torch is not None
        if (self.execution_resources.backend == "torch") != has_torch_build:
            raise ValueError(
                "Torch dependency presence must match the execution backend"
            )

    @classmethod
    def from_normalized(
        cls,
        *,
        schema_version: int,
        harness_source_sha256: str,
        execution_target: OverheadExecutionTargetDict,
        host: OverheadHostDict,
        execution_resources: OverheadExecutionResourcesDict,
        environment: OverheadEnvironmentDict,
        seed: int,
        case_configurations: Mapping[
            tuple[str, str, str, str],
            tuple[str, int],
        ],
        stages: list[str],
        resolved_stage_targets: dict[str, list[str]],
        include_instrumentation: bool,
    ) -> "ExperimentFingerprint":
        python = environment["python"]
        backend = execution_target["backend"]
        if backend not in ("numpy", "torch"):
            raise ValueError(f"unsupported execution backend {backend!r}")
        instrumentation = (
            (
                tuple(stages),
                tuple(
                    (stage, tuple(targets))
                    for stage, targets in sorted(resolved_stage_targets.items())
                ),
            )
            if include_instrumentation
            else None
        )
        return cls(
            schema_version=schema_version,
            harness_source_sha256=harness_source_sha256,
            execution_target=(
                backend,
                execution_target["requested_device"],
                execution_target["resolved_device"],
            ),
            host=(
                host["system"],
                host["release"],
                host["machine"],
                host["cpu_model"],
                host["logical_cpu_count"],
            ),
            execution_resources=ExecutionResourcesFingerprint.from_mapping(
                execution_resources,
                backend=backend,
                context="normalized execution resources",
            ),
            python_runtime=(
                python["implementation_name"],
                python["implementation_version"],
                python["language_version"],
                python["build"],
                python["cache_tag"],
                python["abi_flags"],
                python["optimize"],
                python["debug"],
                python["py_debug"],
                python["hash_seed"],
                python["hash_witness"],
            ),
            dependency_builds=DependencyBuildsFingerprint.from_environment(environment),
            seed=seed,
            case_configurations=tuple(
                CaseConfigurationFingerprint(
                    scenario=key[0],
                    mode=key[1],
                    scale=key[2],
                    case_name=key[3],
                    call_repr=configuration[0],
                    loops=configuration[1],
                )
                for key, configuration in sorted(case_configurations.items())
            ),
            instrumentation=instrumentation,
        )

    @staticmethod
    def mismatched_axes(
        fingerprints: Sequence["ExperimentFingerprint"],
    ) -> tuple[str, ...]:
        """Return incompatible axes across receipts in stable diagnostic order."""
        if not fingerprints:
            return ()
        expected = fingerprints[0]
        mismatches: set[str] = set()
        known_case_configurations: dict[
            tuple[str, str, str, str],
            tuple[str, int],
        ] = {}
        for fingerprint in fingerprints:
            if expected.schema_version != fingerprint.schema_version:
                mismatches.add("schema")
            if expected.harness_source_sha256 != fingerprint.harness_source_sha256:
                mismatches.add("harness_source")
            if expected.execution_target != fingerprint.execution_target:
                mismatches.add("execution_target")
            if expected.host != fingerprint.host:
                mismatches.add("host")
            if expected.execution_resources != fingerprint.execution_resources:
                mismatches.add("execution_resources")
            if (
                expected.python_runtime != fingerprint.python_runtime
                or expected.dependency_builds != fingerprint.dependency_builds
            ):
                mismatches.add("environment")
            if expected.seed != fingerprint.seed:
                mismatches.add("configuration")
            if expected.instrumentation != fingerprint.instrumentation:
                mismatches.add("instrumentation")
            for case in fingerprint.case_configurations:
                existing = known_case_configurations.setdefault(
                    case.key,
                    case.execution_form,
                )
                if existing != case.execution_form:
                    mismatches.add("configuration")
        return tuple(
            axis
            for axis in (
                "schema",
                "harness_source",
                "execution_target",
                "host",
                "execution_resources",
                "environment",
                "configuration",
                "instrumentation",
            )
            if axis in mismatches
        )


__all__ = [
    "CaseConfigurationFingerprint",
    "CgroupCpuHierarchyFingerprint",
    "CgroupV1CpuLevel",
    "CgroupV2CpuLevel",
    "CpuAllocationFingerprint",
    "CpuBandwidthLimit",
    "CpuUtilizationClamp",
    "DependencyBuildFingerprint",
    "ExecutionResourcesFingerprint",
    "ExperimentFingerprint",
    "NativeRuntimeEnvironmentFingerprint",
    "NativeThreadPoolFingerprint",
    "ProcessSchedulingFingerprint",
]
