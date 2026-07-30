import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from .case import BenchmarkCase
from .types import LibraryName
from .workload import DynamicWorkloadMetadata


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Summary stats for one sample distribution."""

    count: int
    p25_ms: float
    median_ms: float
    p75_ms: float
    iqr_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class UnavailableRun:
    """One unavailable library result."""

    reason: str


@dataclass(frozen=True, slots=True)
class FixedRun:
    """Fixed-shape timing summaries for one available library."""

    cold: TimingSummary
    warm: TimingSummary
    warm_rounds: tuple[TimingSummary, ...]


@dataclass(frozen=True, slots=True)
class DynamicRun:
    """Dynamic-shape timing summaries for one available library."""

    summary: TimingSummary
    round_summaries: tuple[TimingSummary, ...]


FixedRunResult = FixedRun | UnavailableRun
DynamicRunResult = DynamicRun | UnavailableRun


@dataclass(frozen=True, slots=True)
class DynamicObservation:
    """One timed dynamic call with its pairing and execution identity."""

    round_index: int
    measured_batch_index: int
    repeat_index: int
    library: LibraryName
    order_position: int
    latency_ms: float

    def __post_init__(self) -> None:
        """Reject invalid observation coordinates and timing values."""
        for field_name, value in (
            ("round_index", self.round_index),
            ("measured_batch_index", self.measured_batch_index),
            ("repeat_index", self.repeat_index),
            ("order_position", self.order_position),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError(
                f"latency_ms must be finite and positive, got {self.latency_ms}"
            )


_ComparisonMemberT = TypeVar("_ComparisonMemberT", bound=str)


@dataclass(frozen=True, slots=True)
class PairedComparison(Generic[_ComparisonMemberT]):
    """Paired latency-ratio estimate with batch-level uncertainty."""

    baseline: _ComparisonMemberT
    competitor: _ComparisonMemberT
    call_pair_count: int
    paired_batch_count: int
    latency_ratio: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_resamples: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        """Reject malformed paired-effect evidence."""
        if self.baseline == self.competitor:
            raise ValueError("paired comparison requires distinct members")
        if self.paired_batch_count < 1:
            raise ValueError("paired_batch_count must be positive")
        if self.call_pair_count < self.paired_batch_count:
            raise ValueError(
                "call_pair_count must cover every paired batch at least once"
            )
        if not math.isfinite(self.latency_ratio) or self.latency_ratio <= 0.0:
            raise ValueError("latency_ratio must be finite and positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if (
            not math.isfinite(self.confidence_interval_low)
            or not math.isfinite(self.confidence_interval_high)
            or self.confidence_interval_low <= 0.0
            or self.confidence_interval_high < self.confidence_interval_low
        ):
            raise ValueError(
                "confidence interval must be finite, positive, and ordered"
            )
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True, slots=True)
class FixedCaseResult:
    """One fixed-shape benchmark case run across libraries."""

    case: BenchmarkCase
    runs: dict[LibraryName, FixedRunResult]
    round_orders: list[tuple[LibraryName, ...]]


@dataclass(frozen=True, slots=True)
class DynamicCaseResult:
    """One dynamic-shape benchmark case with raw and derived evidence."""

    case: BenchmarkCase
    workload: DynamicWorkloadMetadata
    runs: dict[LibraryName, DynamicRunResult]
    round_orders: list[tuple[LibraryName, ...]]
    observations: tuple[DynamicObservation, ...]
    comparisons: tuple[PairedComparison[LibraryName], ...]


_CaseResultT = TypeVar("_CaseResultT", FixedCaseResult, DynamicCaseResult)


@dataclass(frozen=True, slots=True)
class TestResult(Generic[_CaseResultT]):
    """Top-level benchmark result payload."""

    title: str
    configuration: list[str]
    methodology: list[str]
    case_results: list[_CaseResultT]
    notes: list[str]
