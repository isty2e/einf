from dataclasses import dataclass

from .case import BenchmarkCase


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
class LibraryRun:
    """One library timing output for one case."""

    available: bool
    reason: str
    cold: TimingSummary | None
    warm: TimingSummary | None
    dynamic: TimingSummary | None
    warm_rounds: tuple[TimingSummary, ...] | None = None
    dynamic_rounds: tuple[TimingSummary, ...] | None = None


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One benchmark case run across libraries."""

    case: BenchmarkCase
    runs: dict[str, LibraryRun]
    round_orders: list[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class TestResult:
    """Top-level benchmark result payload."""

    title: str
    configuration: list[str]
    methodology: list[str]
    case_results: list[CaseResult]
    notes: list[str]
