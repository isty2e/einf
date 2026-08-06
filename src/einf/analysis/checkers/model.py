from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from einf.analysis.model import DiagnosticSeverity, TextSpan

CheckerFailureKind = Literal[
    "unavailable",
    "spawn_error",
    "timeout",
    "execution_error",
    "output_parse_error",
    "output_limit_exceeded",
]


@dataclass(frozen=True, slots=True)
class CheckerRequest:
    """Canonical target set for one external checker invocation."""

    targets: tuple[Path, ...]
    project_root: Path

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("checker request requires at least one target")


@dataclass(frozen=True, slots=True)
class CheckerOutputLimits:
    """Bounds enforced while parsing one checker's output."""

    max_diagnostics: int
    max_field_length: int


@dataclass(frozen=True, slots=True)
class CheckerDiagnostic:
    """Normalized external checker diagnostic."""

    tool: str
    path: Path
    code: str | None
    message: str
    severity: DiagnosticSeverity
    span: TextSpan | None


@dataclass(frozen=True, slots=True)
class CheckerFailure:
    """Normalized checker invocation failure."""

    tool: str
    kind: CheckerFailureKind
    message: str


@dataclass(frozen=True, slots=True)
class CheckerResult:
    """Normalized result bundle returned by one or more checker adapters."""

    diagnostics: tuple[CheckerDiagnostic, ...]
    failures: tuple[CheckerFailure, ...]

    @classmethod
    def merge(cls, results: Iterable["CheckerResult"]) -> "CheckerResult":
        """Merge adapter results into one deterministic checker result."""
        diagnostics: list[CheckerDiagnostic] = []
        failures: list[CheckerFailure] = []
        for result in results:
            diagnostics.extend(result.diagnostics)
            failures.extend(result.failures)
        diagnostics.sort(key=_diagnostic_sort_key)
        return cls(diagnostics=tuple(diagnostics), failures=tuple(failures))

    def diagnostics_for(self, path: Path) -> tuple[CheckerDiagnostic, ...]:
        """Return diagnostics belonging to one normalized target path."""
        normalized_path = path.resolve(strict=False)
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.path == normalized_path
        )


def _diagnostic_sort_key(
    diagnostic: CheckerDiagnostic,
) -> tuple[str, int, int, str, str, str]:
    span = diagnostic.span
    return (
        str(diagnostic.path),
        span.start.line if span is not None else -1,
        span.start.column if span is not None else -1,
        diagnostic.tool,
        diagnostic.code or "",
        diagnostic.message,
    )


__all__ = [
    "CheckerDiagnostic",
    "CheckerFailure",
    "CheckerFailureKind",
    "CheckerRequest",
    "CheckerResult",
]
