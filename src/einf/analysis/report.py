from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from einf.analysis.checkers import CheckerDiagnostic
from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextSpan
from einf.analysis.parser import ParserSyntaxError, ParserUnavailableError

AnalysisFailureKind = Literal["read_error", "parse_error", "parser_unavailable"]


@dataclass(frozen=True, slots=True)
class AnalysisFailure:
    """Per-file analysis failure outside DSL semantic diagnostics."""

    kind: AnalysisFailureKind
    message: str
    span: TextSpan | None


@dataclass(frozen=True, slots=True)
class AnalysisFileReport:
    """Canonical analysis result for one analyzed path or in-memory document."""

    path: str
    diagnostics: tuple[AnalysisDiagnostic, ...]
    checker_diagnostics: tuple[CheckerDiagnostic, ...]
    axis_tokens: tuple[AxisToken, ...]
    failures: tuple[AnalysisFailure, ...]

    def has_errors(self) -> bool:
        """Return whether this file report contains diagnostics or failures."""
        return bool(self.diagnostics or self.checker_diagnostics or self.failures)


def parse_error_report(
    *,
    path: Path,
    error: ParserSyntaxError,
) -> AnalysisFileReport:
    """Build one per-file report for a backend syntax error."""
    return AnalysisFileReport(
        path=str(path),
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=(
            AnalysisFailure(
                kind="parse_error",
                message=error.message,
                span=error.span,
            ),
        ),
    )


def parser_unavailable_report(
    *,
    path: Path,
    error: ParserUnavailableError,
) -> AnalysisFileReport:
    """Build one per-file report for an unavailable parser backend."""
    return AnalysisFileReport(
        path=str(path),
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=(
            AnalysisFailure(
                kind="parser_unavailable",
                message=error.message,
                span=None,
            ),
        ),
    )


__all__ = [
    "AnalysisFailure",
    "AnalysisFailureKind",
    "AnalysisFileReport",
    "parse_error_report",
    "parser_unavailable_report",
]
