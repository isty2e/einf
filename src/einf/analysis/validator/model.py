from dataclasses import dataclass
from typing import Literal

from einf.analysis.checkers import CheckerDiagnostic, CheckerFailure
from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextSpan

ValidationFailureKind = Literal["read_error", "parse_error"]


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """Validator-level file failure outside DSL semantic diagnostics."""

    kind: ValidationFailureKind
    message: str
    span: TextSpan | None


@dataclass(frozen=True, slots=True)
class ValidationFileReport:
    """Machine-readable validation result for one analyzed path."""

    path: str
    diagnostics: tuple[AnalysisDiagnostic, ...]
    checker_diagnostics: tuple[CheckerDiagnostic, ...]
    axis_tokens: tuple[AxisToken, ...]
    failures: tuple[ValidationFailure, ...]

    def has_errors(self) -> bool:
        """Return whether this file report contains diagnostics or failures."""
        return bool(self.diagnostics or self.checker_diagnostics or self.failures)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Stable machine-readable validator output bundle."""

    schema_version: str
    parser_backend: str
    checker_failures: tuple[CheckerFailure, ...]
    files: tuple[ValidationFileReport, ...]

    def exit_code(self) -> int:
        """Return process exit code for this validation report."""
        if self.checker_failures:
            return 1
        return 1 if any(file_report.has_errors() for file_report in self.files) else 0


__all__ = [
    "ValidationFailure",
    "ValidationFailureKind",
    "ValidationFileReport",
    "ValidationReport",
]
