from dataclasses import dataclass
from typing import Literal

from einf.analysis.checkers import CheckerFailure
from einf.analysis.report import AnalysisFileReport


@dataclass(frozen=True, slots=True)
class ValidationDiscoveryFailure:
    """Failure encountered while discovering files under one validation target."""

    path: str
    kind: Literal["directory_traversal_error"]
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Stable machine-readable validator output bundle."""

    schema_version: str
    parser_backend: str
    checker_failures: tuple[CheckerFailure, ...]
    discovery_failures: tuple[ValidationDiscoveryFailure, ...]
    files: tuple[AnalysisFileReport, ...]

    def exit_code(self) -> int:
        """Return process exit code for this validation report."""
        if self.checker_failures or self.discovery_failures:
            return 1
        return 1 if any(file_report.has_errors() for file_report in self.files) else 0


__all__ = [
    "ValidationDiscoveryFailure",
    "ValidationReport",
]
