from dataclasses import dataclass
from typing import Literal

from einf.analysis.checkers import CheckerFailure
from einf.analysis.report import AnalysisFileReport
from einf.analysis.validator.limits import ValidationLimitFailure


@dataclass(frozen=True, slots=True)
class ValidationDiscoveryFailure:
    """Failure encountered while discovering files under one validation target."""

    path: str
    kind: Literal["directory_traversal_error"]
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Collect validator diagnostics and failures, including incomplete work.

    Parameters
    ----------
    schema_version : str
        Version of the serialized output contract.
    parser_backend : str
        Requested semantic parser.
    checker_failures : tuple[CheckerFailure, ...]
        External checker failures.
    discovery_failures : tuple[ValidationDiscoveryFailure, ...]
        Directory traversal errors.
    files : tuple[AnalysisFileReport, ...]
        Completed file reports; not necessarily the entire requested target set.
    limit_failures : tuple[ValidationLimitFailure, ...], optional
        Resource quotas that stopped the invocation before completion.
    """

    schema_version: str
    parser_backend: str
    checker_failures: tuple[CheckerFailure, ...]
    discovery_failures: tuple[ValidationDiscoveryFailure, ...]
    files: tuple[AnalysisFileReport, ...]
    limit_failures: tuple[ValidationLimitFailure, ...] = ()

    def exit_code(self) -> int:
        """Return the process exit code for this report.

        Returns
        -------
        int
            0 for clean validation, or 1 for diagnostics or failures.
        """
        if self.checker_failures or self.discovery_failures or self.limit_failures:
            return 1
        return 1 if any(file_report.has_errors() for file_report in self.files) else 0


__all__ = [
    "ValidationDiscoveryFailure",
    "ValidationReport",
]
