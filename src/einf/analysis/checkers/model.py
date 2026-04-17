from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from einf.analysis.model import DiagnosticSeverity, TextSpan

CheckerFailureKind = Literal["unavailable", "execution_error", "output_parse_error"]


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
    """Normalized result bundle returned by one checker adapter."""

    diagnostics: tuple[CheckerDiagnostic, ...]
    failures: tuple[CheckerFailure, ...]


__all__ = [
    "CheckerDiagnostic",
    "CheckerFailure",
    "CheckerFailureKind",
    "CheckerResult",
]
