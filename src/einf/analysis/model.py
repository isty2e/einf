from dataclasses import dataclass
from typing import Literal

DiagnosticSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class TextPosition:
    """One source position using 1-based line and 0-based column indexing."""

    line: int
    column: int

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError("text position line must be >= 1")
        if self.column < 0:
            raise ValueError("text position column must be >= 0")


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Half-open source span [start, end)."""

    start: TextPosition
    end: TextPosition

    def __post_init__(self) -> None:
        starts_after_end = self.start.line > self.end.line or (
            self.start.line == self.end.line and self.start.column > self.end.column
        )
        if starts_after_end:
            raise ValueError("text span start must not be after end")


@dataclass(frozen=True, slots=True)
class AnalysisDiagnostic:
    """Static-analysis diagnostic for one source span."""

    code: str
    message: str
    severity: DiagnosticSeverity
    span: TextSpan | None


@dataclass(frozen=True, slots=True)
class AxisToken:
    """Axis token metadata for editor rendering."""

    name: str
    span: TextSpan
    group: int
    roles: tuple[str, ...]
