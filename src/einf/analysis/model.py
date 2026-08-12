from dataclasses import dataclass
from typing import Literal

DiagnosticSeverity = Literal["error", "warning", "info"]
AxisStructuralKind = Literal["axis", "pack"]
AxisOccurrenceSide = Literal["lhs", "rhs"]
AxisCrossSideRelation = Literal["shared", "side_only"]
AxisOperationRole = Literal["introduced", "reduced", "contracted"]

_AXIS_STRUCTURAL_KINDS = frozenset({"axis", "pack"})
_AXIS_OCCURRENCE_SIDES = frozenset({"lhs", "rhs"})
_AXIS_CROSS_SIDE_RELATIONS = frozenset({"shared", "side_only"})
_AXIS_OPERATION_ROLES = frozenset({"introduced", "reduced", "contracted"})


@dataclass(frozen=True, slots=True)
class TextPosition:
    """One canonical source position.

    Parameters
    ----------
    line : int
        One-based source line. Boolean values are not accepted.
    column : int
        Zero-based source column. Boolean values are not accepted.

    Raises
    ------
    TypeError
        If either coordinate is not an integer or is a boolean.
    ValueError
        If ``line`` is less than 1 or ``column`` is negative.
    """

    line: int
    column: int

    def __post_init__(self) -> None:
        if isinstance(self.line, bool) or not isinstance(self.line, int):
            raise TypeError("text position line must be an integer")
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("text position column must be an integer")
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
    """Canonical axis occurrence metadata for analysis consumers."""

    name: str
    kind: AxisStructuralKind
    side: AxisOccurrenceSide
    relation: AxisCrossSideRelation
    role: AxisOperationRole | None
    span: TextSpan
    group: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("axis token name must be a string")
        if not self.name.isidentifier():
            raise ValueError("axis token name must be a valid identifier")
        if self.kind not in _AXIS_STRUCTURAL_KINDS:
            raise ValueError(f"unsupported axis structural kind: {self.kind!r}")
        if self.side not in _AXIS_OCCURRENCE_SIDES:
            raise ValueError(f"unsupported axis occurrence side: {self.side!r}")
        if self.relation not in _AXIS_CROSS_SIDE_RELATIONS:
            raise ValueError(f"unsupported axis cross-side relation: {self.relation!r}")
        if self.role is not None and self.role not in _AXIS_OPERATION_ROLES:
            raise ValueError(f"unsupported axis operation role: {self.role!r}")
        if isinstance(self.group, bool) or not isinstance(self.group, int):
            raise TypeError("axis token group must be an integer")
        if self.group < 0:
            raise ValueError("axis token group must be non-negative")
        if self.role is not None and self.relation != "side_only":
            raise ValueError("axis operation roles require a side-only relation")
        if self.role in {"contracted", "reduced"} and self.side != "lhs":
            raise ValueError(f"{self.role} axes must occur on lhs")
        if self.role == "introduced" and self.side != "rhs":
            raise ValueError("introduced axes must occur on rhs")
