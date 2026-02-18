from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class ShapeLiteral:
    """Compiled shape node for one constant dimension."""

    value: int


@dataclass(frozen=True, slots=True)
class ShapeDimRef:
    """Compiled shape node for one input-shape index reference."""

    index: int


@dataclass(frozen=True, slots=True)
class ShapeAxisName:
    """Compiled shape node for one explicit-only axis symbol."""

    name: str


@dataclass(frozen=True, slots=True)
class ShapeBinary:
    """Compiled shape node for one binary arithmetic expression."""

    operator: str
    left: "ShapeNode"
    right: "ShapeNode"


ShapeNode: TypeAlias = ShapeLiteral | ShapeDimRef | ShapeAxisName | ShapeBinary


__all__ = [
    "ShapeAxisName",
    "ShapeBinary",
    "ShapeDimRef",
    "ShapeLiteral",
    "ShapeNode",
]
