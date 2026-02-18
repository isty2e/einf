from dataclasses import dataclass
from typing import Literal

from .base import AxisTermBase, ScalarAxisTermBase


def _validate_identifier(name: str, *, kind: str) -> None:
    """Validate one symbol name."""
    if not name:
        raise ValueError(f"{kind} name cannot be empty")
    if not name.isidentifier():
        raise ValueError(f"{kind} name must be a valid identifier: {name!r}")


@dataclass(frozen=True, slots=True)
class Axis(ScalarAxisTermBase):
    """Named symbolic axis used to build transform signatures."""

    name: str

    def __post_init__(self) -> None:
        _validate_identifier(self.name, kind="axis")

    def to_dsl(self) -> str:
        """Render this axis as DSL text."""
        return self.name

    def stable_token(self) -> str:
        """Return deterministic structural token for ordering and keys."""
        return f"axis:{self.name}"

    def axis_names(self) -> set[str]:
        """Return scalar-axis names referenced by this term."""
        return {self.name}

    def pack_names(self) -> set[str]:
        """Return axis-pack names referenced by this term."""
        return set()

    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate this term under scalar-axis assignments."""
        return axis_sizes.get(self.name)

    def max_literal(self) -> int:
        """Return max integer literal contained in this term."""
        return 0

    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return min/max attainable values under partial assignments."""
        if self.name in current:
            value = current[self.name]
            return value, value
        bound = variable_bounds[self.name]
        return 0, bound

    def __add__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("+", self, ScalarAxisTermBase.coerce(other))

    def __radd__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("+", ScalarAxisTermBase.coerce(other), self)

    def __mul__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("*", self, ScalarAxisTermBase.coerce(other))

    def __rmul__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("*", ScalarAxisTermBase.coerce(other), self)


@dataclass(frozen=True, slots=True)
class AxisPack(AxisTermBase):
    """Named variadic axis pack used in signature patterns."""

    name: str

    def __post_init__(self) -> None:
        _validate_identifier(self.name, kind="axis pack")

    def to_dsl(self) -> str:
        """Render this axis pack as DSL text."""
        return f"*{self.name}"

    def stable_token(self) -> str:
        """Return deterministic structural token for ordering and keys."""
        return f"pack:{self.name}"

    def axis_names(self) -> set[str]:
        """Return scalar-axis names referenced by this term."""
        return set()

    def pack_names(self) -> set[str]:
        """Return axis-pack names referenced by this term."""
        return {self.name}

    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate this term under scalar-axis assignments."""
        _ = axis_sizes
        return None

    def max_literal(self) -> int:
        """Return max integer literal contained in this term."""
        return 0

    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return min/max attainable values under partial assignments."""
        _ = current
        _ = variable_bounds
        raise TypeError("axis packs are not valid in scalar expressions")


@dataclass(frozen=True, slots=True)
class AxisExpr(ScalarAxisTermBase):
    """Binary axis expression composed with '+' or '*'."""

    operator: Literal["+", "*"]
    left: "ScalarAxisTermBase"
    right: "ScalarAxisTermBase"

    def __post_init__(self) -> None:
        if self.operator not in {"+", "*"}:
            raise ValueError(f"unsupported axis operator: {self.operator!r}")

        object.__setattr__(self, "left", ScalarAxisTermBase.coerce(self.left))
        object.__setattr__(self, "right", ScalarAxisTermBase.coerce(self.right))

    def to_dsl(self) -> str:
        """Render this axis expression as DSL text."""
        left = self.left.to_dsl()
        right = self.right.to_dsl()
        return f"({left} {self.operator} {right})"

    def stable_token(self) -> str:
        """Return deterministic structural token for ordering and keys."""
        left = self.left.stable_token()
        right = self.right.stable_token()
        return f"expr:{self.operator}({left},{right})"

    def axis_names(self) -> set[str]:
        """Return scalar-axis names referenced by this expression."""
        return self.left.axis_names() | self.right.axis_names()

    def pack_names(self) -> set[str]:
        """Return axis-pack names referenced by this expression."""
        return set()

    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate this term under scalar-axis assignments."""
        left_value = self.left.evaluate(axis_sizes)
        right_value = self.right.evaluate(axis_sizes)
        if left_value is None or right_value is None:
            return None
        if self.operator == "+":
            return left_value + right_value
        return left_value * right_value

    def max_literal(self) -> int:
        """Return max integer literal contained in this term."""
        return max(self.left.max_literal(), self.right.max_literal())

    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return min/max attainable values under partial assignments."""
        left_min, left_max = self.left.evaluate_bounds(
            current=current,
            variable_bounds=variable_bounds,
        )
        right_min, right_max = self.right.evaluate_bounds(
            current=current,
            variable_bounds=variable_bounds,
        )
        if self.operator == "+":
            return left_min + right_min, left_max + right_max
        return left_min * right_min, left_max * right_max

    def __add__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("+", self, ScalarAxisTermBase.coerce(other))

    def __radd__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("+", ScalarAxisTermBase.coerce(other), self)

    def __mul__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("*", self, ScalarAxisTermBase.coerce(other))

    def __rmul__(self, other: "ScalarAxisTermBase | int") -> "AxisExpr":
        return AxisExpr("*", ScalarAxisTermBase.coerce(other), self)


@dataclass(frozen=True, slots=True)
class AxisInt(ScalarAxisTermBase):
    """Literal non-negative axis extent."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("axis literal must be a non-negative int")
        if self.value < 0:
            raise ValueError("axis literal must be non-negative")

    def to_dsl(self) -> str:
        """Render this term as DSL text."""
        return str(self.value)

    def stable_token(self) -> str:
        """Return deterministic structural token for ordering and keys."""
        return f"int:{self.value}"

    def axis_names(self) -> set[str]:
        """Return scalar-axis names referenced by this term."""
        return set()

    def pack_names(self) -> set[str]:
        """Return axis-pack names referenced by this term."""
        return set()

    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate this term under scalar-axis assignments."""
        _ = axis_sizes
        return self.value

    def max_literal(self) -> int:
        """Return max integer literal contained in this term."""
        return self.value

    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return min/max attainable values under partial assignments."""
        _ = current
        _ = variable_bounds
        return self.value, self.value
