from abc import ABC, abstractmethod

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class AxisTermBase(ABC):
    """Abstract base for normalized axis terms."""

    @abstractmethod
    def to_dsl(self) -> str:
        """Render this term as DSL text."""

    @abstractmethod
    def stable_token(self) -> str:
        """Return deterministic structural token for ordering and keys."""

    @abstractmethod
    def axis_names(self) -> set[str]:
        """Return scalar-axis names referenced by this term."""

    @abstractmethod
    def pack_names(self) -> set[str]:
        """Return axis-pack names referenced by this term."""

    @abstractmethod
    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate this term under scalar-axis assignments."""

    @abstractmethod
    def max_literal(self) -> int:
        """Return max integer literal contained in this term."""

    @abstractmethod
    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return min/max attainable values under partial assignments."""


class ScalarAxisTermBase(AxisTermBase):
    """Abstract base for scalar (non-pack) axis terms."""

    @classmethod
    def coerce(cls, term: AxisTermBase | int) -> Self:
        """Validate and normalize one scalar axis term."""
        from .terms import AxisInt

        if isinstance(term, int):
            coerced: AxisTermBase = AxisInt(term)
        else:
            coerced = term

        if isinstance(coerced, cls):
            return coerced

        raise TypeError(
            "axis expression terms must be Axis, AxisExpr, or non-negative int"
        )
