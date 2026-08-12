from abc import ABC, abstractmethod

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class AxisTermBase(ABC):
    """Abstract base for structural axis terms."""

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


class ScalarAxisTermBase(AxisTermBase):
    """Abstract base for scalar axis expressions."""

    @abstractmethod
    def evaluate(self, axis_sizes: dict[str, int]) -> int | None:
        """Evaluate the term under scalar-axis assignments.

        Parameters
        ----------
        axis_sizes
            Resolved sizes keyed by axis name.

        Returns
        -------
        int | None
            The evaluated size, or ``None`` if an axis remains unresolved.
        """

    @abstractmethod
    def max_literal(self) -> int:
        """Return the largest integer literal in the term.

        Returns
        -------
        int
            The largest literal, or zero when the term has no literals.
        """

    @abstractmethod
    def evaluate_bounds(
        self,
        *,
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> tuple[int, int]:
        """Return attainable bounds under partial assignments.

        Parameters
        ----------
        current
            Resolved sizes keyed by axis name.
        variable_bounds
            Maximum sizes for unresolved axes.

        Returns
        -------
        tuple[int, int]
            The minimum and maximum attainable values.
        """

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
