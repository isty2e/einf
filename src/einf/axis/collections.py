from collections.abc import Iterator

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from .base import AxisTermBase, ScalarAxisTermBase
from .terms import Axis, AxisInt, AxisPack


class AxisTerms(tuple[AxisTermBase, ...]):
    """Immutable axis-term tuple value object with multiset helpers."""

    @classmethod
    def from_spec(
        cls,
        spec: "AxisTerms | AxisTermBase | int | tuple[AxisTermBase | int, ...]",
    ) -> Self:
        """Build normalized axis terms from one constructor-level spec."""
        if isinstance(spec, cls):
            return spec.normalize()
        if isinstance(spec, tuple):
            normalized_terms = tuple(cls.coerce(term) for term in spec)
            return cls(normalized_terms)
        normalized_term = cls.coerce(spec)
        return cls((normalized_term,))

    @staticmethod
    def coerce(term: AxisTermBase | int) -> AxisTermBase:
        """Validate and normalize one axis term for `AxisTerms`."""
        if isinstance(term, AxisPack):
            return term
        return ScalarAxisTermBase.coerce(term)

    def to_dsl(self) -> str:
        """Render this axis-term tuple in `ax[...]` DSL form."""
        if not self:
            return "ax[()]"

        rendered = ", ".join(term.to_dsl() for term in self)
        return f"ax[{rendered}]"

    def normalize(self) -> Self:
        """Validate terms and return normalized axis-term tuple."""
        if not self:
            return type(self)(())

        normalized: list[AxisTermBase] = []
        for term in self:
            try:
                normalized_term = type(self).coerce(term)
            except TypeError as exc:
                raise TypeError(
                    "axis term must be Axis, AxisExpr, AxisPack, or non-negative int"
                ) from exc
            normalized.append(normalized_term)

        return type(self)(tuple(normalized))

    def term_counts(self) -> dict[AxisTermBase, int]:
        """Count occurrences of each axis term."""
        counts: dict[AxisTermBase, int] = {}
        for term in self:
            if term in counts:
                counts[term] += 1
                continue
            counts[term] = 1
        return counts

    def __sub__(self, other: Self) -> Self:
        """Return ordered multiset difference `self - other`."""
        remaining = list(other)
        difference: list[AxisTermBase] = []
        for term in self:
            try:
                matched_index = remaining.index(term)
            except ValueError:
                difference.append(term)
                continue
            del remaining[matched_index]
        return type(self)(tuple(difference))

    def __and__(self, other: Self) -> Self:
        """Return ordered multiset intersection preserving lhs order."""
        remaining = list(other)
        shared: list[AxisTermBase] = []
        for term in self:
            try:
                matched_index = remaining.index(term)
            except ValueError:
                continue
            shared.append(term)
            del remaining[matched_index]
        return type(self)(tuple(shared))


class ScalarAxisTerms(AxisTerms):
    """Immutable scalar-axis tuple value object (packs are disallowed)."""

    @staticmethod
    def coerce(term: AxisTermBase | int) -> ScalarAxisTermBase:
        """Validate and normalize one scalar axis term for `ScalarAxisTerms`."""
        return ScalarAxisTermBase.coerce(term)

    def __iter__(self) -> Iterator[ScalarAxisTermBase]:
        """Iterate normalized scalar terms."""
        for term in super().__iter__():
            yield ScalarAxisTermBase.coerce(term)

    def is_shape_bindable(self) -> bool:
        """Return whether every term can bind directly from tensor shape dims."""
        return all(isinstance(term, (Axis, AxisInt)) for term in self)


class AxisSide(tuple[AxisTerms, ...]):
    """Immutable side value object storing one or more axis-term tuples."""

    @classmethod
    def coerce(
        cls,
        side: tuple[AxisTerms, ...],
    ) -> Self:
        """Validate and normalize one side to canonical axis-term tuples."""
        try:
            side_items = tuple(side)
        except TypeError as exc:
            raise TypeError("axis side must be tuple[AxisTerms, ...]") from exc

        if not side_items:
            raise ValueError("axis side must define at least one axis-term tuple")

        normalized_side: list[AxisTerms] = []
        for axis_terms in side_items:
            if not isinstance(axis_terms, AxisTerms):
                raise TypeError("axis side entries must be AxisTerms")
            normalized_side.append(axis_terms.normalize())

        return cls(tuple(normalized_side))

    @classmethod
    def from_spec(
        cls,
        spec: AxisTerms | tuple[AxisTerms | tuple[AxisTermBase | int, ...], ...],
        *,
        side_name: str,
    ) -> Self:
        """Normalize one entry-point side spec (`ax[...]` or tuple of them)."""
        if isinstance(spec, AxisTerms):
            return cls((spec.normalize(),))

        if spec == ():
            return cls((AxisTerms(()),))

        normalized_side: list[AxisTerms] = []
        for axis_terms in spec:
            try:
                normalized_side.append(AxisTerms.from_spec(axis_terms))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{side_name} specification must be ax[...] or tuple[ax[...], ...]"
                ) from exc

        return cls(tuple(normalized_side))

    def normalize(self) -> Self:
        """Normalize this side to canonical axis-term tuples."""
        return type(self).coerce(self)

    def symbol_names(self) -> tuple[set[str], set[str]]:
        """Collect scalar-axis names and pack names used by this side."""
        axis_names: set[str] = set()
        pack_names: set[str] = set()
        for axis_terms in self:
            for term in axis_terms:
                axis_names.update(term.axis_names())
                pack_names.update(term.pack_names())
        return axis_names, pack_names

    def axis_names(self) -> set[str]:
        """Collect scalar-axis names used by this side."""
        axis_names, _ = self.symbol_names()
        return axis_names

    def pack_names(self) -> set[str]:
        """Collect pack names used by this side."""
        _, pack_names = self.symbol_names()
        return pack_names
