from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from .base import ScalarAxisTermBase
from .terms import Axis, AxisExpr, AxisInt


@dataclass(frozen=True, slots=True)
class CanonicalMonomial:
    """One normalized monomial term in a scalar-axis polynomial."""

    coefficient: int
    factors: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.coefficient, bool) or not isinstance(self.coefficient, int):
            raise TypeError("monomial coefficient must be int")
        if self.coefficient <= 0:
            raise ValueError("monomial coefficient must be positive")

        if any(not isinstance(name, str) for name in self.factors):
            raise TypeError("monomial factors must be tuple[str, ...]")
        if tuple(sorted(self.factors)) != self.factors:
            raise ValueError("monomial factors must be sorted")

    def render(self) -> str:
        """Render one monomial as deterministic text."""
        if not self.factors:
            return str(self.coefficient)
        product = "*".join(self.factors)
        if self.coefficient == 1:
            return product
        return f"{self.coefficient}*{product}"


def _sorted_factor_key(factors: tuple[str, ...]) -> tuple[str, ...]:
    """Return sorted factor key for one monomial."""
    return tuple(sorted(factors))


def _normalize_monomials(
    monomials: Iterable[CanonicalMonomial],
) -> tuple[CanonicalMonomial, ...]:
    """Merge like terms and return canonical monomial tuple."""
    merged: dict[tuple[str, ...], int] = {}
    for monomial in monomials:
        if monomial.coefficient == 0:
            continue
        factors = _sorted_factor_key(monomial.factors)
        merged[factors] = merged.get(factors, 0) + monomial.coefficient

    normalized = [
        CanonicalMonomial(coefficient=coefficient, factors=factors)
        for factors, coefficient in merged.items()
        if coefficient > 0
    ]
    normalized.sort(key=lambda monomial: (monomial.factors, monomial.coefficient))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CanonicalScalarExpr:
    """Canonical polynomial form for scalar axis expressions.

    The representation encodes expressions over non-negative scalar variables
    with operators `+` and `*`, normalized up to associativity, commutativity,
    and distributivity.
    """

    monomials: tuple[CanonicalMonomial, ...]

    def __post_init__(self) -> None:
        normalized = _normalize_monomials(self.monomials)
        object.__setattr__(self, "monomials", normalized)

    @classmethod
    def zero(cls) -> "CanonicalScalarExpr":
        """Return additive identity expression."""
        return cls(())

    @classmethod
    def from_term(cls, term: ScalarAxisTermBase) -> "CanonicalScalarExpr":
        """Build canonical form from one scalar axis term."""
        return _canonicalize_term(term)

    @classmethod
    def sum_terms(cls, terms: Iterable[ScalarAxisTermBase]) -> "CanonicalScalarExpr":
        """Build canonical sum from one scalar-axis term sequence."""
        result = cls.zero()
        for term in terms:
            result = result + cls.from_term(term)
        return result

    def __add__(self, other: "CanonicalScalarExpr") -> "CanonicalScalarExpr":
        """Return canonical sum."""
        return type(self)(self.monomials + other.monomials)

    def __mul__(self, other: "CanonicalScalarExpr") -> "CanonicalScalarExpr":
        """Return canonical product."""
        if not self.monomials or not other.monomials:
            return type(self).zero()

        products: list[CanonicalMonomial] = []
        for left in self.monomials:
            for right in other.monomials:
                products.append(
                    CanonicalMonomial(
                        coefficient=left.coefficient * right.coefficient,
                        factors=_sorted_factor_key(left.factors + right.factors),
                    )
                )
        return type(self)(tuple(products))

    def stable_token(self) -> str:
        """Return deterministic canonical token."""
        if not self.monomials:
            return "0"
        return "+".join(monomial.render() for monomial in self.monomials)

    def axis_names(self) -> set[str]:
        """Return variable names referenced by this canonical expression."""
        names: set[str] = set()
        for monomial in self.monomials:
            names.update(monomial.factors)
        return names


@lru_cache(maxsize=16_384)
def _canonicalize_term(term: ScalarAxisTermBase) -> CanonicalScalarExpr:
    """Canonicalize one scalar term with memoization."""
    if isinstance(term, AxisInt):
        if term.value == 0:
            return CanonicalScalarExpr.zero()
        return CanonicalScalarExpr((CanonicalMonomial(term.value, ()),))
    if isinstance(term, Axis):
        return CanonicalScalarExpr((CanonicalMonomial(1, (term.name,)),))
    if isinstance(term, AxisExpr):
        left = _canonicalize_term(term.left)
        right = _canonicalize_term(term.right)
        if term.operator == "+":
            return left + right
        if term.operator == "*":
            return left * right
    raise TypeError("unsupported scalar term for canonicalization")


__all__ = [
    "CanonicalMonomial",
    "CanonicalScalarExpr",
]
