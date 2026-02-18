from einf import axes
from einf.axis import CanonicalScalarExpr, ScalarAxisTermBase


def test_canonical_scalar_expr_matches_distributive_equivalence() -> None:
    h1, h2, h3 = axes("h1", "h2", "h3")
    distributed = CanonicalScalarExpr.from_term(
        ScalarAxisTermBase.coerce((h1 + h2) * h3)
    )
    expanded = CanonicalScalarExpr.from_term(
        ScalarAxisTermBase.coerce((h1 * h3) + (h2 * h3))
    )

    assert distributed == expanded


def test_canonical_scalar_expr_matches_commutative_multiplication() -> None:
    h1, h2 = axes("h1", "h2")
    left = CanonicalScalarExpr.from_term(ScalarAxisTermBase.coerce(h1 * h2))
    right = CanonicalScalarExpr.from_term(ScalarAxisTermBase.coerce(h2 * h1))

    assert left == right
    assert left.stable_token() == right.stable_token()


def test_canonical_scalar_expr_detects_non_equivalent_terms() -> None:
    h1, h2, h3 = axes("h1", "h2", "h3")
    left = CanonicalScalarExpr.from_term(ScalarAxisTermBase.coerce((h1 + h2) * h3))
    right = CanonicalScalarExpr.from_term(ScalarAxisTermBase.coerce((h1 * h3) + h2))

    assert left != right
