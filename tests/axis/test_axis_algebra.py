import pytest

import einf.axis.algebra as axis_algebra_module
from einf import ErrorCode, ValidationError, axes
from einf.axis import CanonicalMonomial, CanonicalScalarExpr, ScalarAxisTermBase


def _distinct_monomials(count: int, *, prefix: str) -> tuple[CanonicalMonomial, ...]:
    return tuple(
        CanonicalMonomial(coefficient=1, factors=(f"{prefix}{index:04d}",))
        for index in range(count)
    )


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


def test_canonical_scalar_expr_accepts_exact_product_candidate_limit() -> None:
    left = CanonicalScalarExpr(
        _distinct_monomials(
            axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES // 2,
            prefix="left",
        )
    )
    right = CanonicalScalarExpr(_distinct_monomials(2, prefix="right"))

    product = left * right

    assert len(product.monomials) == (
        axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES
    )


def test_canonical_scalar_expr_rejects_before_product_factor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = CanonicalScalarExpr(
        _distinct_monomials(
            axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES,
            prefix="left",
        )
    )
    right = CanonicalScalarExpr(_distinct_monomials(2, prefix="right"))

    def fail_if_product_factor_is_built(
        factors: tuple[str, ...],
    ) -> tuple[str, ...]:
        _ = factors
        raise AssertionError("product factor construction must not run")

    monkeypatch.setattr(
        axis_algebra_module,
        "_sorted_factor_key",
        fail_if_product_factor_is_built,
    )

    with pytest.raises(ValidationError) as error:
        _ = left * right

    captured = error.value
    assert captured.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value
    assert captured.data == {
        "complexity_kind": "distributive_product_candidates",
        "limit": axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES,
        "attempted": 2 * axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES,
    }


def test_canonical_scalar_expr_does_not_limit_large_linear_sum() -> None:
    candidate_limit = axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES
    left = CanonicalScalarExpr(
        _distinct_monomials(candidate_limit, prefix="linear_left")
    )
    right = CanonicalScalarExpr(_distinct_monomials(1, prefix="linear_right"))

    expression = left + right

    assert len(expression.monomials) == candidate_limit + 1


def test_canonical_scalar_expr_zero_product_bypasses_candidate_limit() -> None:
    candidate_limit = axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES
    large = CanonicalScalarExpr(
        _distinct_monomials(candidate_limit + 1, prefix="large")
    )

    product = large * CanonicalScalarExpr.zero()

    assert product == CanonicalScalarExpr.zero()
