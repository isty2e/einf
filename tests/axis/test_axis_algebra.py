from collections.abc import Iterator

import pytest

import einf.axis.algebra as axis_algebra_module
from einf import ErrorCode, ValidationError, axes
from einf.axis import (
    AxisExpr,
    CanonicalMonomial,
    CanonicalScalarExpr,
    ScalarAxisTermBase,
)


def _distinct_monomials(count: int, *, prefix: str) -> tuple[CanonicalMonomial, ...]:
    return tuple(
        CanonicalMonomial(coefficient=1, factors=(f"{prefix}{index:04d}",))
        for index in range(count)
    )


def _eight_candidate_expression(*, prefix: str) -> AxisExpr:
    a, b, c, d, e, f = axes(*(f"{prefix}_{name}" for name in "abcdef"))
    return ((a + b) * (c + d)) * (e + f)


@pytest.fixture
def small_candidate_limit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        axis_algebra_module,
        "_MAX_CANONICAL_PRODUCT_CANDIDATES",
        4,
    )
    axis_algebra_module._canonicalize_term.cache_clear()
    try:
        yield
    finally:
        axis_algebra_module._canonicalize_term.cache_clear()


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


@pytest.mark.parametrize("zero_on_left", [False, True])
def test_canonical_scalar_expr_zero_product_bypasses_candidate_limit(
    *,
    zero_on_left: bool,
) -> None:
    candidate_limit = axis_algebra_module._MAX_CANONICAL_PRODUCT_CANDIDATES
    large = CanonicalScalarExpr(
        _distinct_monomials(candidate_limit + 1, prefix="large")
    )
    zero = CanonicalScalarExpr.zero()

    product = zero * large if zero_on_left else large * zero

    assert product == zero


@pytest.mark.parametrize(
    ("zero_on_left", "nested"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_canonical_scalar_expr_ast_zero_short_circuits_over_limit_operand(
    small_candidate_limit: None,
    *,
    zero_on_left: bool,
    nested: bool,
) -> None:
    _ = small_candidate_limit
    zero_axis = axes("zero_axis")[0]
    over_limit = _eight_candidate_expression(prefix="zero")
    if nested:
        zero_operand = zero_axis * 0
        expression = (
            zero_operand * over_limit if zero_on_left else over_limit * zero_operand
        )
    else:
        expression = 0 * over_limit if zero_on_left else over_limit * 0

    canonical = CanonicalScalarExpr.from_term(expression)

    assert canonical == CanonicalScalarExpr.zero()


@pytest.mark.parametrize("right_fails", [False, True])
def test_canonical_scalar_expr_preserves_left_failure_after_zero_probe(
    small_candidate_limit: None,
    *,
    right_fails: bool,
) -> None:
    _ = small_candidate_limit
    left = _eight_candidate_expression(prefix="left_probe")
    if right_fails:
        a, b, c, d, e = axes(*(f"right_probe_{name}" for name in "abcde"))
        right = ((a + b) + c) * (d + e)
    else:
        right = axes("nonzero_probe")[0]

    with pytest.raises(ValidationError) as error:
        CanonicalScalarExpr.from_term(left * right)

    assert error.value.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "distributive_product_candidates",
        "limit": 4,
        "attempted": 8,
    }
