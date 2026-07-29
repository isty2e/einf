from contextlib import contextmanager

import pytest

from einf import ErrorCode, Signature, ValidationError, ax, axes
from einf.solver.equations import EquationSolver
from einf.solver.matching import PartialState, ShapeMatcher
from einf.solver.search import DimSearch


def _build_search_with_work_limit(
    signature: Signature,
    *,
    input_shapes: tuple[tuple[int, ...], ...],
    work_limit: int,
) -> DimSearch:
    """Build one dimension search with a deterministic test work limit."""
    return DimSearch(
        signature=signature,
        normalized_shapes=input_shapes,
        initial_state=PartialState(axis_sizes={}, pack_sizes={}, equations=()),
        matcher=ShapeMatcher(),
        equation_solver=EquationSolver(
            axis_names=signature.axis_names(),
            pack_names=signature.pack_names(),
            shapes=input_shapes,
            work_limit=work_limit,
        ),
    )


@contextmanager
def _expect_validation_error(code: ErrorCode):
    """Assert a validation error with one expected error code."""
    with pytest.raises(ValidationError) as error:
        yield

    assert error.value.code == code.value
    assert error.value.external_code == code.value.upper()


def test_solver_inverts_large_affine_equation_with_constant_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[((7 * h) + 3)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((7_000_000_003,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000_000}


def test_solver_classifies_large_product_with_bounded_divisor_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((2_147_483_647,),),
        work_limit=5,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_reports_deterministic_fallback_work_limit() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((h * h) + (w * w))],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((3,),),
        work_limit=2,
    )

    with pytest.raises(ValidationError) as error:
        search.run()

    assert error.value.code == ErrorCode.DIM_SOLVE_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "equation_search_work",
        "limit": 2,
        "attempted": 3,
    }


def test_dimension_validation_stops_after_first_feasible_equation_assignment() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((h * h) + (w * w))],), outputs=(ax[1],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1,),),
        work_limit=2,
    )

    assert search.has_feasible_assignment()


def test_solver_classifies_zero_product_with_bounded_fallback_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((0,),),
        work_limit=3,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_classifies_coefficient_product_with_bounded_divisor_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(2 * h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((4_294_967_294,),),
        work_limit=5,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_inverts_large_square_with_constant_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[(h * h)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1_000_000_000_000_000_000,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000_000}


def test_solver_filters_lazy_product_candidates_through_coupled_equation() -> None:
    h, w = axes("h", "w")
    sig = Signature(
        inputs=(ax[(h * w)], ax[(h + w)]),
        outputs=(ax[h, w],),
    )
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((2_147_483_647,), (2_147_483_648,)),
        work_limit=5,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_preserves_ambiguity_for_axis_erased_by_canonical_zero() -> None:
    h, k = axes("h", "k")
    sig = Signature(inputs=(ax[((0 * h) + k)],), outputs=(ax[h, k],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1_000_000_000,),),
        work_limit=4,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_classifies_shifted_large_product_with_bounded_divisor_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((h * w) + 1)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((2_147_483_648,),),
        work_limit=5,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_inverts_large_cube_with_constant_search_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[(h * h * h)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1_000_000_000_000_000_000,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000}


def test_solver_resolves_product_identity_with_bounded_divisor_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1,),),
        work_limit=3,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1, "w": 1}


def test_solver_rejects_nondivisible_product_coefficient_without_search() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(2 * h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((9,),),
        work_limit=0,
    )

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        search.run()


def test_solver_intersects_large_affine_constraints_with_constant_work() -> None:
    (h,) = axes("h")
    sig = Signature(
        inputs=(ax[((7 * h) + 3)], ax[((5 * h) + 1)]),
        outputs=(ax[h],),
    )
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((7_000_000_003,), (5_000_000_001,)),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000_000}


def test_solver_reports_product_work_limit_at_exact_boundary() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((2_147_483_647,),),
        work_limit=4,
    )

    with pytest.raises(ValidationError) as error:
        search.run()

    assert error.value.code == ErrorCode.DIM_SOLVE_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "equation_search_work",
        "limit": 4,
        "attempted": 5,
    }


def test_solver_rejects_large_nonperfect_cube_without_search() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[(h * h * h)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1_000_000_000_000_000_001,),),
        work_limit=0,
    )

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        search.run()


def test_solver_inverts_large_fourth_power_with_constant_search_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[(h * h * h * h)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((10_000_000_000_000_000,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 10_000}


def test_solver_classifies_zero_residual_shifted_product_with_bounded_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((h * w) + 5)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((5,),),
        work_limit=3,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_rejects_positive_product_with_assigned_zero_factor_without_search() -> (
    None
):
    h, k = axes("h", "k")
    sig = Signature(inputs=(ax[((h * k) + 1)], ax[k]), outputs=(ax[h, k],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((2,), (0,)),
        work_limit=0,
    )

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        search.run()


def test_solver_classifies_repeated_factor_product_with_bounded_divisor_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((36,),),
        work_limit=6,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_rejects_conflicting_affine_constraints_without_search() -> None:
    (h,) = axes("h")
    sig = Signature(
        inputs=(ax[((7 * h) + 3)], ax[((5 * h) + 1)]),
        outputs=(ax[h],),
    )
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((73,), (56,)),
        work_limit=0,
    )

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        search.run()


def test_solver_inverts_zero_cube_with_constant_search_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[(h * h * h)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((0,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 0}


def test_solver_resolves_repeated_factor_product_identity_with_bounded_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * h * w)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((1,),),
        work_limit=3,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1, "w": 1}


def test_solver_classifies_shifted_coefficient_product_with_bounded_work() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((3 * h * w) + 7)],), outputs=(ax[h, w],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((6_442_450_948,),),
        work_limit=5,
    )

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        search.run()


def test_solver_inverts_distributive_affine_expression_with_constant_work() -> None:
    (h,) = axes("h")
    sig = Signature(inputs=(ax[((h + 2) * 3)],), outputs=(ax[h],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((3_000_000_006,),),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000_000}


def test_solver_combines_assigned_affine_monomials_with_constant_work() -> None:
    h, w, k = axes("h", "w", "k")
    sig = Signature(
        inputs=(ax[((h * w) + (h * k) + 4)], ax[w], ax[k]),
        outputs=(ax[h, w, k],),
    )
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((5_000_000_004,), (2,), (3,)),
        work_limit=1,
    )

    result = search.run()

    assert result.axis_sizes == {"h": 1_000_000_000, "k": 3, "w": 2}


def test_dimension_validation_preserves_fallback_work_limit_diagnostic() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[((h * h) + (w * w))],), outputs=(ax[1],))
    search = _build_search_with_work_limit(
        sig,
        input_shapes=((3,),),
        work_limit=2,
    )

    with pytest.raises(ValidationError) as error:
        search.has_feasible_assignment()

    assert error.value.code == ErrorCode.DIM_SOLVE_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "equation_search_work",
        "limit": 2,
        "attempted": 3,
    }
