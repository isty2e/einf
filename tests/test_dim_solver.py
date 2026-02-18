from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from array_api_compat import numpy as array_api_numpy

from einf import ErrorCode, Signature, ValidationError, ax, axes, packs, view
from einf.solver import solve_dimensions


@dataclass(frozen=True, slots=True)
class DummyTensor:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy

    def __getitem__(self, key: object) -> "DummyTensor":
        _ = key
        return self


def _classify_h_w_multiplication_target(target: int) -> str:
    """Brute-force classify h*w=target over small non-negative domain."""
    solutions: list[tuple[int, int]] = []
    for h in range(0, 33):
        for w in range(0, 33):
            if h * w == target:
                solutions.append((h, w))
                if len(solutions) >= 2:
                    return "ambiguous"
    if len(solutions) == 1:
        return "unique"
    return "inconsistent"


def _classify_h_w_addition_target(target: int) -> str:
    """Brute-force classify h+w=target over small non-negative domain."""
    solutions: list[tuple[int, int]] = []
    for h in range(0, 33):
        for w in range(0, 33):
            if h + w == target:
                solutions.append((h, w))
                if len(solutions) >= 2:
                    return "ambiguous"
    if len(solutions) == 1:
        return "unique"
    return "inconsistent"


def _solve_or_classify(sig, input_shapes):
    """Run solver and convert outcomes to category labels."""
    try:
        result = solve_dimensions(sig, input_shapes=input_shapes)
        return "unique", result
    except ValidationError as error:
        if error.code == ErrorCode.AMBIGUOUS_DIMS.value:
            return "ambiguous", None
        if error.code == ErrorCode.INCONSISTENT_DIMS.value:
            return "inconsistent", None
        raise


@contextmanager
def _expect_validation_error(code: ErrorCode):
    """Assert a validation error with one expected error code."""
    with pytest.raises(ValidationError) as error:
        yield

    assert error.value.code == code.value
    assert error.value.external_code == code.value.upper()


def test_solver_resolves_unique_scalar_solution_with_explicit_size() -> None:
    b, h, w, c = axes("b", "h", "w", "c")
    sig = Signature(inputs=(ax[b, (h * w), c],), outputs=(ax[b, h, w, c],))
    result = solve_dimensions(sig, input_shapes=((2, 12, 4),), explicit_sizes={"h": 3})

    assert result.axis_sizes == {"b": 2, "c": 4, "h": 3, "w": 4}
    assert result.pack_sizes == {}


def test_solver_cache_returns_defensive_copies() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    first = solve_dimensions(sig, input_shapes=((3,),))
    first.axis_sizes["b"] = 99
    second = solve_dimensions(sig, input_shapes=((3,),))

    assert second.axis_sizes == {"b": 3}


def test_solver_reports_ambiguous_scalar_solution_without_enough_constraints() -> None:
    b, h, w, c = axes("b", "h", "w", "c")
    sig = Signature(inputs=(ax[b, (h * w), c],), outputs=(ax[b, h, w, c],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((2, 12, 4),))

    assert error.value.code == "ambiguous_dims"
    assert error.value.external_code == "AMBIGUOUS_DIMS"


def test_solver_resolves_repeated_axis_equality_when_dims_match() -> None:
    i = axes("i")[0]
    sig = Signature(inputs=(ax[i, i],), outputs=(ax[i, i],))
    result = solve_dimensions(sig, input_shapes=((4, 4),))

    assert result.axis_sizes == {"i": 4}


def test_solver_reports_inconsistent_for_repeated_axis_equality_mismatch() -> None:
    i = axes("i")[0]
    sig = Signature(inputs=(ax[i, i],), outputs=(ax[i, i],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((4, 5),))

    assert error.value.code == "inconsistent_dims"
    assert error.value.external_code == "INCONSISTENT_DIMS"


def test_solver_reports_inconsistent_scalar_solution_for_conflicting_explicit_size() -> (
    None
):
    b, h, w, c = axes("b", "h", "w", "c")
    sig = Signature(inputs=(ax[b, (h * w), c],), outputs=(ax[b, h, w, c],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((2, 12, 4),), explicit_sizes={"h": 5})

    assert error.value.code == "inconsistent_dims"
    assert error.value.external_code == "INCONSISTENT_DIMS"


def test_solver_accepts_distributive_equivalent_constraints() -> None:
    h1, h2, h3 = axes("h1", "h2", "h3")
    sig = Signature(
        inputs=(ax[((h1 + h2) * h3)], ax[((h1 * h3) + (h2 * h3))]),
        outputs=(ax[h1, h2, h3],),
    )
    result = solve_dimensions(
        sig,
        input_shapes=((9,), (9,)),
        explicit_sizes={"h1": 1, "h2": 2},
    )

    assert result.axis_sizes == {"h1": 1, "h2": 2, "h3": 3}


def test_solver_rejects_conflicting_distributive_equivalent_constraints() -> None:
    h1, h2, h3 = axes("h1", "h2", "h3")
    sig = Signature(
        inputs=(ax[((h1 + h2) * h3)], ax[((h1 * h3) + (h2 * h3))]),
        outputs=(ax[h1, h2, h3],),
    )

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(
            sig,
            input_shapes=((9,), (8,)),
            explicit_sizes={"h1": 1, "h2": 2},
        )


def test_solver_resolves_unique_additive_expression_assignment() -> None:
    h = axes("h")[0]
    sig = Signature(inputs=(ax[(h + 2)],), outputs=(ax[h],))
    result = solve_dimensions(sig, input_shapes=((7,),))

    assert result.axis_sizes == {"h": 5}


def test_solver_reports_inconsistent_for_impossible_additive_expression() -> None:
    h = axes("h")[0]
    sig = Signature(inputs=(ax[(h + 2)],), outputs=(ax[h],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((1,),))


def test_solver_reports_ambiguous_for_additive_two_variable_equation() -> None:
    h, k = axes("h", "k")
    sig = Signature(inputs=(ax[(h + k)],), outputs=(ax[h, k],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((5,),))


def test_solver_resolves_unique_multiplicative_expression_assignment() -> None:
    h = axes("h")[0]
    sig = Signature(inputs=(ax[(h * h)],), outputs=(ax[h],))
    result = solve_dimensions(sig, input_shapes=((9,),))

    assert result.axis_sizes == {"h": 3}


def test_solver_reports_inconsistent_for_multiplicative_fractional_constraint() -> None:
    h, k = axes("h", "k")
    sig = Signature(inputs=(ax[(h * (k + 1))], ax[h]), outputs=(ax[h, k],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((5,), (2,)))


def test_solver_resolves_zero_square_expression_uniquely() -> None:
    h = axes("h")[0]
    sig = Signature(inputs=(ax[(h * h)],), outputs=(ax[h],))
    result = solve_dimensions(sig, input_shapes=((0,),))

    assert result.axis_sizes == {"h": 0}


def test_solver_reports_ambiguous_for_expression_with_infinite_nonnegative_solutions() -> (
    None
):
    h = axes("h")[0]
    sig = Signature(inputs=(ax[(0 * h)],), outputs=(ax[h],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((0,),))

    assert error.value.code == "ambiguous_dims"
    assert error.value.external_code == "AMBIGUOUS_DIMS"


def test_solver_rejects_float_input_shape_entries() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=((2.0,),))  # type: ignore[arg-type]


def test_solver_rejects_bool_input_shape_entries() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=((True,),))  # type: ignore[arg-type]


def test_solver_resolves_unique_pack_swap_assignment() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 3, t2],), outputs=(ax[t2, 3, t1],))
    result = solve_dimensions(sig, input_shapes=((2, 3, 4),))

    assert result.axis_sizes == {}
    assert result.pack_sizes == {"T1": (2,), "T2": (4,)}


def test_solver_reports_ambiguous_for_adjacent_pack_variables() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, t2],), outputs=(ax[t1, t2],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((2, 3),))


def test_solver_resolves_adjacent_repeated_pack_when_halves_match() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, t],), outputs=(ax[t, t],))
    result = solve_dimensions(sig, input_shapes=((1, 2, 1, 2),))

    assert result.pack_sizes == {"T": (1, 2)}


def test_solver_reports_inconsistent_for_adjacent_repeated_pack_when_halves_differ() -> (
    None
):
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, t],), outputs=(ax[t, t],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((1, 2, 1, 3),))


def test_solver_uses_explicit_scalar_size_to_disambiguate_pack_split() -> None:
    h = axes("h")[0]
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, h, t2],), outputs=(ax[t2, h, t1],))
    result = solve_dimensions(sig, input_shapes=((2, 5, 4),), explicit_sizes={"h": 5})

    assert result.axis_sizes == {"h": 5}
    assert result.pack_sizes == {"T1": (2,), "T2": (4,)}


def test_solver_reports_ambiguous_pack_split_without_explicit_scalar_size() -> None:
    h = axes("h")[0]
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, h, t2],), outputs=(ax[t2, h, t1],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((2, 5, 4),))


def test_solver_reports_ambiguous_for_pack_expression_middle_split_without_explicit_axis() -> (
    None
):
    n = axes("n")[0]
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, (n + 1), t2],), outputs=(ax[n, t1, t2],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((4, 5, 6),))


def test_solver_uses_explicit_axis_to_disambiguate_pack_expression_middle_split() -> (
    None
):
    n = axes("n")[0]
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, (n + 1), t2],), outputs=(ax[n, t1, t2],))
    result = solve_dimensions(
        sig,
        input_shapes=((4, 5, 6),),
        explicit_sizes={"n": 4},
    )

    assert result.axis_sizes == {"n": 4}
    assert result.pack_sizes == {"T1": (4,), "T2": (6,)}


def test_solver_resolves_zero_literal_pack_boundary_uniquely() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 0, t2],), outputs=(ax[t2, 0, t1],))
    result = solve_dimensions(sig, input_shapes=((0,),))

    assert result.pack_sizes == {"T1": (), "T2": ()}


def test_solver_reports_ambiguous_zero_literal_pack_boundary_with_extra_zero_axis() -> (
    None
):
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 0, t2],), outputs=(ax[t2, 0, t1],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((0, 0),))


def test_solver_reports_ambiguous_pack_split_when_delimiter_can_shift() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 3, t2],), outputs=(ax[t2, 3, t1],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((3, 3),))

    assert error.value.code == "ambiguous_dims"
    assert error.value.external_code == "AMBIGUOUS_DIMS"


def test_solver_supports_zero_length_pack_bindings() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 3, t2],), outputs=(ax[t2, 3, t1],))
    result = solve_dimensions(sig, input_shapes=((3,),))

    assert result.pack_sizes == {"T1": (), "T2": ()}


def test_solver_enforces_repeated_pack_value_consistency() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3, t],), outputs=(ax[t, 3, t],))
    result = solve_dimensions(sig, input_shapes=((2, 3, 2),))

    assert result.pack_sizes == {"T": (2,)}


def test_solver_reports_inconsistent_when_repeated_pack_values_conflict() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3, t],), outputs=(ax[t, 3, t],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((2, 3, 4),))

    assert error.value.code == "inconsistent_dims"
    assert error.value.external_code == "INCONSISTENT_DIMS"


def test_solver_enforces_pack_consistency_across_multiple_inputs() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3], ax[t, 4]), outputs=(ax[t, 7],))
    result = solve_dimensions(sig, input_shapes=((2, 3), (2, 4)))

    assert result.pack_sizes == {"T": (2,)}


def test_solver_reports_inconsistent_for_cross_input_pack_conflict() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3], ax[t, 4]), outputs=(ax[t, 7],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((2, 3), (5, 4)))


def test_solver_reports_ambiguous_for_output_only_axis_variable() -> None:
    b, h = axes("b", "h")
    sig = Signature(inputs=(ax[b],), outputs=(ax[b, h],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((5,),))


def test_solver_reports_inconsistent_when_constrained_axes_unsat_even_with_free_output_axis() -> (
    None
):
    a, b, c = axes("a", "b", "c")
    sig = Signature(inputs=(ax[(b + 2), a],), outputs=(ax[a, b, c],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((0, 4),))


def test_solver_reports_ambiguous_when_constraints_sat_but_output_axis_remains_free() -> (
    None
):
    a, b, c = axes("a", "b", "c")
    sig = Signature(inputs=(ax[(b + 2), a],), outputs=(ax[a, b, c],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((3, 4),))


def test_solver_reports_inconsistent_when_output_only_pack_exists_but_constraints_conflict() -> (
    None
):
    a, b = axes("a", "b")
    t = packs("T")[0]
    sig = Signature(inputs=(ax[(a * b), b, (a * b)],), outputs=(ax[t, b],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((4, 1, 2),))


def test_solver_reports_ambiguous_when_output_only_pack_exists_and_constraints_sat() -> (
    None
):
    a, b = axes("a", "b")
    t = packs("T")[0]
    sig = Signature(inputs=(ax[(a * b), b, (a * b)],), outputs=(ax[t, b],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((4, 2, 4),))


def test_solver_reports_ambiguous_for_output_only_pack_variable() -> None:
    b = axes("b")[0]
    t = packs("T")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b, t],))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        solve_dimensions(sig, input_shapes=((5,),))


def test_solver_rejects_unused_explicit_axis_binding() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(ValueError):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"x": 3})


def test_solver_rejects_bool_explicit_axis_binding() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"b": True})


def test_solver_rejects_explicit_binding_for_pack_name() -> None:
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3],), outputs=(ax[t, 3],))

    with pytest.raises(ValueError):
        solve_dimensions(sig, input_shapes=((2, 3),), explicit_sizes={"T": 2})


def test_solver_accepts_explicit_binding_for_output_only_axis() -> None:
    b, h = axes("b", "h")
    sig = Signature(inputs=(ax[b],), outputs=(ax[b, h],))
    result = solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"h": 3})

    assert result.axis_sizes == {"b": 5, "h": 3}


def test_solver_reports_inconsistent_for_conflicting_explicit_input_axis() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"b": 6})


def test_solver_rejects_non_string_explicit_binding_key() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={1: 3})  # type: ignore[dict-item]


def test_solver_rejects_float_explicit_axis_binding() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"b": 5.0})  # type: ignore[arg-type]


def test_solver_rejects_negative_explicit_axis_binding() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(ValueError):
        solve_dimensions(sig, input_shapes=((5,),), explicit_sizes={"b": -1})


def test_solver_rejects_input_shape_arity_mismatch() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b], ax[b]), outputs=(ax[b],))

    with pytest.raises(ValueError):
        solve_dimensions(sig, input_shapes=((5,),))


def test_solver_rejects_non_tuple_input_shape() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(TypeError):
        solve_dimensions(sig, input_shapes=([5],))  # type: ignore[arg-type]


def test_solver_rejects_negative_input_shape_entries() -> None:
    b = axes("b")[0]
    sig = Signature(inputs=(ax[b],), outputs=(ax[b],))

    with pytest.raises(ValueError):
        solve_dimensions(sig, input_shapes=((-1,),))


def test_solver_reports_inconsistent_for_cross_input_repeated_pack_conflict_with_empty_rank() -> (
    None
):
    t = packs("T")[0]
    sig = Signature(inputs=(ax[t, 3], ax[t]), outputs=(ax[t],))

    with _expect_validation_error(ErrorCode.INCONSISTENT_DIMS):
        solve_dimensions(sig, input_shapes=((1, 3), ()))


def test_solver_matches_bruteforce_classification_for_h_times_w_small_targets() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h * w)],), outputs=(ax[h, w],))

    for target in range(0, 13):
        expected = _classify_h_w_multiplication_target(target)
        observed, _result = _solve_or_classify(sig, input_shapes=((target,),))
        assert observed == expected


def test_solver_matches_bruteforce_classification_for_h_plus_w_small_targets() -> None:
    h, w = axes("h", "w")
    sig = Signature(inputs=(ax[(h + w)],), outputs=(ax[h, w],))

    for target in range(0, 13):
        expected = _classify_h_w_addition_target(target)
        observed, _result = _solve_or_classify(sig, input_shapes=((target,),))
        assert observed == expected


def test_tensorop_call_runs_solver_before_execution() -> None:
    b, h, w, c = axes("b", "h", "w", "c")
    op = view(ax[b, (h * w), c], ax[b, h, w, c])
    x = DummyTensor(shape=(2, 12, 4))

    with _expect_validation_error(ErrorCode.AMBIGUOUS_DIMS):
        _ = op(x)


def test_tensorop_call_reaches_execution_placeholder_after_unique_solve() -> None:
    b, h, w, c = axes("b", "h", "w", "c")
    op = view(ax[b, (h * w), c], ax[b, h, w, c]).with_sizes(h=3)
    x = array_api_numpy.asarray(
        [[[float(i + j + k) for k in range(4)] for j in range(12)] for i in range(2)]
    )

    out = op(x)
    assert out.shape == (2, 3, 4, 4)
