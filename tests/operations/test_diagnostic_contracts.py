import pytest

from einf import ErrorCode, ValidationError, ax, axes, contract, einop, reduce


def _assert_core_diagnostic_contract(
    *,
    error: ValidationError,
    expected_code: ErrorCode,
) -> None:
    """Assert core diagnostic code/external-code contract."""
    assert error.code == expected_code.value
    assert error.external_code == expected_code.value.upper()


def test_diagnostic_contract_reduce_ambiguous_partial_multiplicity() -> None:
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op.reduce_by("sum")

    captured = error.value
    _assert_core_diagnostic_contract(
        error=captured,
        expected_code=ErrorCode.AMBIGUOUS_DIMS,
    )
    assert "ambiguous dims" in captured.message
    assert "reduce multiplicity mapping" in " ".join(captured.related)
    assert set(captured.data.keys()) == {"terms"}
    assert captured.data["terms"] == "b"


def test_diagnostic_contract_contract_non_atomic_axis_reports_dsl_term() -> None:
    b, h, w, c = axes("b", "h", "w", "c")

    with pytest.raises(ValidationError) as error:
        _ = contract((ax[b, (h * w)], ax[(h * w), c]), ax[b, c])

    captured = error.value
    _assert_core_diagnostic_contract(
        error=captured,
        expected_code=ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
    )
    assert "contract non-atomic axis" in captured.message
    assert "contract axis expression" in " ".join(captured.related)
    assert captured.data["term"] == "(h * w)"
    assert captured.data["side"] == "lhs"
    assert captured.data["input_index"] == 0
    assert captured.data["axis_index"] == 1


def test_diagnostic_contract_einop_reduce_by_requires_unary_signature() -> None:
    i, k, j = axes("i", "k", "j")
    op = einop((ax[i, k], ax[k, j]), ax[i, j])

    with pytest.raises(ValidationError) as error:
        _ = op.reduce_by("sum")

    captured = error.value
    _assert_core_diagnostic_contract(
        error=captured,
        expected_code=ErrorCode.OP_ARITY_MISMATCH,
    )
    assert "reduce_by is only defined for unary reduce signatures" in captured.message
    assert set(captured.data.keys()) == {"lhs_arity", "rhs_arity"}
    assert captured.data["lhs_arity"] == 2
    assert captured.data["rhs_arity"] == 1
