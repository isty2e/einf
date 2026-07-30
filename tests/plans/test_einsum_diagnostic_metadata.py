import pytest

from einf.axis import Axis, AxisSide, AxisTerms, ScalarAxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.steps.einsum.equation import build_contract_equation
from einf.steps.einsum.step import _build_equation_from_scalar_terms


def _atomic_contract_diagnostic_cases() -> tuple[
    tuple[AxisSide, AxisTerms, ErrorCode], ...
]:
    input_axis = Axis("input_axis")
    output_axis = Axis("output_axis")
    expression = input_axis * output_axis
    return (
        (
            AxisSide((AxisTerms((expression,)),)),
            AxisTerms((input_axis,)),
            ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
        ),
        (
            AxisSide((AxisTerms((input_axis,)),)),
            AxisTerms((expression,)),
            ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
        ),
        (
            AxisSide((AxisTerms((input_axis,)),)),
            AxisTerms((input_axis, input_axis)),
            ErrorCode.INCONSISTENT_DIMS,
        ),
        (
            AxisSide((AxisTerms((input_axis,)),)),
            AxisTerms((output_axis,)),
            ErrorCode.INCONSISTENT_DIMS,
        ),
    )


def _scalar_contract_diagnostic_cases() -> tuple[
    tuple[tuple[ScalarAxisTerms, ...], ScalarAxisTerms], ...
]:
    input_axis = Axis("scalar_input_axis")
    output_axis = Axis("scalar_output_axis")
    return (
        (
            (ScalarAxisTerms((input_axis,)),),
            ScalarAxisTerms((input_axis, input_axis)),
        ),
        (
            (ScalarAxisTerms((input_axis,)),),
            ScalarAxisTerms((output_axis,)),
        ),
    )


@pytest.mark.parametrize(
    ("input_axis_lists", "output_axis_list", "expected_code"),
    _atomic_contract_diagnostic_cases(),
    ids=("non-atomic-input", "non-atomic-output", "duplicate-output", "missing-output"),
)
def test_atomic_contract_equation_diagnostics_report_contract_operation(
    input_axis_lists: AxisSide,
    output_axis_list: AxisTerms,
    expected_code: ErrorCode,
) -> None:
    with pytest.raises(ValidationError) as error:
        build_contract_equation(
            input_axis_lists=input_axis_lists,
            output_axis_list=output_axis_list,
        )

    assert error.value.code == expected_code.value
    assert error.value.data == {"operation": "contract"}


@pytest.mark.parametrize(
    ("input_axis_terms", "output_axis_terms"),
    _scalar_contract_diagnostic_cases(),
    ids=("duplicate-output", "missing-output"),
)
def test_scalar_contract_equation_diagnostics_report_contract_operation(
    input_axis_terms: tuple[ScalarAxisTerms, ...],
    output_axis_terms: ScalarAxisTerms,
) -> None:
    with pytest.raises(ValidationError) as error:
        _build_equation_from_scalar_terms(
            input_axis_terms=input_axis_terms,
            output_axis_terms=output_axis_terms,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data == {"operation": "contract"}
