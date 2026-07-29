import pytest

from einf.axis import Axis, AxisSide, AxisTerms, ScalarAxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.lowering.einop.equation import build_einop_equations
from einf.steps.einsum.equation import build_contract_equation
from einf.steps.einsum.step import _build_equation_from_scalar_terms

_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EINSUM_SYMBOL_LIMIT = len(_EINSUM_SYMBOLS)
_HIGH_ARITY_INPUT_COUNT = _EINSUM_SYMBOL_LIMIT + 1


def _expected_repeated_axis_equation() -> str:
    inputs = ",".join(("a",) * _HIGH_ARITY_INPUT_COUNT)
    return f"{inputs}->a"


def _distinct_axes(count: int) -> tuple[Axis, ...]:
    return tuple(Axis(f"axis_{index}") for index in range(count))


def _expected_full_budget_equation() -> str:
    return f"{','.join(_EINSUM_SYMBOLS)}->a"


def test_contract_equation_budget_counts_distinct_labels() -> None:
    shared_axis = Axis("shared")
    repeated_inputs = AxisSide(
        tuple(AxisTerms((shared_axis,)) for _ in range(_HIGH_ARITY_INPUT_COUNT))
    )

    equation = build_contract_equation(
        input_axis_lists=repeated_inputs,
        output_axis_list=AxisTerms((shared_axis,)),
    )

    assert equation == _expected_repeated_axis_equation()

    full_budget_axes = _distinct_axes(_EINSUM_SYMBOL_LIMIT)
    full_budget_equation = build_contract_equation(
        input_axis_lists=AxisSide(
            tuple(AxisTerms((axis,)) for axis in full_budget_axes)
        ),
        output_axis_list=AxisTerms((full_budget_axes[0],)),
    )
    assert full_budget_equation == _expected_full_budget_equation()

    distinct_axes = _distinct_axes(_HIGH_ARITY_INPUT_COUNT)
    with pytest.raises(ValidationError) as error:
        build_contract_equation(
            input_axis_lists=AxisSide(
                tuple(AxisTerms((axis,)) for axis in distinct_axes)
            ),
            output_axis_list=AxisTerms((distinct_axes[0],)),
        )
    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_scalar_equation_budget_counts_distinct_labels() -> None:
    shared_axis = Axis("shared")
    repeated_inputs = tuple(
        ScalarAxisTerms((shared_axis,)) for _ in range(_HIGH_ARITY_INPUT_COUNT)
    )

    equation = _build_equation_from_scalar_terms(
        input_axis_terms=repeated_inputs,
        output_axis_terms=ScalarAxisTerms((shared_axis,)),
    )

    assert equation == _expected_repeated_axis_equation()

    full_budget_axes = _distinct_axes(_EINSUM_SYMBOL_LIMIT)
    full_budget_equation = _build_equation_from_scalar_terms(
        input_axis_terms=tuple(ScalarAxisTerms((axis,)) for axis in full_budget_axes),
        output_axis_terms=ScalarAxisTerms((full_budget_axes[0],)),
    )
    assert full_budget_equation == _expected_full_budget_equation()

    distinct_axes = _distinct_axes(_HIGH_ARITY_INPUT_COUNT)
    with pytest.raises(ValidationError) as error:
        _build_equation_from_scalar_terms(
            input_axis_terms=tuple(ScalarAxisTerms((axis,)) for axis in distinct_axes),
            output_axis_terms=ScalarAxisTerms((distinct_axes[0],)),
        )
    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_einop_equation_budget_counts_distinct_labels() -> None:
    shared_axis = Axis("shared")
    repeated_inputs = tuple(
        AxisTerms((shared_axis,)) for _ in range(_HIGH_ARITY_INPUT_COUNT)
    )

    equations = build_einop_equations(
        input_axis_lists=repeated_inputs,
        output_axis_lists=(AxisTerms((shared_axis,)),),
    )

    assert equations == (_expected_repeated_axis_equation(),)

    full_budget_axes = _distinct_axes(_EINSUM_SYMBOL_LIMIT)
    full_budget_equations = build_einop_equations(
        input_axis_lists=tuple(AxisTerms((axis,)) for axis in full_budget_axes),
        output_axis_lists=(AxisTerms((full_budget_axes[0],)),),
    )
    assert full_budget_equations == (_expected_full_budget_equation(),)

    distinct_axes = _distinct_axes(_HIGH_ARITY_INPUT_COUNT)
    with pytest.raises(ValidationError) as error:
        build_einop_equations(
            input_axis_lists=tuple(AxisTerms((axis,)) for axis in distinct_axes),
            output_axis_lists=(AxisTerms((distinct_axes[0],)),),
        )
    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
