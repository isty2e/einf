from dataclasses import dataclass

import pytest

from einf import ErrorCode, ExecutionError, Signature, ValidationError, ax, axes, view
from einf.solver import solve_dimensions


@dataclass(frozen=True, slots=True)
class DummyTensor:
    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> "DummyTensor":
        _ = key
        return self


def test_validation_error_exposes_structured_fields() -> None:
    error = ValidationError(
        code=ErrorCode.AMBIGUOUS_DIMS,
        message="ambiguous dims: multiple assignments exist",
        help="add with_sizes constraints",
        related=("dim solver",),
        data={"target": 12},
    )

    assert error.code == "ambiguous_dims"
    assert error.external_code == "AMBIGUOUS_DIMS"
    assert error.severity == "error"
    assert error.help == "add with_sizes constraints"
    assert error.related[0] == "dim solver"
    assert error.data == {"target": 12}
    assert str(error) == "ambiguous dims: multiple assignments exist"


def test_validation_error_rejects_blank_related_note() -> None:
    with pytest.raises(ValueError):
        ValidationError(
            code="AXIS_NAMES_DROPPED",
            message="axis names dropped",
            related=(" ",),
        )


def test_execution_error_exposes_channel() -> None:
    error = ExecutionError(
        code=ErrorCode.OP_ARITY_MISMATCH,
        message="execution mismatch",
    )
    assert error.channel == "execution_error"


def test_dim_solver_ambiguity_contains_help_and_related_metadata() -> None:
    b, h, w, c = axes("b", "h", "w", "c")
    sig = Signature(inputs=(ax[b, (h * w), c],), outputs=(ax[b, h, w, c],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(sig, input_shapes=((2, 12, 4),))

    assert error.value.code == "ambiguous_dims"
    assert error.value.external_code == "AMBIGUOUS_DIMS"
    assert error.value.help is not None
    assert "with_sizes constraints" in error.value.help
    assert "dim solver" in error.value.related


def test_with_sizes_negative_binding_raises_inconsistent_dims_diagnostic() -> None:
    b, h = axes("b", "h")
    op = view(ax[b], ax[b, h])

    with pytest.raises(ValidationError) as error:
        _ = op.with_sizes(h=-1)

    assert error.value.code == "inconsistent_dims"
    assert error.value.external_code == "INCONSISTENT_DIMS"
    assert error.value.help == "provide non-negative with_sizes bindings"
    assert "with_sizes binding" in error.value.related
    assert error.value.data == {"operation": "view", "dim": "h", "value": -1}


def test_validation_error_accepts_upper_snake_for_compatibility() -> None:
    error = ValidationError(
        code="RUNTIME_ABORT",
        message="runtime aborted",
    )

    assert error.code == "runtime_abort"
    assert error.external_code == "RUNTIME_ABORT"


def test_tensorop_overflow_arity_uses_validation_error() -> None:
    b = axes("b")[0]
    op = view(ax[b], ax[b])
    x = DummyTensor(shape=(3,))

    with pytest.raises(ValidationError) as error:
        _ = getattr(op, "__call__")(x, x)

    assert error.value.code == "multi_input_not_allowed"
    assert error.value.external_code == "MULTI_INPUT_NOT_ALLOWED"
    assert error.value.channel == "validation_error"
