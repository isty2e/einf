from collections.abc import Iterator
from dataclasses import dataclass

import pytest

import einf.axis.algebra as axis_algebra_module
from einf import (
    ErrorCode,
    ExecutionError,
    Signature,
    ValidationError,
    ax,
    axes,
    einop,
    rearrange,
    view,
)
from einf.axis import AxisTerms
from einf.diagnostics import TensorOpError
from einf.operations.tensor_op import TensorOp as RuntimeTensorOp
from einf.solver import solve_dimensions


@dataclass(frozen=True, slots=True)
class DummyTensor:
    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> "DummyTensor":
        _ = key
        return self


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


def _complexity_terms(
    *, prefix: str
) -> tuple[AxisTerms, tuple[AxisTerms, AxisTerms], dict[str, int]]:
    a, b, c, d, e, f = axes(*(f"{prefix}_{name}" for name in "abcdef"))
    rest = (c + d) * (e + f)
    whole = (a + b) * rest
    parts = (ax[a * rest], ax[b * rest])
    return (
        ax[whole],
        parts,
        {axis.name: 1 for axis in (a, b, c, d, e, f)},
    )


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


def test_tensor_op_errors_are_distinct_from_primitive_value_errors() -> None:
    assert issubclass(TensorOpError, Exception)
    assert not issubclass(TensorOpError, ValueError)


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


def test_dim_solver_preserves_axis_expression_complexity_diagnostic(
    small_candidate_limit: None,
) -> None:
    _ = small_candidate_limit
    a, b, c, d, e, f = axes(
        "solver_limit_a",
        "solver_limit_b",
        "solver_limit_c",
        "solver_limit_d",
        "solver_limit_e",
        "solver_limit_f",
    )
    expression = ((a + b) * (c + d)) * (e + f)
    signature = Signature(inputs=(ax[expression],), outputs=(ax[a, b, c, d, e, f],))

    with pytest.raises(ValidationError) as error:
        solve_dimensions(signature, input_shapes=((8,),))

    assert error.value.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "distributive_product_candidates",
        "limit": 4,
        "attempted": 8,
    }


def test_view_preserves_axis_expression_complexity_diagnostic(
    small_candidate_limit: None,
) -> None:
    _ = small_candidate_limit
    whole, parts, _ = _complexity_terms(prefix="view_limit")

    with pytest.raises(ValidationError) as error:
        _ = view(whole, parts)

    assert error.value.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value
    assert error.value.data == {
        "complexity_kind": "distributive_product_candidates",
        "limit": 4,
        "attempted": 8,
    }


@pytest.mark.parametrize("concat", [False, True])
def test_rearrange_preserves_axis_expression_complexity_diagnostic(
    small_candidate_limit: None,
    *,
    concat: bool,
) -> None:
    _ = small_candidate_limit
    whole, parts, _ = _complexity_terms(prefix="rearrange_limit")

    with pytest.raises(ValidationError) as error:
        if concat:
            _ = rearrange(parts, whole)
        else:
            _ = rearrange(whole, parts)

    assert error.value.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value
    assert error.value.external_code == "AXIS_EXPRESSION_TOO_COMPLEX"
    assert error.value.data == {
        "complexity_kind": "distributive_product_candidates",
        "limit": 4,
        "attempted": 8,
    }


@pytest.mark.parametrize("concat", [False, True])
def test_einop_layout_normalization_avoids_canonical_expansion(
    small_candidate_limit: None,
    *,
    concat: bool,
) -> None:
    _ = small_candidate_limit
    whole, parts, sizes = _complexity_terms(prefix="einop_limit")
    op = einop(parts, whole) if concat else einop(whole, parts)
    plan = op.with_sizes(**sizes).plan_dict()

    assert plan["kind"] == "layout_normalized"
    assert plan["executable_now"] is True


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
    assert isinstance(op, RuntimeTensorOp)

    with pytest.raises(ValidationError) as error:
        _ = op.__call__(x, x)

    assert error.value.code == "multi_input_not_allowed"
    assert error.value.external_code == "MULTI_INPUT_NOT_ALLOWED"
    assert error.value.channel == "validation_error"
