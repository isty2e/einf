import pytest

from einf import axes, packs
from einf.axis import AxisTermBase, ScalarAxisTermBase


class _StructuralScalarTerm(ScalarAxisTermBase):
    def to_dsl(self) -> str:
        return "structural"

    def stable_token(self) -> str:
        return "structural"

    def axis_names(self) -> set[str]:
        return set()

    def pack_names(self) -> set[str]:
        return set()


def test_axis_pack_exposes_only_structural_term_contract() -> None:
    (pack,) = packs("batch")

    assert isinstance(pack, AxisTermBase)
    assert not isinstance(pack, ScalarAxisTermBase)
    assert pack.to_dsl() == "*batch"
    assert pack.axis_names() == set()
    assert pack.pack_names() == {"batch"}
    assert not hasattr(pack, "evaluate")
    assert not hasattr(pack, "max_literal")
    assert not hasattr(pack, "evaluate_bounds")


def test_scalar_axis_term_requires_scalar_algebra_implementations() -> None:
    assert _StructuralScalarTerm.__abstractmethods__ == frozenset(
        {"evaluate", "max_literal", "evaluate_bounds"}
    )

    with pytest.raises(TypeError, match="abstract"):
        type.__call__(_StructuralScalarTerm)


def test_scalar_axis_term_coercion_rejects_axis_pack() -> None:
    (pack,) = packs("batch")

    with pytest.raises(TypeError, match="axis expression terms"):
        ScalarAxisTermBase.coerce(pack)


def test_scalar_axis_terms_own_scalar_algebra() -> None:
    height, width = axes("height", "width")
    expression = (height + 2) * width

    assert isinstance(expression, ScalarAxisTermBase)
    assert expression.evaluate({"height": 3, "width": 4}) == 20
    assert expression.max_literal() == 2
    assert expression.evaluate_bounds(
        current={"height": 3},
        variable_bounds={"width": 7},
    ) == (0, 35)
