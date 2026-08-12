from einf import axes, packs
from einf.axis import AxisTermBase, ScalarAxisTermBase


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
