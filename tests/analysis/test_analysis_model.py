import pytest

from einf.analysis.model import AxisToken, TextPosition, TextSpan

_SPAN = TextSpan(
    start=TextPosition(line=1, column=0),
    end=TextPosition(line=1, column=1),
)


def test_axis_token_rejects_operation_role_for_shared_symbol() -> None:
    with pytest.raises(ValueError, match="side-only relation"):
        AxisToken(
            name="b",
            kind="axis",
            side="lhs",
            relation="shared",
            role="contracted",
            span=_SPAN,
            group=0,
        )


def test_axis_token_rejects_operation_role_on_wrong_side() -> None:
    with pytest.raises(ValueError, match="introduced axes must occur on rhs"):
        AxisToken(
            name="b",
            kind="axis",
            side="lhs",
            relation="side_only",
            role="introduced",
            span=_SPAN,
            group=0,
        )


def test_axis_token_rejects_negative_group() -> None:
    with pytest.raises(ValueError, match="group must be non-negative"):
        AxisToken(
            name="b",
            kind="axis",
            side="lhs",
            relation="shared",
            role=None,
            span=_SPAN,
            group=-1,
        )


def test_axis_token_rejects_boolean_group() -> None:
    with pytest.raises(TypeError, match="group must be an integer"):
        AxisToken(
            name="b",
            kind="axis",
            side="lhs",
            relation="shared",
            role=None,
            span=_SPAN,
            group=True,
        )
