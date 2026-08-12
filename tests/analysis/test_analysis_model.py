from typing import cast

import pytest

from einf.analysis.model import AxisToken, TextPosition, TextSpan

_SPAN = TextSpan(
    start=TextPosition(line=1, column=0),
    end=TextPosition(line=1, column=1),
)


@pytest.mark.parametrize(
    ("line", "column", "field"),
    (
        (True, 0, "line"),
        (cast(int, 1.5), 0, "line"),
        (cast(int, "1"), 0, "line"),
        (1, False, "column"),
        (1, cast(int, 0.5), "column"),
        (1, cast(int, "0"), "column"),
    ),
)
def test_text_position_rejects_non_integer_coordinates(
    line: int,
    column: int,
    field: str,
) -> None:
    with pytest.raises(TypeError, match=f"text position {field} must be an integer"):
        TextPosition(line=line, column=column)


def test_text_position_accepts_indexing_boundaries() -> None:
    assert TextPosition(line=1, column=0) == TextPosition(line=1, column=0)


def test_text_position_preserves_line_range_validation() -> None:
    with pytest.raises(ValueError, match="line must be >= 1"):
        TextPosition(line=0, column=0)


def test_text_position_preserves_column_range_validation() -> None:
    with pytest.raises(ValueError, match="column must be >= 0"):
        TextPosition(line=1, column=-1)


def test_text_span_accepts_zero_width_half_open_range() -> None:
    position = TextPosition(line=2, column=3)

    assert TextSpan(start=position, end=position) == TextSpan(
        start=position,
        end=position,
    )


def test_text_span_rejects_start_after_end() -> None:
    with pytest.raises(ValueError, match="start must not be after end"):
        TextSpan(
            start=TextPosition(line=2, column=0),
            end=TextPosition(line=1, column=0),
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
