from functools import reduce
from operator import or_

from einf.analysis.model import AxisToken

TOKEN_TYPES = (
    "parameter",
    "variable",
    "property",
    "function",
    "method",
    "type",
    "class",
    "enumMember",
)
TOKEN_MODIFIERS = ("introduced", "reduced", "contracted", "pack")
ROLE_TO_MODIFIER_INDEX = {
    role_name: modifier_index
    for modifier_index, role_name in enumerate(TOKEN_MODIFIERS)
}


def encode_semantic_tokens(axis_tokens: tuple[AxisToken, ...]) -> list[int]:
    """Encode axis tokens into LSP semantic token integer data."""
    data: list[int] = []
    previous_line = 0
    previous_column = 0

    for axis_token in sorted(
        axis_tokens,
        key=lambda token: (
            token.span.start.line,
            token.span.start.column,
            token.span.end.line,
            token.span.end.column,
            token.name,
            token.group,
        ),
    ):
        start_line = axis_token.span.start.line - 1
        start_column = axis_token.span.start.column
        length = max(1, axis_token.span.end.column - axis_token.span.start.column)
        token_type_index = axis_token.group % len(TOKEN_TYPES)
        modifier_mask = reduce(
            or_,
            (
                1 << ROLE_TO_MODIFIER_INDEX[role_name]
                for role_name in axis_token.roles
                if role_name in ROLE_TO_MODIFIER_INDEX
            ),
            0,
        )

        if start_line == previous_line:
            delta_line = 0
            delta_column = start_column - previous_column
        else:
            delta_line = start_line - previous_line
            delta_column = start_column

        data.extend([delta_line, delta_column, length, token_type_index, modifier_mask])
        previous_line = start_line
        previous_column = start_column

    return data


__all__ = ["TOKEN_MODIFIERS", "TOKEN_TYPES", "encode_semantic_tokens"]
