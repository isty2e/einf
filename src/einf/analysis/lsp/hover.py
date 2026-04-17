from lsprotocol import types as lsp

from einf.analysis.model import AxisToken, TextPosition

from .presentations import build_hover_markdown, find_axis_token_context


def build_hover(
    *,
    axis_tokens: tuple[AxisToken, ...],
    position: TextPosition,
) -> lsp.Hover | None:
    """Build hover content for one axis token position."""
    context = find_axis_token_context(axis_tokens=axis_tokens, position=position)
    if context is None:
        return None
    return lsp.Hover(
        contents=lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown,
            value=build_hover_markdown(context),
        ),
    )


__all__ = ["build_hover"]
