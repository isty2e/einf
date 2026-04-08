from lsprotocol import types as lsp

from einf.analysis.model import AxisToken, TextPosition, TextSpan

from .presentations import build_inlay_label, iter_inlay_contexts


def build_inlay_hints(
    *,
    axis_tokens: tuple[AxisToken, ...],
    visible_range: TextSpan | None,
) -> list[lsp.InlayHint]:
    """Build inlay hints for selected axis roles in the visible range."""
    hints: list[lsp.InlayHint] = []
    for context in iter_inlay_contexts(
        axis_tokens=axis_tokens,
        visible_range=visible_range,
    ):
        label = build_inlay_label(context)
        if label is None:
            continue
        hints.append(
            lsp.InlayHint(
                position=_position_from_text_position(context.token.span.end),
                label=label,
                kind=lsp.InlayHintKind.Type,
                padding_left=True,
            )
        )
    return hints


def _position_from_text_position(position: TextPosition) -> lsp.Position:
    return lsp.Position(line=position.line - 1, character=position.column)


__all__ = ["build_inlay_hints"]
