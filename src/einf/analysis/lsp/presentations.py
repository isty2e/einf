from dataclasses import dataclass

from einf.analysis.model import AxisToken, TextPosition, TextSpan

_ROLE_INLAY_LABELS = {
    "contracted": "contract",
    "reduced": "reduce",
    "introduced": "introduce",
}


@dataclass(frozen=True, slots=True)
class AxisTokenContext:
    """One axis token plus its group peers for richer editor presentation."""

    token: AxisToken
    peers: tuple[AxisToken, ...]

    @property
    def presentation_labels(self) -> tuple[str, ...]:
        """Return labels that should surface in richer editor metadata."""
        labels: list[str] = []
        if self.token.role is not None:
            labels.append(_ROLE_INLAY_LABELS[self.token.role])
        if self.token.kind == "pack":
            labels.append("pack")
        return tuple(labels)

    @property
    def peer_count(self) -> int:
        """Return the number of occurrences in the same axis group."""
        return len(self.peers)


def find_axis_token_context(
    *,
    axis_tokens: tuple[AxisToken, ...],
    position: TextPosition,
) -> AxisTokenContext | None:
    """Return the token context containing one source position."""
    grouped_tokens = _group_axis_tokens(axis_tokens)
    for axis_token in axis_tokens:
        if not _span_contains_position(axis_token.span, position):
            continue
        return AxisTokenContext(
            token=axis_token,
            peers=grouped_tokens.get(axis_token.group, (axis_token,)),
        )
    return None


def iter_inlay_contexts(
    *,
    axis_tokens: tuple[AxisToken, ...],
    visible_range: TextSpan | None,
) -> tuple[AxisTokenContext, ...]:
    """Return ordered token contexts eligible for inlay-hint rendering."""
    grouped_tokens = _group_axis_tokens(axis_tokens)
    contexts: list[AxisTokenContext] = []
    for axis_token in axis_tokens:
        if visible_range is not None and not _spans_intersect(
            axis_token.span, visible_range
        ):
            continue
        context = AxisTokenContext(
            token=axis_token,
            peers=grouped_tokens.get(axis_token.group, (axis_token,)),
        )
        if not context.presentation_labels:
            continue
        contexts.append(context)
    return tuple(contexts)


def build_hover_markdown(context: AxisTokenContext) -> str:
    """Build one markdown hover payload for an axis token context."""
    peer_lines = "\n".join(
        f"- `{peer.name}` at {_format_span(peer.span)} "
        f"(side: {peer.side}, relation: {peer.relation}, "
        f"operation role: {peer.role or 'none'})"
        for peer in context.peers
    )
    title = "Axis pack" if context.token.kind == "pack" else "Axis"
    return (
        f"**{title}** `{context.token.name}`\n\n"
        f"- kind: {context.token.kind}\n"
        f"- side: {context.token.side}\n"
        f"- relation: {context.token.relation}\n"
        f"- operation role: {context.token.role or 'none'}\n"
        f"- group: {context.token.group}\n"
        f"- occurrences: {context.peer_count}\n\n"
        f"**Group peers**\n{peer_lines}"
    )


def build_inlay_label(context: AxisTokenContext) -> str | None:
    """Build one inlay-hint label for a token context."""
    if not context.presentation_labels:
        return None
    return ", ".join(context.presentation_labels)


def _group_axis_tokens(
    axis_tokens: tuple[AxisToken, ...],
) -> dict[int, tuple[AxisToken, ...]]:
    grouped_tokens: dict[int, list[AxisToken]] = {}
    for axis_token in axis_tokens:
        grouped_tokens.setdefault(axis_token.group, []).append(axis_token)
    return {
        group: tuple(
            sorted(
                group_tokens,
                key=lambda axis_token: (
                    axis_token.span.start.line,
                    axis_token.span.start.column,
                    axis_token.span.end.line,
                    axis_token.span.end.column,
                    axis_token.name,
                ),
            )
        )
        for group, group_tokens in grouped_tokens.items()
    }


def _span_contains_position(span: TextSpan, position: TextPosition) -> bool:
    starts_before = (span.start.line, span.start.column) <= (
        position.line,
        position.column,
    )
    ends_after = (position.line, position.column) < (span.end.line, span.end.column)
    return starts_before and ends_after


def _spans_intersect(lhs: TextSpan, rhs: TextSpan) -> bool:
    return (lhs.start.line, lhs.start.column) < (rhs.end.line, rhs.end.column) and (
        rhs.start.line,
        rhs.start.column,
    ) < (lhs.end.line, lhs.end.column)


def _format_span(span: TextSpan) -> str:
    return f"L{span.start.line}:C{span.start.column}"


__all__ = [
    "AxisTokenContext",
    "build_hover_markdown",
    "build_inlay_label",
    "find_axis_token_context",
    "iter_inlay_contexts",
]
