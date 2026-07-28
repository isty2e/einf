from dataclasses import dataclass

from einf.analysis.model import AxisToken, TextPosition, TextSpan

PRESENTATION_ROLE_ORDER = ("contracted", "reduced", "introduced", "pack")
INLAY_LABELS = {
    "contracted": "contract",
    "reduced": "reduce",
    "introduced": "introduce",
    "pack": "pack",
}


@dataclass(frozen=True, slots=True)
class AxisTokenContext:
    """One axis token plus its group peers for richer editor presentation."""

    token: AxisToken
    peers: tuple[AxisToken, ...]

    @property
    def presentation_roles(self) -> tuple[str, ...]:
        """Return roles that should surface in richer editor metadata."""
        return tuple(
            role_name
            for role_name in PRESENTATION_ROLE_ORDER
            if role_name in self.token.roles
        )

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
        if not context.presentation_roles:
            continue
        contexts.append(context)
    return tuple(contexts)


def build_hover_markdown(context: AxisTokenContext) -> str:
    """Build one markdown hover payload for an axis token context."""
    peer_lines = "\n".join(
        f"- `{peer.name}` at {_format_span(peer.span)} ({', '.join(peer.roles)})"
        for peer in context.peers
    )
    roles = ", ".join(context.token.roles)
    presentation_roles = ", ".join(context.presentation_roles) or "none"
    return (
        f"**Axis** `{context.token.name}`\n\n"
        f"- roles: {roles}\n"
        f"- richer roles: {presentation_roles}\n"
        f"- group: {context.token.group}\n"
        f"- occurrences: {context.peer_count}\n\n"
        f"**Group peers**\n{peer_lines}"
    )


def build_inlay_label(context: AxisTokenContext) -> str | None:
    """Build one inlay-hint label for a token context."""
    if not context.presentation_roles:
        return None
    return ", ".join(
        INLAY_LABELS[role_name] for role_name in context.presentation_roles
    )


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
    "INLAY_LABELS",
    "PRESENTATION_ROLE_ORDER",
    "AxisTokenContext",
    "build_hover_markdown",
    "build_inlay_label",
    "find_axis_token_context",
    "iter_inlay_contexts",
]
