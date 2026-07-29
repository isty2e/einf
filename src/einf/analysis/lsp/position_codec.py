from dataclasses import dataclass, field

from lsprotocol import types as lsp
from pygls.workspace.position_codec import PositionCodec

from einf.analysis.model import TextPosition, TextSpan

_SUPPORTED_ENCODINGS = frozenset(
    {
        lsp.PositionEncodingKind.Utf8,
        lsp.PositionEncodingKind.Utf16,
        lsp.PositionEncodingKind.Utf32,
    }
)


@dataclass(frozen=True, slots=True)
class LspPositionCodec:
    """Convert one source snapshot between canonical and negotiated LSP positions."""

    lines: tuple[str, ...]
    encoding: lsp.PositionEncodingKind | str = lsp.PositionEncodingKind.Utf16
    _wire_codec: PositionCodec = field(init=False, repr=False)
    _wire_offsets_by_line: dict[int, tuple[int, ...] | None] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(f"unsupported LSP position encoding: {self.encoding}")
        object.__setattr__(self, "_wire_codec", PositionCodec(self.encoding))
        object.__setattr__(self, "_wire_offsets_by_line", {})

    def from_lsp_position(self, position: lsp.Position) -> TextPosition:
        """Convert a negotiated wire position to a canonical source position."""
        canonical = self._wire_codec.position_from_client_units(
            self.lines,
            lsp.Position(line=position.line, character=position.character),
        )
        return TextPosition(line=canonical.line + 1, column=canonical.character)

    def from_lsp_range(self, lsp_range: lsp.Range) -> TextSpan:
        """Convert a negotiated wire range to a canonical source span."""
        return TextSpan(
            start=self.from_lsp_position(lsp_range.start),
            end=self.from_lsp_position(lsp_range.end),
        )

    def to_lsp_position(self, position: TextPosition) -> lsp.Position:
        """Convert a canonical source position to negotiated wire units."""
        line_index = position.line - 1
        if line_index >= len(self.lines):
            return lsp.Position(line=len(self.lines), character=0)

        column = min(position.column, len(self.lines[line_index]))
        offsets = self._wire_offsets_for_line(line_index)
        if offsets is None:
            return lsp.Position(line=line_index, character=column)

        return lsp.Position(
            line=line_index,
            character=offsets[column],
        )

    def to_lsp_range(self, span: TextSpan) -> lsp.Range:
        """Convert a canonical source span to a negotiated wire range."""
        return lsp.Range(
            start=self.to_lsp_position(span.start),
            end=self.to_lsp_position(span.end),
        )

    def _wire_offsets_for_line(self, line_index: int) -> tuple[int, ...] | None:
        if line_index in self._wire_offsets_by_line:
            return self._wire_offsets_by_line[line_index]

        line = self.lines[line_index]
        if self.encoding == lsp.PositionEncodingKind.Utf32 or line.isascii():
            self._wire_offsets_by_line[line_index] = None
            return None
        offsets = [0]
        for character in line:
            offsets.append(offsets[-1] + self._wire_codec.client_num_units(character))
        result = tuple(offsets)
        self._wire_offsets_by_line[line_index] = result
        return result


__all__ = ["LspPositionCodec"]
