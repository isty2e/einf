import ast
from bisect import bisect_left
from dataclasses import dataclass, field

from einf.analysis.model import TextPosition, TextSpan


@dataclass(frozen=True, slots=True)
class SourceText:
    """Source text with column conversion and span slicing helpers."""

    source: str
    _lines: tuple[str, ...] = field(init=False, repr=False)
    _line_utf8_columns: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lines = tuple(self.source.splitlines(keepends=True))
        line_utf8_columns = tuple(
            self._build_line_utf8_columns(line.rstrip("\r\n")) for line in lines
        )
        object.__setattr__(self, "_lines", lines)
        object.__setattr__(self, "_line_utf8_columns", line_utf8_columns)

    @staticmethod
    def _build_line_utf8_columns(line: str) -> tuple[int, ...]:
        """Return cumulative UTF-8 byte offsets for one source line."""
        columns = [0]
        total = 0
        for character in line:
            total += len(character.encode("utf-8"))
            columns.append(total)
        return tuple(columns)

    def line_text(self, line: int) -> str:
        """Return one source line without line terminator."""
        if line < 1 or line > len(self._lines):
            raise ValueError("line is out of source bounds")
        return self._lines[line - 1].rstrip("\r\n")

    def character_column(self, *, line: int, utf8_byte_column: int) -> int:
        """Convert one UTF-8 byte column into a character column."""
        if utf8_byte_column < 0:
            raise ValueError("utf8 byte column must be >= 0")

        if line < 1 or line > len(self._line_utf8_columns):
            raise ValueError("line is out of source bounds")

        line_columns = self._line_utf8_columns[line - 1]
        column_index = bisect_left(line_columns, utf8_byte_column)
        if (
            column_index >= len(line_columns)
            or line_columns[column_index] != utf8_byte_column
        ):
            raise ValueError("utf8 byte column does not align to a character boundary")
        return column_index

    def span_from_ast_node(self, node: ast.AST) -> TextSpan | None:
        """Convert one stdlib AST node span into canonical character columns."""
        start_line = getattr(node, "lineno", None)
        start_column = getattr(node, "col_offset", None)
        end_line = getattr(node, "end_lineno", None)
        end_column = getattr(node, "end_col_offset", None)
        if (
            type(start_line) is not int
            or type(start_column) is not int
            or type(end_line) is not int
            or type(end_column) is not int
        ):
            return None

        return TextSpan(
            start=TextPosition(
                line=start_line,
                column=self.character_column(
                    line=start_line,
                    utf8_byte_column=start_column,
                ),
            ),
            end=TextPosition(
                line=end_line,
                column=self.character_column(
                    line=end_line,
                    utf8_byte_column=end_column,
                ),
            ),
        )

    def slice(self, span: TextSpan) -> str:
        """Extract source text for one canonical character span."""
        start_line_index = span.start.line - 1
        end_line_index = span.end.line - 1
        if (
            start_line_index < 0
            or end_line_index < 0
            or start_line_index >= len(self._lines)
            or end_line_index >= len(self._lines)
        ):
            raise ValueError("span is out of source bounds")

        if start_line_index == end_line_index:
            return self._lines[start_line_index][span.start.column : span.end.column]

        parts: list[str] = [self._lines[start_line_index][span.start.column :]]
        for line_index in range(start_line_index + 1, end_line_index):
            parts.append(self._lines[line_index])
        parts.append(self._lines[end_line_index][: span.end.column])
        return "".join(parts)
