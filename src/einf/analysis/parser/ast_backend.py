import ast
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.source import SourceText

from .base import ParsedModule, ParsedNode, ParserSyntaxError, TextEdit


def _value_from_ast_node(node: ast.AST) -> str | None:
    """Extract stable scalar value for value-bearing ast nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        constant_value = node.value
        if isinstance(constant_value, str):
            return constant_value
        if isinstance(constant_value, bool):
            return str(constant_value)
        if isinstance(constant_value, int):
            return str(constant_value)
        if isinstance(constant_value, float):
            return repr(constant_value)
    return None


def _syntax_error_span(error: SyntaxError) -> TextSpan | None:
    line = error.lineno
    column = error.offset
    if line is None or column is None or line < 1 or column < 1:
        return None

    start = TextPosition(line=line, column=column - 1)
    end_line = error.end_lineno if error.end_lineno is not None else line
    end_column = error.end_offset if error.end_offset is not None else column + 1
    if end_line < 1 or end_column < 1:
        return TextSpan(
            start=start,
            end=TextPosition(line=start.line, column=start.column + 1),
        )

    end = TextPosition(
        line=end_line,
        column=max(start.column + 1, end_column - 1),
    )
    return TextSpan(start=start, end=end)


@dataclass(frozen=True, slots=True)
class AstParserBackend:
    """Parser backend based on Python stdlib ast."""

    name: str = "ast"

    def validate_available(self) -> None:
        """Confirm that the standard-library parser is available."""

    def parse(self, source: str, path: Path) -> ParsedModule:
        """Parse source text and normalize into ParsedModule."""
        try:
            module_node = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            raise ParserSyntaxError(
                message=str(error),
                span=_syntax_error_span(error),
            ) from error

        source_text = SourceText(source)
        nodes: list[ParsedNode] = []

        def visit(ast_node: ast.AST) -> int:
            node_id = len(nodes)
            nodes.append(
                ParsedNode(
                    node_id=node_id,
                    kind="pending",
                    span=None,
                    value=None,
                    child_ids=(),
                )
            )
            child_ids = tuple(visit(child) for child in ast.iter_child_nodes(ast_node))
            nodes[node_id] = ParsedNode(
                node_id=node_id,
                kind=type(ast_node).__name__,
                span=source_text.span_from_ast_node(ast_node),
                value=_value_from_ast_node(ast_node),
                child_ids=child_ids,
            )
            return node_id

        root_id = visit(module_node)
        return ParsedModule(
            path=path,
            source=source,
            nodes=tuple(nodes),
            root_id=root_id,
        )

    def reparse(
        self,
        previous: ParsedModule,
        edits: tuple[TextEdit, ...],
        new_source: str,
    ) -> ParsedModule:
        """Reparse by full-parse fallback for incremental edits."""
        _ = edits
        return self.parse(source=new_source, path=previous.path)
