import ast
from dataclasses import dataclass

from einf.analysis.model import (
    AnalysisDiagnostic,
    AxisOccurrenceSide,
    AxisStructuralKind,
    TextPosition,
    TextSpan,
)
from einf.analysis.passes.call_resolution import CallBindings
from einf.analysis.source import SourceText
from einf.axis import AxisSide, AxisTerms
from einf.operations.tensor_op import TensorOp
from einf.reduction.schema import Reducer

_ReducePhaseArg = tuple[AxisTerms, Reducer]
_ReduceByFirstArg = Reducer | _ReducePhaseArg


@dataclass(frozen=True, slots=True)
class _AxisOccurrence:
    """One axis-token occurrence from one side expression."""

    name: str
    kind: AxisStructuralKind
    side: AxisOccurrenceSide
    span: TextSpan


@dataclass(frozen=True, slots=True)
class _SideParseResult:
    """Parsed side summary with canonical axis side and occurrences."""

    axis_side: AxisSide
    axis_names: frozenset[str]
    pack_names: frozenset[str]
    occurrences: tuple[_AxisOccurrence, ...]

    def symbol_names(self, kind: AxisStructuralKind) -> frozenset[str]:
        """Return canonical symbol names for one structural kind."""
        if kind == "axis":
            return self.axis_names
        return self.pack_names


@dataclass(frozen=True, slots=True)
class _CallParseResult:
    """Parsed `einf` op-call summary."""

    op_name: str
    span: TextSpan
    lhs: _SideParseResult
    rhs: _SideParseResult


@dataclass(frozen=True, slots=True)
class _EvaluatedCall:
    """Evaluation result for one parsed call expression."""

    base_call: _CallParseResult | None
    op: TensorOp | None
    diagnostics: tuple[AnalysisDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _BaseCallArguments:
    """Bound base-op constructor arguments."""

    lhs_expr: ast.expr
    rhs_expr: ast.expr


@dataclass(frozen=True, slots=True)
class _SnippetContext:
    """Source text plus one snippet base span for local AST node mapping."""

    module_source: SourceText
    base_span: TextSpan
    bindings: CallBindings

    def _absolute_column(self, *, local_line: int, local_utf8_byte_column: int) -> int:
        """Map one snippet-local UTF-8 byte column into absolute char column."""
        if local_line < 1:
            raise ValueError("local line must be >= 1")
        if local_utf8_byte_column < 0:
            raise ValueError("local utf8 byte column must be >= 0")

        absolute_line = self.base_span.start.line + local_line - 1
        if local_line == 1:
            prefix = self.module_source.line_text(absolute_line)[
                : self.base_span.start.column
            ]
            prefix_utf8_bytes = len(prefix.encode("utf-8"))
            absolute_utf8_byte_column = prefix_utf8_bytes + local_utf8_byte_column
        else:
            absolute_utf8_byte_column = local_utf8_byte_column

        return self.module_source.character_column(
            line=absolute_line,
            utf8_byte_column=absolute_utf8_byte_column,
        )

    def span_from_ast_node(self, node: ast.AST) -> TextSpan | None:
        """Convert one snippet-local AST span into an absolute module span."""
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

        absolute_start_line = self.base_span.start.line + start_line - 1
        absolute_end_line = self.base_span.start.line + end_line - 1
        return TextSpan(
            start=TextPosition(
                line=absolute_start_line,
                column=self._absolute_column(
                    local_line=start_line,
                    local_utf8_byte_column=start_column,
                ),
            ),
            end=TextPosition(
                line=absolute_end_line,
                column=self._absolute_column(
                    local_line=end_line,
                    local_utf8_byte_column=end_column,
                ),
            ),
        )
