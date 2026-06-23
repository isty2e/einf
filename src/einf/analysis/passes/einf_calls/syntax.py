import ast

from einf.analysis.model import AnalysisDiagnostic
from einf.axis import Axis, AxisExpr, AxisPack, AxisSide, AxisTerms, ScalarAxisTermBase

from .diagnostics import (
    _ANALYSIS_AXIS_TERM_ERROR,
    _ANALYSIS_SIDE_SPEC_ERROR,
    _diagnostic,
)
from .model import _AxisOccurrence, _CallSide, _SideParseResult, _SnippetContext

def _is_ax_subscript(expr: ast.expr) -> bool:
    """Return whether one expression is `ax[...]`."""
    if not isinstance(expr, ast.Subscript):
        return False
    value = expr.value
    if isinstance(value, ast.Name):
        return value.id == "ax"
    if isinstance(value, ast.Attribute):
        return value.attr == "ax"
    return False

def _parse_scalar_axis_term(
    *,
    term: ast.AST,
    side: _CallSide,
    context: _SnippetContext,
    occurrences: list[_AxisOccurrence],
    diagnostics: list[AnalysisDiagnostic],
) -> ScalarAxisTermBase | None:
    """Parse one scalar axis term expression."""
    if isinstance(term, ast.Name):
        term_span = context.span_from_ast_node(term)
        if term_span is not None:
            occurrences.append(_AxisOccurrence(name=term.id, side=side, span=term_span))
        return Axis(term.id)

    if isinstance(term, ast.Constant):
        value = term.value
        if type(value) is int:
            try:
                return ScalarAxisTermBase.coerce(value)
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    _diagnostic(
                        code=_ANALYSIS_AXIS_TERM_ERROR,
                        message=f"invalid integer axis term: {error}",
                        span=context.span_from_ast_node(term),
                    )
                )
                return None

        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_AXIS_TERM_ERROR,
                message="axis term constants must be non-negative integers",
                span=context.span_from_ast_node(term),
            )
        )
        return None

    if isinstance(term, ast.BinOp):
        if isinstance(term.op, ast.Add):
            operator = "+"
        elif isinstance(term.op, ast.Mult):
            operator = "*"
        else:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_TERM_ERROR,
                    message="only '+' and '*' are supported in axis expressions",
                    span=context.span_from_ast_node(term),
                )
            )
            return None

        left = _parse_scalar_axis_term(
            term=term.left,
            side=side,
            context=context,
            occurrences=occurrences,
            diagnostics=diagnostics,
        )
        right = _parse_scalar_axis_term(
            term=term.right,
            side=side,
            context=context,
            occurrences=occurrences,
            diagnostics=diagnostics,
        )
        if left is None or right is None:
            return None

        if operator == "+":
            return AxisExpr("+", left, right)
        return AxisExpr("*", left, right)

    if isinstance(term, ast.UnaryOp):
        if not isinstance(term.op, ast.UAdd):
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_TERM_ERROR,
                    message="only unary '+' is supported for axis expressions",
                    span=context.span_from_ast_node(term),
                )
            )
            return None

        return _parse_scalar_axis_term(
            term=term.operand,
            side=side,
            context=context,
            occurrences=occurrences,
            diagnostics=diagnostics,
        )

    diagnostics.append(
        _diagnostic(
            code=_ANALYSIS_AXIS_TERM_ERROR,
            message="unsupported axis term syntax",
            span=context.span_from_ast_node(term),
        )
    )
    return None


def _parse_axis_term(
    *,
    term: ast.AST,
    side: _CallSide,
    context: _SnippetContext,
    occurrences: list[_AxisOccurrence],
    diagnostics: list[AnalysisDiagnostic],
) -> AxisPack | ScalarAxisTermBase | None:
    """Parse one scalar/pack axis term expression."""
    if isinstance(term, ast.Starred):
        starred_value = term.value
        if not isinstance(starred_value, ast.Name):
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_TERM_ERROR,
                    message="axis pack must be a named symbol: *T",
                    span=context.span_from_ast_node(term),
                )
            )
            return None
        return AxisPack(starred_value.id)

    return _parse_scalar_axis_term(
        term=term,
        side=side,
        context=context,
        occurrences=occurrences,
        diagnostics=diagnostics,
    )


def _parse_axis_terms_expression(
    *,
    expr: ast.expr,
    side: _CallSide,
    context: _SnippetContext,
    occurrences: list[_AxisOccurrence],
    diagnostics: list[AnalysisDiagnostic],
) -> AxisTerms | None:
    """Parse one `ax[...]` expression to canonical AxisTerms."""
    if not _is_ax_subscript(expr):
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_SIDE_SPEC_ERROR,
                message="side entries must be ax[...] expressions",
                span=context.span_from_ast_node(expr),
            )
        )
        return None

    subscript_expr = expr
    assert isinstance(subscript_expr, ast.Subscript)
    slice_expr = subscript_expr.slice
    if isinstance(slice_expr, ast.Tuple):
        term_exprs = tuple(slice_expr.elts)
    else:
        term_exprs = (slice_expr,)

    terms = []
    for term_expr in term_exprs:
        parsed_term = _parse_axis_term(
            term=term_expr,
            side=side,
            context=context,
            occurrences=occurrences,
            diagnostics=diagnostics,
        )
        if parsed_term is None:
            return None
        terms.append(parsed_term)

    try:
        return AxisTerms.from_spec(tuple(terms))
    except (TypeError, ValueError) as error:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_AXIS_TERM_ERROR,
                message=str(error),
                span=context.span_from_ast_node(expr),
            )
        )
        return None


def _parse_side_spec(
    *,
    expr: ast.expr,
    side: _CallSide,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> _SideParseResult | None:
    """Parse one call side spec (`ax[...]` or tuple of `ax[...]`)."""
    axis_entries: tuple[ast.expr, ...]
    if isinstance(expr, ast.Tuple) and not _is_ax_subscript(expr):
        axis_entries = tuple(expr.elts)
    else:
        axis_entries = (expr,)

    if not axis_entries:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_SIDE_SPEC_ERROR,
                message="side specification must contain at least one ax[...] entry",
                span=context.span_from_ast_node(expr),
            )
        )
        return None

    occurrences: list[_AxisOccurrence] = []
    side_terms: list[AxisTerms] = []
    for axis_entry in axis_entries:
        parsed_terms = _parse_axis_terms_expression(
            expr=axis_entry,
            side=side,
            context=context,
            occurrences=occurrences,
            diagnostics=diagnostics,
        )
        if parsed_terms is None:
            return None
        side_terms.append(parsed_terms)

    axis_side = AxisSide.coerce(tuple(side_terms))
    return _SideParseResult(
        axis_side=axis_side,
        axis_names=frozenset(occurrence.name for occurrence in occurrences),
        occurrences=tuple(occurrences),
    )
