import ast
import inspect
from dataclasses import dataclass
from typing import Literal

from einf.analysis.model import (
    AnalysisDiagnostic,
    AxisToken,
    TextPosition,
    TextSpan,
)
from einf.analysis.parser import ParsedModule
from einf.analysis.passes.call_resolution import (
    EINF_OP_NAMES,
    CallBindings,
    build_call_bindings,
)
from einf.analysis.source import SourceText
from einf.axis import Axis, AxisExpr, AxisPack, AxisSide, AxisTerms, ScalarAxisTermBase
from einf.diagnostics import ValidationError
from einf.operations import TensorOp, contract, einop, rearrange, reduce, repeat, view
from einf.operations.validation import validate_contract_atomic_terms
from einf.reduction.schema import Reducer
from einf.signature import Signature
from einf.tensor_types import TensorLike

_VALIDATE_RHS_SUBSET_OPS = frozenset({"contract", "einop", "reduce"})

_ANALYSIS_CALL_SHAPE_ERROR = "ANALYSIS_CALL_SHAPE_ERROR"
_ANALYSIS_SIDE_SPEC_ERROR = "ANALYSIS_SIDE_SPEC_ERROR"
_ANALYSIS_AXIS_TERM_ERROR = "ANALYSIS_AXIS_TERM_ERROR"
_ANALYSIS_WITH_SIZES_ERROR = "ANALYSIS_WITH_SIZES_ERROR"
_ANALYSIS_REDUCE_BY_ERROR = "ANALYSIS_REDUCE_BY_ERROR"
_ANALYSIS_AXIS_NOT_IN_INPUT = "ANALYSIS_AXIS_NOT_IN_INPUT"

_CallSide = Literal["lhs", "rhs"]
_ReducePhaseArg = tuple[AxisTerms, Reducer]
_ReduceByFirstArg = Reducer | _ReducePhaseArg
_ENTRYPOINT_SIGNATURES = {
    "view": inspect.signature(view),
    "rearrange": inspect.signature(rearrange),
    "repeat": inspect.signature(repeat),
    "reduce": inspect.signature(reduce),
    "contract": inspect.signature(contract),
    "einop": inspect.signature(einop),
}


def _analysis_reducer_callable(*_args: TensorLike) -> int:
    """Static placeholder reducer for symbol-only callable references."""
    return 0


@dataclass(frozen=True, slots=True)
class _AxisOccurrence:
    """One axis-token occurrence from one side expression."""

    name: str
    side: _CallSide
    span: TextSpan


@dataclass(frozen=True, slots=True)
class _SideParseResult:
    """Parsed side summary with canonical axis side and occurrences."""

    axis_side: AxisSide
    axis_names: frozenset[str]
    occurrences: tuple[_AxisOccurrence, ...]


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


def _diagnostic(
    *,
    code: str,
    message: str,
    span: TextSpan | None,
) -> AnalysisDiagnostic:
    """Build one static-analysis error diagnostic."""
    return AnalysisDiagnostic(
        code=code,
        message=message,
        severity="error",
        span=span,
    )


def _validation_error_to_diagnostic(
    *,
    error: ValidationError,
    span: TextSpan | None,
) -> AnalysisDiagnostic:
    """Convert one runtime ValidationError to analysis diagnostic payload."""
    return _diagnostic(
        code=error.code,
        message=error.message,
        span=span,
    )


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


def _build_base_tensor_op(
    *,
    op_name: str,
    lhs: AxisSide,
    rhs: AxisSide,
) -> TensorOp:
    """Construct one base TensorOp for parsed op call."""
    if op_name == "contract":
        validate_contract_atomic_terms(Signature(inputs=lhs, outputs=rhs))

    supports_reducer = op_name in {"reduce", "einop"}
    return TensorOp.from_base_spec(
        name=op_name,
        lhs=lhs,
        rhs=rhs,
        supports_reducer=supports_reducer,
    )


def _bind_base_call_arguments(
    *,
    op_name: str,
    call_expr: ast.Call,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> _BaseCallArguments | None:
    """Bind one base operation constructor call against the real entrypoint."""
    call_span = context.span_from_ast_node(call_expr)

    if any(isinstance(argument, ast.Starred) for argument in call_expr.args):
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message=(
                    "operation constructors do not support starred positional "
                    "arguments in static analysis"
                ),
                span=call_span,
            )
        )
        return None

    if any(keyword.arg is None for keyword in call_expr.keywords):
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message=(
                    "operation constructors do not support **kwargs in static analysis"
                ),
                span=call_span,
            )
        )
        return None

    signature = _ENTRYPOINT_SIGNATURES.get(op_name)
    if signature is None:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message=f"unsupported operation constructor: {op_name}",
                span=call_span,
            )
        )
        return None

    keyword_arguments = {
        keyword.arg: keyword.value
        for keyword in call_expr.keywords
        if keyword.arg is not None
    }
    try:
        bound_arguments = signature.bind(*call_expr.args, **keyword_arguments)
    except TypeError as error:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message=str(error),
                span=call_span,
            )
        )
        return None

    lhs_expr = bound_arguments.arguments.get("lhs")
    rhs_expr = bound_arguments.arguments.get("rhs")
    if not isinstance(lhs_expr, ast.expr) or not isinstance(rhs_expr, ast.expr):
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message="operation constructor binding did not produce lhs/rhs",
                span=call_span,
            )
        )
        return None

    return _BaseCallArguments(lhs_expr=lhs_expr, rhs_expr=rhs_expr)


def _parse_base_op_call(
    *,
    call_expr: ast.Call,
    context: _SnippetContext,
) -> _EvaluatedCall | None:
    """Parse/evaluate one base `einf.<op>(lhs, rhs)` call."""
    op_name = context.bindings.resolve_einf_op(call_expr.func)
    if op_name is None or op_name not in EINF_OP_NAMES:
        return None

    call_diagnostics: list[AnalysisDiagnostic] = []
    bound_arguments = _bind_base_call_arguments(
        op_name=op_name,
        call_expr=call_expr,
        context=context,
        diagnostics=call_diagnostics,
    )
    if bound_arguments is None:
        return _EvaluatedCall(
            base_call=None, op=None, diagnostics=tuple(call_diagnostics)
        )

    lhs = _parse_side_spec(
        expr=bound_arguments.lhs_expr,
        side="lhs",
        context=context,
        diagnostics=call_diagnostics,
    )
    rhs = _parse_side_spec(
        expr=bound_arguments.rhs_expr,
        side="rhs",
        context=context,
        diagnostics=call_diagnostics,
    )

    call_span = context.span_from_ast_node(call_expr)
    if call_span is None or lhs is None or rhs is None:
        return _EvaluatedCall(
            base_call=None, op=None, diagnostics=tuple(call_diagnostics)
        )

    parsed_call = _CallParseResult(
        op_name=op_name,
        span=call_span,
        lhs=lhs,
        rhs=rhs,
    )

    try:
        op = _build_base_tensor_op(
            op_name=op_name, lhs=lhs.axis_side, rhs=rhs.axis_side
        )
    except ValidationError as error:
        call_diagnostics.append(
            _validation_error_to_diagnostic(error=error, span=call_span)
        )
        return _EvaluatedCall(
            base_call=parsed_call,
            op=None,
            diagnostics=tuple(call_diagnostics),
        )
    except (TypeError, ValueError, AttributeError) as error:
        call_diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_CALL_SHAPE_ERROR,
                message=str(error),
                span=call_span,
            )
        )
        return _EvaluatedCall(
            base_call=parsed_call,
            op=None,
            diagnostics=tuple(call_diagnostics),
        )

    return _EvaluatedCall(
        base_call=parsed_call,
        op=op,
        diagnostics=tuple(call_diagnostics),
    )


def _parse_with_sizes_call(
    *,
    call_expr: ast.Call,
    op: TensorOp,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> TensorOp | None:
    """Parse/apply one `.with_sizes(...)` call."""
    call_span = context.span_from_ast_node(call_expr)

    if call_expr.args:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_WITH_SIZES_ERROR,
                message="with_sizes only accepts keyword bindings",
                span=call_span,
            )
        )
        return None

    def parse_integer_literal(expr: ast.expr) -> int | None:
        if isinstance(expr, ast.Constant) and type(expr.value) is int:
            return expr.value
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            operand = expr.operand
            if isinstance(operand, ast.Constant) and type(operand.value) is int:
                if isinstance(expr.op, ast.USub):
                    return -operand.value
                return operand.value
        return None

    sizes: dict[str, int] = {}
    has_parse_error = False
    for keyword in call_expr.keywords:
        if keyword.arg is None:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_WITH_SIZES_ERROR,
                    message="with_sizes does not support **kwargs in static analysis",
                    span=context.span_from_ast_node(keyword),
                )
            )
            has_parse_error = True
            continue

        value_expr = keyword.value
        parsed_value = parse_integer_literal(value_expr)
        if parsed_value is not None:
            sizes[keyword.arg] = parsed_value
            continue

        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_WITH_SIZES_ERROR,
                message="with_sizes values must be integer literals for static analysis",
                span=context.span_from_ast_node(value_expr),
            )
        )
        has_parse_error = True

    if has_parse_error:
        return None

    try:
        return op.with_sizes(**sizes)
    except ValidationError as error:
        diagnostics.append(_validation_error_to_diagnostic(error=error, span=call_span))
        return None
    except (TypeError, ValueError) as error:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_WITH_SIZES_ERROR,
                message=str(error),
                span=call_span,
            )
        )
        return None


def _parse_reducer_literal(
    *,
    reducer_expr: ast.expr,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> Reducer | None:
    """Parse one reducer literal for static reduce_by analysis."""
    if isinstance(reducer_expr, ast.Constant) and isinstance(reducer_expr.value, str):
        return reducer_expr.value

    if isinstance(reducer_expr, ast.Name):
        return _analysis_reducer_callable

    if isinstance(reducer_expr, ast.Attribute):
        return _analysis_reducer_callable

    diagnostics.append(
        _diagnostic(
            code=_ANALYSIS_REDUCE_BY_ERROR,
            message=(
                "reduce_by reducers must be string literals or simple callable symbols"
            ),
            span=context.span_from_ast_node(reducer_expr),
        )
    )
    return None


def _parse_reduce_phase_argument(
    *,
    phase_expr: ast.expr,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> _ReducePhaseArg | None:
    """Parse one reduce_by phase tuple `(ax[...], reducer)`."""
    if not isinstance(phase_expr, ast.Tuple) or len(phase_expr.elts) != 2:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_REDUCE_BY_ERROR,
                message="reduce_by phase must be `(ax[...], reducer)`",
                span=context.span_from_ast_node(phase_expr),
            )
        )
        return None

    phase_axis_expr = phase_expr.elts[0]
    occurrences: list[_AxisOccurrence] = []
    parsed_terms = _parse_axis_terms_expression(
        expr=phase_axis_expr,
        side="rhs",
        context=context,
        occurrences=occurrences,
        diagnostics=diagnostics,
    )
    if parsed_terms is None:
        return None

    reducer = _parse_reducer_literal(
        reducer_expr=phase_expr.elts[1],
        context=context,
        diagnostics=diagnostics,
    )
    if reducer is None:
        return None

    return (parsed_terms, reducer)


def _parse_reduce_by_call(
    *,
    call_expr: ast.Call,
    op: TensorOp,
    context: _SnippetContext,
    diagnostics: list[AnalysisDiagnostic],
) -> TensorOp | None:
    """Parse/apply one `.reduce_by(...)` call."""
    call_span = context.span_from_ast_node(call_expr)

    if call_expr.keywords:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_REDUCE_BY_ERROR,
                message="reduce_by only accepts positional arguments",
                span=call_span,
            )
        )
        return None

    if not call_expr.args:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_REDUCE_BY_ERROR,
                message="reduce_by requires at least one reducer argument",
                span=call_span,
            )
        )
        return None

    parsed_args: list[_ReduceByFirstArg] = []
    for index, argument in enumerate(call_expr.args):
        if index == 0:
            if isinstance(argument, ast.Tuple):
                phase = _parse_reduce_phase_argument(
                    phase_expr=argument,
                    context=context,
                    diagnostics=diagnostics,
                )
                if phase is None:
                    return None
                parsed_args.append(phase)
                continue

            reducer = _parse_reducer_literal(
                reducer_expr=argument,
                context=context,
                diagnostics=diagnostics,
            )
            if reducer is None:
                return None
            parsed_args.append(reducer)
            continue

        phase = _parse_reduce_phase_argument(
            phase_expr=argument,
            context=context,
            diagnostics=diagnostics,
        )
        if phase is None:
            return None
        parsed_args.append(phase)

    first = parsed_args[0]
    tail = tuple(parsed_args[1:])
    try:
        if isinstance(first, tuple):
            if any(not isinstance(phase, tuple) for phase in tail):
                diagnostics.append(
                    _diagnostic(
                        code=_ANALYSIS_REDUCE_BY_ERROR,
                        message=(
                            "when first argument is a reducer phase, "
                            "all remaining arguments must be reducer phases"
                        ),
                        span=call_span,
                    )
                )
                return None
            typed_tail = tuple(phase for phase in tail if isinstance(phase, tuple))
            return op.reduce_by(first, *typed_tail)

        if tail:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_REDUCE_BY_ERROR,
                    message=(
                        "when first argument is a reducer literal/callable, "
                        "no extra phase arguments are allowed"
                    ),
                    span=call_span,
                )
            )
            return None

        return op.reduce_by(first)
    except ValidationError as error:
        diagnostics.append(_validation_error_to_diagnostic(error=error, span=call_span))
        return None
    except (TypeError, ValueError, AttributeError) as error:
        diagnostics.append(
            _diagnostic(
                code=_ANALYSIS_REDUCE_BY_ERROR,
                message=str(error),
                span=call_span,
            )
        )
        return None


def _evaluate_call_expression(
    *,
    call_expr: ast.Call,
    context: _SnippetContext,
) -> _EvaluatedCall | None:
    """Parse/evaluate one call expression with optional TensorOp method chain."""
    direct = _parse_base_op_call(call_expr=call_expr, context=context)
    if direct is not None:
        return direct

    if not isinstance(call_expr.func, ast.Attribute):
        return None
    method_name = call_expr.func.attr
    receiver = call_expr.func.value
    if not isinstance(receiver, ast.Call):
        return None

    receiver_eval = _evaluate_call_expression(call_expr=receiver, context=context)
    if receiver_eval is None:
        return None

    diagnostics = list(receiver_eval.diagnostics)
    op = receiver_eval.op
    if op is None:
        return _EvaluatedCall(
            base_call=receiver_eval.base_call,
            op=None,
            diagnostics=tuple(diagnostics),
        )

    if method_name == "with_sizes":
        updated_op = _parse_with_sizes_call(
            call_expr=call_expr,
            op=op,
            context=context,
            diagnostics=diagnostics,
        )
    elif method_name == "reduce_by":
        updated_op = _parse_reduce_by_call(
            call_expr=call_expr,
            op=op,
            context=context,
            diagnostics=diagnostics,
        )
    else:
        updated_op = None

    return _EvaluatedCall(
        base_call=receiver_eval.base_call,
        op=updated_op,
        diagnostics=tuple(diagnostics),
    )


def _parse_call_expression(
    *,
    module_source: SourceText,
    call_span: TextSpan,
    bindings: CallBindings,
) -> _EvaluatedCall | None:
    """Parse one call span into one evaluated `einf` call record."""
    expression_source = module_source.slice(call_span)
    try:
        parsed_expression = ast.parse(expression_source, mode="eval")
    except SyntaxError:
        return None

    body = parsed_expression.body
    if not isinstance(body, ast.Call):
        return None
    return _evaluate_call_expression(
        call_expr=body,
        context=_SnippetContext(
            module_source=module_source,
            base_span=call_span,
            bindings=bindings,
        ),
    )


def _axis_roles(
    *,
    op_name: str,
    side: _CallSide,
    axis_name: str,
    lhs_axis_names: frozenset[str],
    rhs_axis_names: frozenset[str],
) -> tuple[str, ...]:
    """Derive semantic token roles for one axis occurrence."""
    roles: list[str] = [side]
    in_lhs = axis_name in lhs_axis_names
    in_rhs = axis_name in rhs_axis_names

    if in_lhs and in_rhs:
        roles.append("shared")
    if op_name in {"contract", "einop"} and in_lhs and not in_rhs:
        roles.append("contracted")
    if op_name == "reduce" and in_lhs and not in_rhs:
        roles.append("reduced")
    if op_name == "repeat" and in_rhs and not in_lhs:
        roles.append("introduced")
    return tuple(roles)


def _build_missing_rhs_axis_diagnostics(
    *,
    call: _CallParseResult,
) -> tuple[AnalysisDiagnostic, ...]:
    """Build diagnostics for rhs axes missing from lhs."""
    if call.op_name not in _VALIDATE_RHS_SUBSET_OPS:
        return ()

    missing_axis_names = sorted(call.rhs.axis_names - call.lhs.axis_names)
    if not missing_axis_names:
        return ()

    diagnostics: list[AnalysisDiagnostic] = []
    for missing_axis_name in missing_axis_names:
        matched_occurrences = [
            occurrence
            for occurrence in call.rhs.occurrences
            if occurrence.name == missing_axis_name
        ]
        if not matched_occurrences:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                    message=(
                        f"axis '{missing_axis_name}' appears on rhs but not in lhs"
                    ),
                    span=None,
                )
            )
            continue

        for occurrence in matched_occurrences:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                    message=(
                        f"axis '{missing_axis_name}' appears on rhs but not in lhs"
                    ),
                    span=occurrence.span,
                )
            )
    return tuple(diagnostics)


def _parent_map(module: ParsedModule) -> dict[int, int]:
    """Build node-id parent index for one parsed module."""
    parent_by_child: dict[int, int] = {}
    for node in module.nodes:
        for child_id in node.child_ids:
            parent_by_child[child_id] = node.node_id
    return parent_by_child


def _is_method_chain_inner_call(
    *,
    module: ParsedModule,
    parent_by_child: dict[int, int],
    node_id: int,
) -> bool:
    """Return whether one call node is inner call of `call().method(...)` chain."""
    parent_id = parent_by_child.get(node_id)
    if parent_id is None:
        return False
    parent = module.node(parent_id)
    if parent.kind != "Attribute":
        return False

    grandparent_id = parent_by_child.get(parent_id)
    if grandparent_id is None:
        return False
    grandparent = module.node(grandparent_id)
    return grandparent.kind == "Call"


def analyze_einf_calls(
    module: ParsedModule,
) -> tuple[tuple[AnalysisDiagnostic, ...], tuple[AxisToken, ...]]:
    """Analyze `einf` calls in one parsed module."""
    module_source = SourceText(module.source)
    call_bindings = build_call_bindings(
        source=module.source,
        path=module.path,
        source_text=module_source,
    )
    diagnostics: list[AnalysisDiagnostic] = []
    raw_tokens: list[tuple[str, TextSpan, tuple[str, ...]]] = []
    seen_base_call_spans: set[TextSpan] = set()

    parent_by_child = _parent_map(module)
    for parsed_node in module.nodes:
        if parsed_node.kind != "Call" or parsed_node.span is None:
            continue
        if _is_method_chain_inner_call(
            module=module,
            parent_by_child=parent_by_child,
            node_id=parsed_node.node_id,
        ):
            continue

        bindings = call_bindings.get(parsed_node.span)
        if bindings is None:
            continue

        call_result = _parse_call_expression(
            module_source=module_source,
            call_span=parsed_node.span,
            bindings=bindings,
        )
        if call_result is None:
            continue

        diagnostics.extend(call_result.diagnostics)
        call = call_result.base_call
        if call is None:
            continue
        if call.span in seen_base_call_spans:
            continue
        seen_base_call_spans.add(call.span)

        lhs_axis_names = call.lhs.axis_names
        rhs_axis_names = call.rhs.axis_names
        diagnostics.extend(_build_missing_rhs_axis_diagnostics(call=call))

        for occurrence in (*call.lhs.occurrences, *call.rhs.occurrences):
            raw_tokens.append(
                (
                    occurrence.name,
                    occurrence.span,
                    _axis_roles(
                        op_name=call.op_name,
                        side=occurrence.side,
                        axis_name=occurrence.name,
                        lhs_axis_names=lhs_axis_names,
                        rhs_axis_names=rhs_axis_names,
                    ),
                )
            )

    group_by_axis_name: dict[str, int] = {}
    axis_tokens: list[AxisToken] = []
    for axis_name, span, roles in raw_tokens:
        group = group_by_axis_name.get(axis_name)
        if group is None:
            group = len(group_by_axis_name)
            group_by_axis_name[axis_name] = group
        axis_tokens.append(
            AxisToken(
                name=axis_name,
                span=span,
                group=group,
                roles=roles,
            )
        )

    axis_tokens.sort(
        key=lambda token: (token.span.start.line, token.span.start.column, token.name)
    )
    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic.span.start.line if diagnostic.span is not None else -1,
            diagnostic.span.start.column if diagnostic.span is not None else -1,
            diagnostic.code,
        )
    )
    return tuple(diagnostics), tuple(axis_tokens)
