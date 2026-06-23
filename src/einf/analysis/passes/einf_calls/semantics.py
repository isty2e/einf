import ast
import inspect

from einf.analysis.model import AnalysisDiagnostic, TextSpan
from einf.analysis.passes.call_resolution import EINF_OP_NAMES, CallBindings
from einf.analysis.source import SourceText
from einf.axis import AxisSide
from einf.diagnostics import ValidationError
from einf.operations import TensorOp, contract, einop, rearrange, reduce, repeat, view
from einf.operations.validation import validate_contract_atomic_terms
from einf.signature import Signature

from .diagnostics import (
    _ANALYSIS_CALL_SHAPE_ERROR,
    _ANALYSIS_WITH_SIZES_ERROR,
    _diagnostic,
    _validation_error_to_diagnostic,
)
from .model import (
    _BaseCallArguments,
    _CallParseResult,
    _EvaluatedCall,
    _SnippetContext,
)
from .reducers import _parse_reduce_by_call
from .syntax import _parse_side_spec

_ENTRYPOINT_SIGNATURES = {
    "view": inspect.signature(view),
    "rearrange": inspect.signature(rearrange),
    "repeat": inspect.signature(repeat),
    "reduce": inspect.signature(reduce),
    "contract": inspect.signature(contract),
    "einop": inspect.signature(einop),
}

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
