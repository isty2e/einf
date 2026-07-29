import ast

from einf.analysis.model import AnalysisDiagnostic
from einf.diagnostics import ValidationError
from einf.operations.tensor_op import TensorOp
from einf.reduction.schema import Reducer
from einf.tensor_types import TensorLike

from .diagnostics import (
    _ANALYSIS_REDUCE_BY_ERROR,
    _diagnostic,
    _validation_error_to_diagnostic,
)
from .model import (
    _AxisOccurrence,
    _ReduceByFirstArg,
    _ReducePhaseArg,
    _SnippetContext,
)
from .syntax import _parse_axis_terms_expression


def _analysis_reducer_callable(*_args: TensorLike) -> int:
    """Static placeholder reducer for symbol-only callable references."""
    return 0


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
