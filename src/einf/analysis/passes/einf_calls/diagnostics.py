from einf.analysis.model import AnalysisDiagnostic, TextSpan
from einf.diagnostics import ValidationError

_ANALYSIS_CALL_SHAPE_ERROR = "ANALYSIS_CALL_SHAPE_ERROR"
_ANALYSIS_SIDE_SPEC_ERROR = "ANALYSIS_SIDE_SPEC_ERROR"
_ANALYSIS_AXIS_TERM_ERROR = "ANALYSIS_AXIS_TERM_ERROR"
_ANALYSIS_WITH_SIZES_ERROR = "ANALYSIS_WITH_SIZES_ERROR"
_ANALYSIS_REDUCE_BY_ERROR = "ANALYSIS_REDUCE_BY_ERROR"
_ANALYSIS_AXIS_NOT_IN_INPUT = "ANALYSIS_AXIS_NOT_IN_INPUT"

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
