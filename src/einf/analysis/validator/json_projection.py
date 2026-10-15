from typing import Literal, TypedDict

from einf.analysis.checkers import CheckerDiagnostic, CheckerFailure
from einf.analysis.checkers.model import CheckerFailureKind
from einf.analysis.model import (
    AnalysisDiagnostic,
    AxisCrossSideRelation,
    AxisOccurrenceSide,
    AxisOperationRole,
    AxisStructuralKind,
    AxisToken,
    DiagnosticSeverity,
    TextPosition,
    TextSpan,
)
from einf.analysis.report import (
    AnalysisFailure,
    AnalysisFailureKind,
    AnalysisFileReport,
)
from einf.analysis.validator.limits import (
    ValidationLimitFailure,
    ValidationLimitResource,
)
from einf.analysis.validator.model import (
    ValidationDiscoveryFailure,
    ValidationReport,
)


class TextPositionJson(TypedDict):
    """One-based line and zero-based character column in JSON output."""

    line: int
    column: int


class TextSpanJson(TypedDict):
    """Start and end positions of a source span in JSON output."""

    start: TextPositionJson
    end: TextPositionJson


class AnalysisDiagnosticJson(TypedDict):
    """Semantic diagnostic fields in validator JSON output."""

    code: str
    message: str
    severity: DiagnosticSeverity
    span: TextSpanJson | None


class CheckerDiagnosticJson(TypedDict):
    """External checker diagnostic fields in validator JSON output."""

    tool: str
    path: str
    code: str | None
    message: str
    severity: DiagnosticSeverity
    span: TextSpanJson | None


class CheckerFailureJson(TypedDict):
    """External checker failure fields in validator JSON output."""

    tool: str
    kind: CheckerFailureKind
    message: str


class ValidationDiscoveryFailureJson(TypedDict):
    """Directory traversal failure fields in validator JSON output."""

    path: str
    kind: Literal["directory_traversal_error"]
    message: str


class ValidationLimitFailureJson(TypedDict):
    """Resource quota failure fields in validator JSON output."""

    resource: ValidationLimitResource
    limit: int
    observed: int
    path: str


class AxisTokenJson(TypedDict):
    """Axis identity, role and source location in validator JSON output."""

    name: str
    kind: AxisStructuralKind
    side: AxisOccurrenceSide
    relation: AxisCrossSideRelation
    role: AxisOperationRole | None
    span: TextSpanJson
    group: int


class AnalysisFailureJson(TypedDict):
    """Per-file analysis failure fields in validator JSON output."""

    kind: AnalysisFailureKind
    message: str
    span: TextSpanJson | None


class AnalysisFileReportJson(TypedDict):
    """Diagnostics, axis tokens and failures for one source file."""

    path: str
    diagnostics: list[AnalysisDiagnosticJson]
    checker_diagnostics: list[CheckerDiagnosticJson]
    axis_tokens: list[AxisTokenJson]
    failures: list[AnalysisFailureJson]


class ValidationReportJson(TypedDict):
    """JSON-compatible projection of one validator report."""

    schema_version: str
    parser_backend: str
    checker_failures: list[CheckerFailureJson]
    discovery_failures: list[ValidationDiscoveryFailureJson]
    limit_failures: list[ValidationLimitFailureJson]
    files: list[AnalysisFileReportJson]


def _project_text_position(position: TextPosition) -> TextPositionJson:
    return {"line": position.line, "column": position.column}


def _project_text_span(span: TextSpan) -> TextSpanJson:
    return {
        "start": _project_text_position(span.start),
        "end": _project_text_position(span.end),
    }


def _project_optional_text_span(span: TextSpan | None) -> TextSpanJson | None:
    if span is None:
        return None
    return _project_text_span(span)


def _project_analysis_diagnostic(
    diagnostic: AnalysisDiagnostic,
) -> AnalysisDiagnosticJson:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "span": _project_optional_text_span(diagnostic.span),
    }


def _project_checker_diagnostic(
    diagnostic: CheckerDiagnostic,
) -> CheckerDiagnosticJson:
    return {
        "tool": diagnostic.tool,
        "path": str(diagnostic.path),
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "span": _project_optional_text_span(diagnostic.span),
    }


def _project_checker_failure(failure: CheckerFailure) -> CheckerFailureJson:
    return {
        "tool": failure.tool,
        "kind": failure.kind,
        "message": failure.message,
    }


def _project_discovery_failure(
    failure: ValidationDiscoveryFailure,
) -> ValidationDiscoveryFailureJson:
    return {
        "path": failure.path,
        "kind": failure.kind,
        "message": failure.message,
    }


def _project_limit_failure(
    failure: ValidationLimitFailure,
) -> ValidationLimitFailureJson:
    return {
        "resource": failure.resource,
        "limit": failure.limit,
        "observed": failure.observed,
        "path": failure.path,
    }


def _project_axis_token(token: AxisToken) -> AxisTokenJson:
    return {
        "name": token.name,
        "kind": token.kind,
        "side": token.side,
        "relation": token.relation,
        "role": token.role,
        "span": _project_text_span(token.span),
        "group": token.group,
    }


def _project_analysis_failure(
    failure: AnalysisFailure,
) -> AnalysisFailureJson:
    return {
        "kind": failure.kind,
        "message": failure.message,
        "span": _project_optional_text_span(failure.span),
    }


def _project_file_report(
    file_report: AnalysisFileReport,
) -> AnalysisFileReportJson:
    return {
        "path": file_report.path,
        "diagnostics": [
            _project_analysis_diagnostic(diagnostic)
            for diagnostic in file_report.diagnostics
        ],
        "checker_diagnostics": [
            _project_checker_diagnostic(diagnostic)
            for diagnostic in file_report.checker_diagnostics
        ],
        "axis_tokens": [
            _project_axis_token(token) for token in file_report.axis_tokens
        ],
        "failures": [
            _project_analysis_failure(failure) for failure in file_report.failures
        ],
    }


def project_validation_report(report: ValidationReport) -> ValidationReportJson:
    """Project a canonical validation report into the stable JSON schema."""
    return {
        "schema_version": report.schema_version,
        "parser_backend": report.parser_backend,
        "checker_failures": [
            _project_checker_failure(failure) for failure in report.checker_failures
        ],
        "discovery_failures": [
            _project_discovery_failure(failure) for failure in report.discovery_failures
        ],
        "limit_failures": [
            _project_limit_failure(failure) for failure in report.limit_failures
        ],
        "files": [_project_file_report(file_report) for file_report in report.files],
    }


__all__ = ["ValidationReportJson", "project_validation_report"]
