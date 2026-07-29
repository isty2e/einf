from typing import TypedDict

from einf.analysis.checkers import CheckerDiagnostic, CheckerFailure
from einf.analysis.checkers.model import CheckerFailureKind
from einf.analysis.model import (
    AnalysisDiagnostic,
    AxisToken,
    DiagnosticSeverity,
    TextPosition,
    TextSpan,
)
from einf.analysis.validator.model import (
    ValidationFailure,
    ValidationFailureKind,
    ValidationFileReport,
    ValidationReport,
)


class _TextPositionJson(TypedDict):
    line: int
    column: int


class _TextSpanJson(TypedDict):
    start: _TextPositionJson
    end: _TextPositionJson


class _AnalysisDiagnosticJson(TypedDict):
    code: str
    message: str
    severity: DiagnosticSeverity
    span: _TextSpanJson | None


class _CheckerDiagnosticJson(TypedDict):
    tool: str
    path: str
    code: str | None
    message: str
    severity: DiagnosticSeverity
    span: _TextSpanJson | None


class _CheckerFailureJson(TypedDict):
    tool: str
    kind: CheckerFailureKind
    message: str


class _AxisTokenJson(TypedDict):
    name: str
    span: _TextSpanJson
    group: int
    roles: list[str]


class _ValidationFailureJson(TypedDict):
    kind: ValidationFailureKind
    message: str
    span: _TextSpanJson | None


class _ValidationFileReportJson(TypedDict):
    path: str
    diagnostics: list[_AnalysisDiagnosticJson]
    checker_diagnostics: list[_CheckerDiagnosticJson]
    axis_tokens: list[_AxisTokenJson]
    failures: list[_ValidationFailureJson]


class ValidationReportJson(TypedDict):
    """JSON-compatible projection of one validator report."""

    schema_version: str
    parser_backend: str
    checker_failures: list[_CheckerFailureJson]
    files: list[_ValidationFileReportJson]


def _project_text_position(position: TextPosition) -> _TextPositionJson:
    return {"line": position.line, "column": position.column}


def _project_text_span(span: TextSpan) -> _TextSpanJson:
    return {
        "start": _project_text_position(span.start),
        "end": _project_text_position(span.end),
    }


def _project_optional_text_span(span: TextSpan | None) -> _TextSpanJson | None:
    if span is None:
        return None
    return _project_text_span(span)


def _project_analysis_diagnostic(
    diagnostic: AnalysisDiagnostic,
) -> _AnalysisDiagnosticJson:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "span": _project_optional_text_span(diagnostic.span),
    }


def _project_checker_diagnostic(
    diagnostic: CheckerDiagnostic,
) -> _CheckerDiagnosticJson:
    return {
        "tool": diagnostic.tool,
        "path": str(diagnostic.path),
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "span": _project_optional_text_span(diagnostic.span),
    }


def _project_checker_failure(failure: CheckerFailure) -> _CheckerFailureJson:
    return {
        "tool": failure.tool,
        "kind": failure.kind,
        "message": failure.message,
    }


def _project_axis_token(token: AxisToken) -> _AxisTokenJson:
    return {
        "name": token.name,
        "span": _project_text_span(token.span),
        "group": token.group,
        "roles": list(token.roles),
    }


def _project_validation_failure(
    failure: ValidationFailure,
) -> _ValidationFailureJson:
    return {
        "kind": failure.kind,
        "message": failure.message,
        "span": _project_optional_text_span(failure.span),
    }


def _project_file_report(
    file_report: ValidationFileReport,
) -> _ValidationFileReportJson:
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
            _project_validation_failure(failure) for failure in file_report.failures
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
        "files": [_project_file_report(file_report) for file_report in report.files],
    }


__all__ = ["ValidationReportJson", "project_validation_report"]
