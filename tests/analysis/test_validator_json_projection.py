import json
from pathlib import Path

from einf.analysis.checkers import CheckerDiagnostic, CheckerFailure
from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextPosition, TextSpan
from einf.analysis.validator.json_projection import project_validation_report
from einf.analysis.validator.model import (
    ValidationFailure,
    ValidationFileReport,
    ValidationReport,
)


def test_project_validation_report_converts_every_nested_domain_value() -> None:
    span = TextSpan(
        start=TextPosition(line=2, column=3),
        end=TextPosition(line=2, column=7),
    )
    report = ValidationReport(
        schema_version="0.1",
        parser_backend="ast",
        checker_failures=(
            CheckerFailure(
                tool="basedpyright",
                kind="execution_error",
                message="checker failed",
            ),
        ),
        files=(
            ValidationFileReport(
                path="/project/sample.py",
                diagnostics=(
                    AnalysisDiagnostic(
                        code="ANALYSIS_TEST",
                        message="semantic diagnostic",
                        severity="error",
                        span=span,
                    ),
                ),
                checker_diagnostics=(
                    CheckerDiagnostic(
                        tool="basedpyright",
                        path=Path("/project/sample.py"),
                        code="reportTest",
                        message="checker diagnostic",
                        severity="warning",
                        span=None,
                    ),
                ),
                axis_tokens=(
                    AxisToken(
                        name="batch",
                        span=span,
                        group=1,
                        roles=("input", "output"),
                    ),
                ),
                failures=(
                    ValidationFailure(
                        kind="parse_error",
                        message="parse failed",
                        span=span,
                    ),
                ),
            ),
        ),
    )

    projection = project_validation_report(report)

    assert projection == {
        "schema_version": "0.1",
        "parser_backend": "ast",
        "checker_failures": [
            {
                "tool": "basedpyright",
                "kind": "execution_error",
                "message": "checker failed",
            }
        ],
        "files": [
            {
                "path": "/project/sample.py",
                "diagnostics": [
                    {
                        "code": "ANALYSIS_TEST",
                        "message": "semantic diagnostic",
                        "severity": "error",
                        "span": {
                            "start": {"line": 2, "column": 3},
                            "end": {"line": 2, "column": 7},
                        },
                    }
                ],
                "checker_diagnostics": [
                    {
                        "tool": "basedpyright",
                        "path": "/project/sample.py",
                        "code": "reportTest",
                        "message": "checker diagnostic",
                        "severity": "warning",
                        "span": None,
                    }
                ],
                "axis_tokens": [
                    {
                        "name": "batch",
                        "span": {
                            "start": {"line": 2, "column": 3},
                            "end": {"line": 2, "column": 7},
                        },
                        "group": 1,
                        "roles": ["input", "output"],
                    }
                ],
                "failures": [
                    {
                        "kind": "parse_error",
                        "message": "parse failed",
                        "span": {
                            "start": {"line": 2, "column": 3},
                            "end": {"line": 2, "column": 7},
                        },
                    }
                ],
            }
        ],
    }
    json.dumps(projection)
