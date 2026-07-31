import asyncio
import json
import sys
from pathlib import Path

from einf.analysis.checkers.base import CheckerAdapter
from einf.analysis.checkers.execution import CheckerExecutionPolicy, CheckerExecutor
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.checkers.pyrefly import PyreflyAdapter
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.checkers.ty import TyAdapter
from einf.analysis.checkers.zuban import ZubanAdapter
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import AstParserBackend
from einf.analysis.validator.run import run_validation


class _StubCheckerAdapter(CheckerAdapter):
    name = "stub"
    executable = sys.executable

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        _ = request
        return (sys.executable, "-c", "")

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr
        target = request.targets[0]
        return CheckerResult(
            diagnostics=(
                CheckerDiagnostic(
                    tool=self.name,
                    path=target,
                    code="stub-rule",
                    message="stub checker diagnostic",
                    severity="warning",
                    span=TextSpan(
                        start=TextPosition(line=1, column=0),
                        end=TextPosition(line=1, column=1),
                    ),
                ),
            ),
            failures=(
                CheckerFailure(
                    tool=self.name,
                    kind="execution_error",
                    message="stub checker failed after emitting diagnostics",
                ),
            ),
        )


PYRIGHT_OUTPUT = """
{
  "generalDiagnostics": [
    {
      "file": "/tmp/sample.py",
      "severity": "warning",
      "message": "pyright message",
      "range": {
        "start": {"line": 2, "character": 16},
        "end": {"line": 2, "character": 17}
      },
      "rule": "reportAssignmentType"
    }
  ]
}
"""


PYREFLY_OUTPUT = """
{
  "errors": [
    {
      "line": 3,
      "column": 17,
      "stop_line": 3,
      "stop_column": 18,
      "path": "/tmp/sample.py",
      "code": -2,
      "name": "bad-assignment",
      "severity": "error",
      "description": "pyrefly message"
    }
  ]
}
"""


TY_OUTPUT = (
    "/tmp/sample.py:3:17: error[invalid-assignment] ty message\nFound 1 diagnostic\n"
)


ZUBAN_OUTPUT = (
    "sample.py:3:17:3:18: error: zuban message  [assignment]\n"
    "Found 1 error in 1 file (checked 1 source file)\n"
)


def _write_checker_executable(
    *,
    path: Path,
    stdout: str,
    exit_code: int,
) -> None:
    path.write_text(
        f"#!{sys.executable}\nprint({stdout!r})\nraise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _request(project_root: Path) -> CheckerRequest:
    return CheckerRequest(
        targets=(project_root / "sample.py",),
        project_root=project_root,
    )


def _run_adapter(
    adapter: CheckerAdapter,
    *,
    target: Path,
    project_root: Path,
) -> CheckerResult:
    async def execute() -> CheckerResult:
        executor = CheckerExecutor(CheckerExecutionPolicy())
        return await executor.run(
            adapter,
            CheckerRequest(targets=(target,), project_root=project_root),
        )

    return asyncio.run(execute())


def test_pyright_adapter_parses_json_output() -> None:
    adapter = PyrightAdapter(name="basedpyright", executable="basedpyright")
    result = adapter.parse_output(
        stdout=PYRIGHT_OUTPUT,
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.tool == "basedpyright"
    assert diagnostic.code == "reportAssignmentType"
    assert diagnostic.severity == "warning"
    assert diagnostic.path == Path("/tmp/sample.py").resolve(strict=False)
    assert diagnostic.span == TextSpan(
        start=TextPosition(line=3, column=16),
        end=TextPosition(line=3, column=17),
    )


def test_pyright_adapter_preserves_diagnostic_without_range() -> None:
    adapter = PyrightAdapter(name="basedpyright", executable="basedpyright")
    result = adapter.parse_output(
        stdout=json.dumps(
            {
                "generalDiagnostics": [
                    {
                        "file": "/tmp/a.py",
                        "severity": "error",
                        "message": "Import cycle detected",
                        "rule": "reportImportCycles",
                    }
                ]
            }
        ),
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "reportImportCycles"
    assert diagnostic.message == "Import cycle detected"
    assert diagnostic.span is None


def test_pyrefly_adapter_parses_json_output() -> None:
    adapter = PyreflyAdapter()
    result = adapter.parse_output(
        stdout=PYREFLY_OUTPUT,
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "bad-assignment"
    assert diagnostic.severity == "error"
    assert diagnostic.path == Path("/tmp/sample.py").resolve(strict=False)
    assert diagnostic.span == TextSpan(
        start=TextPosition(line=3, column=16),
        end=TextPosition(line=3, column=17),
    )


def test_pyrefly_adapter_normalizes_severity() -> None:
    expected_severities = {
        "error": "error",
        "warn": "warning",
        "info": "info",
    }

    for severity, expected in expected_severities.items():
        record = json.loads(PYREFLY_OUTPUT)["errors"][0]
        record["severity"] = severity
        result = PyreflyAdapter().parse_output(
            stdout=json.dumps({"errors": [record]}),
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert result.failures == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == expected


def test_pyrefly_adapter_rejects_invalid_severity() -> None:
    for severity in (None, "warning", 1):
        record = json.loads(PYREFLY_OUTPUT)["errors"][0]
        if severity is None:
            del record["severity"]
        else:
            record["severity"] = severity
        result = PyreflyAdapter().parse_output(
            stdout=json.dumps({"errors": [record]}),
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_parse_error"


def test_json_adapters_preserve_valid_diagnostics_and_reject_malformed_records() -> (
    None
):
    pyright_record = json.loads(PYRIGHT_OUTPUT)["generalDiagnostics"][0]
    pyrefly_record = json.loads(PYREFLY_OUTPUT)["errors"][0]
    cases: tuple[tuple[CheckerAdapter, str], ...] = (
        (
            PyrightAdapter(name="pyright", executable="pyright"),
            json.dumps(
                {
                    "generalDiagnostics": [
                        pyright_record,
                        {"file": "sample.py"},
                        None,
                    ]
                }
            ),
        ),
        (
            PyreflyAdapter(),
            json.dumps(
                {
                    "errors": [
                        pyrefly_record,
                        {"path": "sample.py"},
                        None,
                    ]
                }
            ),
        ),
    )

    for adapter, stdout in cases:
        result = adapter.parse_output(
            stdout=stdout,
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert len(result.diagnostics) == 1
        assert len(result.failures) == 1
        assert result.failures[0].tool == adapter.name
        assert result.failures[0].kind == "output_parse_error"


def test_text_adapters_preserve_valid_diagnostics_and_reject_malformed_lines() -> None:
    cases: tuple[tuple[CheckerAdapter, str], ...] = (
        (
            TyAdapter(),
            TY_OUTPUT + "unrecognized ty line\nanother unrecognized ty line\n",
        ),
        (
            ZubanAdapter(),
            ZUBAN_OUTPUT + "unrecognized zuban line\nanother unrecognized zuban line\n",
        ),
    )

    for adapter, stdout in cases:
        result = adapter.parse_output(
            stdout=stdout,
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert len(result.diagnostics) == 1
        assert len(result.failures) == 1
        assert result.failures[0].tool == adapter.name
        assert result.failures[0].kind == "output_parse_error"


def test_checker_adapters_reject_invalid_report_paths() -> None:
    cases: tuple[tuple[CheckerAdapter, str], ...] = (
        (
            PyrightAdapter(name="pyright", executable="pyright"),
            json.dumps(
                {
                    "generalDiagnostics": [
                        {
                            "file": "invalid\0.py",
                            "severity": "error",
                            "message": "invalid path",
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 1},
                            },
                        }
                    ]
                }
            ),
        ),
        (
            PyreflyAdapter(),
            json.dumps(
                {
                    "errors": [
                        {
                            "path": "invalid\0.py",
                            "description": "invalid path",
                            "line": 1,
                            "column": 1,
                            "stop_line": 1,
                            "stop_column": 2,
                        }
                    ]
                }
            ),
        ),
        (TyAdapter(), "invalid\0.py:1:1: error invalid path\n"),
        (
            ZubanAdapter(),
            "invalid\0.py:1:1:1:2: error: invalid path\n",
        ),
    )

    for adapter, stdout in cases:
        result = adapter.parse_output(
            stdout=stdout,
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].tool == adapter.name
        assert result.failures[0].kind == "output_parse_error"


def test_checker_adapters_reject_invalid_diagnostic_coordinates() -> None:
    cases: tuple[tuple[CheckerAdapter, str], ...] = (
        (
            PyrightAdapter(name="pyright", executable="pyright"),
            json.dumps(
                {
                    "generalDiagnostics": [
                        {
                            "file": "sample.py",
                            "severity": "error",
                            "message": "invalid coordinate",
                            "range": {
                                "start": {"line": -1, "character": 0},
                                "end": {"line": 0, "character": 1},
                            },
                        }
                    ]
                }
            ),
        ),
        (
            PyreflyAdapter(),
            json.dumps(
                {
                    "errors": [
                        {
                            "path": "sample.py",
                            "description": "invalid coordinate",
                            "line": 0,
                            "column": 1,
                            "stop_line": 1,
                            "stop_column": 2,
                        }
                    ]
                }
            ),
        ),
        (TyAdapter(), "sample.py:0:1: error invalid coordinate\n"),
        (
            ZubanAdapter(),
            "sample.py:1:1:0:1: error: invalid coordinate\n",
        ),
    )

    for adapter, stdout in cases:
        result = adapter.parse_output(
            stdout=stdout,
            stderr="",
            request=_request(Path("/tmp")),
        )

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].tool == adapter.name
        assert result.failures[0].kind == "output_parse_error"


def test_ty_adapter_parses_concise_output() -> None:
    adapter = TyAdapter()
    result = adapter.parse_output(
        stdout=TY_OUTPUT,
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "invalid-assignment"
    assert diagnostic.message == "ty message"
    assert diagnostic.path == Path("/tmp/sample.py").resolve(strict=False)
    assert diagnostic.span == TextSpan(
        start=TextPosition(line=3, column=16),
        end=TextPosition(line=3, column=17),
    )


def test_zuban_adapter_parses_text_output() -> None:
    adapter = ZubanAdapter()
    result = adapter.parse_output(
        stdout=ZUBAN_OUTPUT,
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "assignment"
    assert diagnostic.message == "zuban message"
    assert diagnostic.path == Path("/tmp/sample.py").resolve(strict=False)
    assert diagnostic.span == TextSpan(
        start=TextPosition(line=3, column=16),
        end=TextPosition(line=3, column=17),
    )


def test_zuban_adapter_preserves_valid_multiline_span() -> None:
    result = ZubanAdapter().parse_output(
        stdout="sample.py:1:5:2:1: error: multiline diagnostic\n",
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].span == TextSpan(
        start=TextPosition(line=1, column=4),
        end=TextPosition(line=2, column=0),
    )


def test_ty_adapter_rejects_unrecognized_output() -> None:
    result = TyAdapter().parse_output(
        stdout="new ty diagnostic format\n",
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"
    assert result.failures[0].message == (
        "ty output contained an unrecognized line: new ty diagnostic format"
    )


def test_ty_adapter_accepts_success_summary() -> None:
    result = TyAdapter().parse_output(
        stdout="All checks passed!\n",
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.diagnostics == ()
    assert result.failures == ()


def test_zuban_adapter_rejects_unrecognized_output() -> None:
    result = ZubanAdapter().parse_output(
        stdout="new zuban diagnostic format\n",
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"
    assert result.failures[0].message == (
        "zuban output contained an unrecognized line: new zuban diagnostic format"
    )


def test_zuban_adapter_normalizes_note_diagnostics() -> None:
    result = ZubanAdapter().parse_output(
        stdout=(
            'sample.py:1:13:1:14: note: Revealed type is "Literal[1]?"\n'
            "Success: no issues found in 1 source file\n"
        ),
        stderr="",
        request=_request(Path("/tmp")),
    )

    assert result.failures == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code is None
    assert diagnostic.severity == "info"
    assert diagnostic.message == 'Revealed type is "Literal[1]?"'


def test_ty_adapter_rejects_nonzero_summary_without_diagnostics(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ty"
    _write_checker_executable(
        path=executable,
        stdout="Found 1 diagnostic",
        exit_code=1,
    )
    target = tmp_path / "sample.py"
    target.write_text("value: int = 1\n", encoding="utf-8")

    result = _run_adapter(
        TyAdapter(executable=str(executable)),
        target=target,
        project_root=tmp_path,
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"


def test_zuban_adapter_rejects_nonzero_summary_without_diagnostics(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "zuban"
    _write_checker_executable(
        path=executable,
        stdout="Found 1 error in 1 file (checked 1 source file)",
        exit_code=1,
    )
    target = tmp_path / "sample.py"
    target.write_text("value: int = 1\n", encoding="utf-8")

    result = _run_adapter(
        ZubanAdapter(executable=str(executable)),
        target=target,
        project_root=tmp_path,
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"


def test_run_validation_merges_checker_diagnostics_and_failures(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text(
        "from einf import ax, axes, rearrange\n"
        'b = axes("b")[0]\n'
        "rearrange(ax[b], ax[b])\n",
        encoding="utf-8",
    )

    report = run_validation(
        targets=(target,),
        parser_backend=AstParserBackend(),
        checker_adapters=(_StubCheckerAdapter(),),
    )

    assert len(report.checker_failures) == 1
    assert report.checker_failures[0].tool == "stub"
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.diagnostics == ()
    assert len(file_report.checker_diagnostics) == 1
    assert file_report.checker_diagnostics[0].tool == "stub"
    assert file_report.checker_diagnostics[0].severity == "warning"
    assert report.exit_code() == 1


def test_checker_adapter_reports_unavailable_executable(tmp_path: Path) -> None:
    adapter = PyrightAdapter(name="pyright", executable="definitely-missing-checker")

    result = _run_adapter(
        adapter,
        target=tmp_path / "sample.py",
        project_root=tmp_path,
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].tool == "pyright"
    assert result.failures[0].kind == "unavailable"
