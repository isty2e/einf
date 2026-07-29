import asyncio
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
