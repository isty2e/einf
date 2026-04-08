import json
from pathlib import Path

from einf.analysis.parser import AstParserBackend
from einf.analysis.validator.cli import main
from einf.analysis.validator.run import run_validation

VALID_SOURCE = """from einf import ax, axes, rearrange
b = axes(\"b\")[0]
rearrange(ax[b], ax[b])
"""


DIAGNOSTIC_SOURCE = """from einf import ax, axes, reduce
b, n, z = axes(\"b\", \"n\", \"z\")
reduce(ax[b, n], ax[b, z])
"""


SYNTAX_ERROR_SOURCE = "from einf import rearrange\nrearrange(\n"


def test_run_validation_reports_semantic_diagnostics(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text(DIAGNOSTIC_SOURCE, encoding="utf-8")

    report = run_validation(targets=(target,), parser_backend=AstParserBackend())

    assert report.schema_version == "0.1"
    assert report.parser_backend == "ast"
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path == str(target.resolve())
    assert len(file_report.diagnostics) == 1
    assert file_report.checker_diagnostics == ()
    assert file_report.diagnostics[0].code == "ANALYSIS_AXIS_NOT_IN_INPUT"
    assert file_report.failures == ()
    assert report.checker_failures == ()
    assert report.exit_code() == 1


def test_run_validation_reports_syntax_error_failure(tmp_path: Path) -> None:
    target = tmp_path / "broken.py"
    target.write_text(SYNTAX_ERROR_SOURCE, encoding="utf-8")

    report = run_validation(targets=(target,), parser_backend=AstParserBackend())

    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.diagnostics == ()
    assert file_report.checker_diagnostics == ()
    assert file_report.axis_tokens == ()
    assert len(file_report.failures) == 1
    failure = file_report.failures[0]
    assert failure.kind == "parse_error"
    assert failure.span is not None
    assert failure.span.start.line == 2
    assert report.exit_code() == 1


def test_run_validation_reports_read_error_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"

    report = run_validation(targets=(missing,), parser_backend=AstParserBackend())

    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path == str(missing.resolve())
    assert file_report.diagnostics == ()
    assert file_report.checker_diagnostics == ()
    assert file_report.axis_tokens == ()
    assert len(file_report.failures) == 1
    assert file_report.failures[0].kind == "read_error"
    assert report.exit_code() == 1


def test_run_validation_recurses_directories_in_sorted_order(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    later = package_dir / "z_last.py"
    earlier = package_dir / "a_first.py"
    later.write_text(VALID_SOURCE, encoding="utf-8")
    earlier.write_text(VALID_SOURCE, encoding="utf-8")

    report = run_validation(targets=(package_dir,), parser_backend=AstParserBackend())

    assert tuple(file_report.path for file_report in report.files) == (
        str(earlier.resolve()),
        str(later.resolve()),
    )
    assert all(file_report.checker_diagnostics == () for file_report in report.files)
    assert report.checker_failures == ()
    assert report.exit_code() == 0


def test_validator_cli_main_prints_json_and_returns_exit_code(
    tmp_path: Path,
    capsys,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text(VALID_SOURCE, encoding="utf-8")

    exit_code = main([str(target)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["schema_version"] == "0.1"
    assert payload["parser_backend"] == "ast"
    assert payload["checker_failures"] == []
    assert len(payload["files"]) == 1
    assert payload["files"][0]["path"] == str(target.resolve())
    assert payload["files"][0]["diagnostics"] == []
    assert payload["files"][0]["checker_diagnostics"] == []
    assert payload["files"][0]["failures"] == []
