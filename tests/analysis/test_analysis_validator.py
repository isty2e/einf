import errno
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from einf.analysis.parser import AstParserBackend
from einf.analysis.validator.cli import build_argument_parser, main
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


def _write_fake_basedpyright(
    *,
    bin_directory: Path,
    stdout: str,
    exit_code: int,
) -> None:
    bin_directory.mkdir()
    executable = bin_directory / "basedpyright"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"print({stdout!r})\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def _run_validator_subprocess(
    *,
    target: Path,
    checker_bin_directory: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = (
        f"{checker_bin_directory}{os.pathsep}{environment.get('PATH', '')}"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from einf.analysis.validator.cli import main; raise SystemExit(main())",
            "--checker",
            "basedpyright",
            str(target),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_run_validation_reports_semantic_diagnostics(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text(DIAGNOSTIC_SOURCE, encoding="utf-8")

    report = run_validation(targets=(target,), parser_backend=AstParserBackend())

    assert report.schema_version == "0.2"
    assert report.parser_backend == "ast"
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path == str(target.resolve())
    assert len(file_report.diagnostics) == 1
    assert file_report.checker_diagnostics == ()
    assert file_report.diagnostics[0].code == "ANALYSIS_AXIS_NOT_IN_INPUT"
    assert file_report.failures == ()
    assert report.checker_failures == ()
    assert report.discovery_failures == ()
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
    assert report.discovery_failures == ()
    assert report.exit_code() == 0


def test_run_validation_reports_traversal_failures_in_sorted_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    source = package_dir / "visible.py"
    source.write_text(VALID_SOURCE, encoding="utf-8")
    blocked_directories = (
        package_dir / "z_blocked",
        package_dir / "a_blocked",
    )
    for blocked_directory in blocked_directories:
        blocked_directory.mkdir()
        (blocked_directory / "hidden.py").write_text(VALID_SOURCE, encoding="utf-8")

    original_scandir = os.scandir
    blocked_paths = {path.resolve() for path in blocked_directories}

    def fail_blocked_scandir(
        path: str | os.PathLike[str],
    ) -> Iterator[os.DirEntry[str]]:
        if Path(path).resolve() in blocked_paths:
            raise PermissionError(
                errno.EACCES,
                "permission denied",
                path,
            )
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", fail_blocked_scandir)

    report = run_validation(targets=(package_dir,), parser_backend=AstParserBackend())

    assert tuple(file_report.path for file_report in report.files) == (
        str(source.resolve()),
    )
    assert tuple(failure.path for failure in report.discovery_failures) == (
        str((package_dir / "a_blocked").resolve()),
        str((package_dir / "z_blocked").resolve()),
    )
    assert all(
        failure.kind == "directory_traversal_error"
        for failure in report.discovery_failures
    )
    assert report.exit_code() == 1


def test_run_validation_follows_symlinked_directories_without_cycles(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "linked.py"
    source.write_text(VALID_SOURCE, encoding="utf-8")
    (package_dir / "linked").symlink_to(source_dir, target_is_directory=True)
    (source_dir / "back").symlink_to(package_dir, target_is_directory=True)

    report = run_validation(targets=(package_dir,), parser_backend=AstParserBackend())

    assert tuple(file_report.path for file_report in report.files) == (
        str(source.resolve()),
    )
    assert report.discovery_failures == ()
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
    assert payload["schema_version"] == "0.2"
    assert payload["parser_backend"] == "ast"
    assert payload["checker_failures"] == []
    assert payload["discovery_failures"] == []
    assert len(payload["files"]) == 1
    assert payload["files"][0]["path"] == str(target.resolve())
    assert payload["files"][0]["diagnostics"] == []
    assert payload["files"][0]["checker_diagnostics"] == []
    assert payload["files"][0]["failures"] == []


def test_validator_cli_parses_checker_execution_policy() -> None:
    arguments = build_argument_parser().parse_args(
        [
            "sample.py",
            "--checker",
            "basedpyright",
            "--checker-timeout-seconds",
            "2.5",
            "--checker-max-concurrency",
            "3",
        ]
    )

    assert arguments.checker_timeout_seconds == 2.5
    assert arguments.checker_max_concurrency == 3


def test_validator_cli_subprocess_serializes_mixed_diagnostics(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text(DIAGNOSTIC_SOURCE, encoding="utf-8")
    checker_bin_directory = tmp_path / "bin"
    checker_output = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(target.resolve()),
                    "severity": "warning",
                    "message": "fake checker diagnostic",
                    "rule": "fake-rule",
                    "range": {
                        "start": {"line": 1, "character": 2},
                        "end": {"line": 1, "character": 5},
                    },
                }
            ]
        }
    )
    _write_fake_basedpyright(
        bin_directory=checker_bin_directory,
        stdout=checker_output,
        exit_code=1,
    )

    process = _run_validator_subprocess(
        target=target,
        checker_bin_directory=checker_bin_directory,
    )
    payload = json.loads(process.stdout)

    assert process.returncode == 1
    assert payload["checker_failures"] == []
    assert len(payload["files"]) == 1
    file_payload = payload["files"][0]
    assert len(file_payload["diagnostics"]) == 1
    assert file_payload["checker_diagnostics"] == [
        {
            "code": "fake-rule",
            "message": "fake checker diagnostic",
            "path": str(target.resolve()),
            "severity": "warning",
            "span": {
                "end": {"column": 5, "line": 2},
                "start": {"column": 2, "line": 2},
            },
            "tool": "basedpyright",
        }
    ]


def test_validator_cli_subprocess_serializes_checker_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text(VALID_SOURCE, encoding="utf-8")
    checker_bin_directory = tmp_path / "bin"
    _write_fake_basedpyright(
        bin_directory=checker_bin_directory,
        stdout="not JSON",
        exit_code=2,
    )

    process = _run_validator_subprocess(
        target=target,
        checker_bin_directory=checker_bin_directory,
    )
    payload = json.loads(process.stdout)

    assert process.returncode == 1
    assert payload["checker_failures"][0]["tool"] == "basedpyright"
    assert payload["checker_failures"][0]["kind"] == "output_parse_error"
    assert payload["files"][0]["checker_diagnostics"] == []
