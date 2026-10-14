import shutil
from pathlib import Path

import pytest

from einf.analysis.checkers import (
    SUPPORTED_CHECKER_NAMES,
    CheckerAdapter,
    build_checker_adapters,
)
from einf.analysis.parser import AstParserBackend
from einf.analysis.validator.json_projection import project_validation_report
from einf.analysis.validator.run import run_validation

_PROJECT_CONFIG = """\
[tool.pyrefly]
preset = "default"
[tool.pyright]
typeCheckingMode = "standard"
reportImportCycles = "error"
[tool.basedpyright]
typeCheckingMode = "standard"
reportImportCycles = "error"
[tool.ty]
[tool.zuban]
mode = "default"
"""
_ASSIGNMENT_CODES = {
    "pyright": "reportAssignmentType",
    "basedpyright": "reportAssignmentType",
    "zuban": "assignment",
    "ty": "invalid-assignment",
    "pyrefly": "bad-assignment",
}


@pytest.fixture(params=SUPPORTED_CHECKER_NAMES)
def checker_adapter(request: pytest.FixtureRequest) -> CheckerAdapter:
    name = request.param
    assert isinstance(name, str)
    if shutil.which(name) is None:
        pytest.skip(f"{name} executable is not installed")

    return build_checker_adapters((name,))[0]


@pytest.fixture
def checker_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "checker project").resolve()
    root.mkdir()
    (root / "pyproject.toml").write_text(_PROJECT_CONFIG, encoding="utf-8")

    user_config = tmp_path / "user config"
    user_config.mkdir()
    for variable in ("HOME", "XDG_CONFIG_HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(variable, str(user_config))
    monkeypatch.delenv("TY_CONFIG_FILE", raising=False)

    return root


def test_real_checker_accepts_clean_source(
    checker_adapter: CheckerAdapter, checker_project: Path
) -> None:
    target = checker_project / "sample.py"
    target.write_text("result: int = 1\n", encoding="utf-8")

    report = run_validation(
        targets=(target,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker_adapter,),
    )

    assert report.exit_code() == 0
    assert report.checker_failures == ()
    assert report.discovery_failures == ()
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path == str(target)
    assert file_report.diagnostics == ()
    assert file_report.checker_diagnostics == ()
    assert file_report.failures == ()

    payload = project_validation_report(report)
    assert payload["checker_failures"] == []
    assert payload["files"][0]["checker_diagnostics"] == []


@pytest.mark.parametrize(
    "prefix", ("", 'tag = "\U0001f680"; '), ids=("ascii", "unicode")
)
def test_real_checker_preserves_assignment_diagnostic(
    checker_adapter: CheckerAdapter, checker_project: Path, prefix: str
) -> None:
    target = checker_project / "sample.py"
    source = prefix + 'result: int = "wrong"\n'
    target.write_text(source, encoding="utf-8")

    report = run_validation(
        targets=(target,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker_adapter,),
    )

    assert report.exit_code() == 1
    assert report.checker_failures == ()
    assert report.discovery_failures == ()
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path == str(target)
    assert file_report.diagnostics == ()
    assert file_report.failures == ()
    assert len(file_report.checker_diagnostics) == 1

    diagnostic = file_report.checker_diagnostics[0]
    assert diagnostic.tool == checker_adapter.name
    assert diagnostic.path == target
    assert diagnostic.code == _ASSIGNMENT_CODES[checker_adapter.name]
    assert diagnostic.severity == "error"
    assert diagnostic.message.strip()
    assert diagnostic.span is not None
    assert diagnostic.span.start.line == 1
    assert diagnostic.span.start.column == source.index('"wrong"')
    assert diagnostic.span.end.line == 1
    assert (
        diagnostic.span.start.column
        < diagnostic.span.end.column
        <= len(source.rstrip("\n"))
    )

    payload = project_validation_report(report)
    projected = payload["files"][0]["checker_diagnostics"][0]
    assert projected["tool"] == checker_adapter.name
    assert projected["path"] == str(target)
    assert projected["code"] == _ASSIGNMENT_CODES[checker_adapter.name]
    assert projected["severity"] == "error"
    assert projected["message"] == diagnostic.message
    assert projected["span"] is not None
    assert projected["span"]["start"] == {
        "line": 1,
        "column": source.index('"wrong"'),
    }


def test_real_checker_preserves_informational_diagnostic(
    checker_adapter: CheckerAdapter, checker_project: Path
) -> None:
    target = checker_project / "sample.py"
    target.write_text(
        "from typing import reveal_type\nreveal_type(1)\n", encoding="utf-8"
    )

    report = run_validation(
        targets=(target,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker_adapter,),
    )

    assert report.checker_failures == ()
    assert report.discovery_failures == ()
    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.diagnostics == ()
    assert file_report.failures == ()
    assert len(file_report.checker_diagnostics) == 1

    diagnostic = file_report.checker_diagnostics[0]
    assert diagnostic.tool == checker_adapter.name
    assert diagnostic.path == target
    assert diagnostic.severity == "info"
    assert diagnostic.message.strip()
    assert diagnostic.span is not None
    assert diagnostic.span.start.line == 2

    payload = project_validation_report(report)
    assert payload["files"][0]["checker_diagnostics"][0]["severity"] == "info"
    # The validator gates on diagnostic presence, not just error severity.
    assert report.exit_code() == 1


@pytest.mark.parametrize("checker_adapter", ("pyright", "basedpyright"), indirect=True)
def test_real_pyright_preserves_diagnostics_without_ranges(
    checker_adapter: CheckerAdapter, checker_project: Path
) -> None:
    first = checker_project / "first.py"
    second = checker_project / "second.py"
    first.write_text(
        "from second import SECOND\nFIRST = 1\nassert SECOND == 2\n", encoding="utf-8"
    )
    second.write_text(
        "from first import FIRST\nSECOND = 2\nassert FIRST == 1\n", encoding="utf-8"
    )

    report = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker_adapter,),
    )

    assert report.exit_code() == 1
    assert report.checker_failures == ()
    assert report.discovery_failures == ()
    assert len(report.files) == 2
    assert all(file.diagnostics == () and file.failures == () for file in report.files)
    diagnostics = tuple(
        diagnostic for file in report.files for diagnostic in file.checker_diagnostics
    )
    assert diagnostics
    for diagnostic in diagnostics:
        assert diagnostic.tool == checker_adapter.name
        assert diagnostic.path in (first, second)
        assert diagnostic.code == "reportImportCycles"
        assert diagnostic.severity == "error"
        assert diagnostic.span is None
        assert diagnostic.message.strip()

    payload = project_validation_report(report)
    projected = tuple(
        diagnostic
        for file in payload["files"]
        for diagnostic in file["checker_diagnostics"]
    )
    assert len(projected) == len(diagnostics)
    assert all(diagnostic["span"] is None for diagnostic in projected)
