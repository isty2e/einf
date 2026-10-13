import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from einf.analysis.checkers.execution import CheckerExecutionPolicy, CheckerExecutor
from einf.analysis.checkers.model import (
    CheckerOutputLimits,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import AstParserBackend
from einf.analysis.validator.json_projection import project_validation_report
from einf.analysis.validator.run import run_validation


def _payload(path: Path, coordinates: tuple[tuple[int, int, int, int], ...]) -> str:
    return json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(path),
                    "severity": "error",
                    "message": f"diagnostic {index}",
                    "rule": "reportUndefinedVariable",
                    "range": {
                        "start": {"line": start_line, "character": start_column},
                        "end": {"line": end_line, "character": end_column},
                    },
                }
                for index, (
                    start_line,
                    start_column,
                    end_line,
                    end_column,
                ) in enumerate(coordinates)
            ]
        }
    )


def _parse(path: Path, stdout: str, *, name: str = "basedpyright") -> CheckerResult:
    return PyrightAdapter(name=name, executable=name).parse_output(
        stdout=stdout,
        stderr="",
        request=CheckerRequest(targets=(path,), project_root=path.parent),
    )


@pytest.mark.parametrize("name", ("pyright", "basedpyright"))
@pytest.mark.parametrize(
    "prefix",
    (
        '"ascii"; ',
        '"\u754c\ud55c"; ',
        '"e\u0301"; ',
        '"\U0001f600"; ',
        '"\U00010400\U00010401"; ',
        '\ufeff"\U0001f600"; ',
        '"\U0001f600\u2028\u2029\x85\x0b\x0c"; ',
    ),
)
def test_pyright_positions_select_source_characters(
    tmp_path: Path, name: str, prefix: str
) -> None:
    target = tmp_path / "sample.py"
    token = "missing_symbol"
    target.write_text(prefix + token + "\n", encoding="utf-8")
    wire_start = len(prefix.encode("utf-16-le")) // 2
    stdout = _payload(target, ((0, wire_start, 0, wire_start + len(token)),))

    result = _parse(target, stdout, name=name)

    assert result.failures == ()
    assert result.diagnostics[0].span == TextSpan(
        start=TextPosition(1, len(prefix)),
        end=TextPosition(1, len(prefix) + len(token)),
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n", "\r"))
def test_pyright_positions_convert_both_multiline_endpoints(
    tmp_path: Path, newline: str
) -> None:
    target = tmp_path / "sample.py"
    target.write_bytes(
        f"\U0001f600abc{newline}\U00010400\U00010401xyz{newline}".encode()
    )

    result = _parse(target, _payload(target, ((0, 2, 1, 6),)))

    assert result.failures == ()
    assert result.diagnostics[0].span == TextSpan(
        TextPosition(1, 1), TextPosition(2, 4)
    )


@pytest.mark.parametrize(
    ("source", "coordinates", "position"),
    (
        ("", (0, 0, 0, 0), TextPosition(1, 0)),
        ("\U0001f600", (0, 2, 0, 2), TextPosition(1, 1)),
        ("\U0001f600\n", (1, 0, 1, 0), TextPosition(2, 0)),
        ("\n\n", (2, 0, 2, 0), TextPosition(3, 0)),
    ),
)
def test_pyright_positions_preserve_empty_ranges_at_eof(
    tmp_path: Path,
    source: str,
    coordinates: tuple[int, int, int, int],
    position: TextPosition,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text(source, encoding="utf-8")

    result = _parse(target, _payload(target, (coordinates,)))

    assert result.failures == ()
    assert result.diagnostics[0].span == TextSpan(position, position)


@pytest.mark.parametrize(
    "coordinates",
    (
        (0, 1, 0, 2),
        (0, 0, 0, 1),
        (0, 2, 0, 4),
        (2, 0, 2, 0),
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 2, 0, 0),
        (1, 0, 0, 0),
        (True, 0, 0, 0),
        (0, 0.0, 0, 0),
    ),
)
def test_pyright_positions_reject_invalid_boundaries(
    tmp_path: Path, coordinates: tuple[int, int | float, int, int]
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("\U0001f600x\n", encoding="utf-8")
    start_line, start_column, end_line, end_column = coordinates
    stdout = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(target),
                    "severity": "error",
                    "message": "bad coordinate",
                    "range": {
                        "start": {"line": start_line, "character": start_column},
                        "end": {"line": end_line, "character": end_column},
                    },
                }
            ]
        }
    )

    result = _parse(target, stdout)

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"


@pytest.mark.parametrize("source", (None, b"\xff"))
def test_pyright_positions_preserve_diagnostics_when_source_is_unavailable(
    tmp_path: Path, source: bytes | None
) -> None:
    target = tmp_path / "sample.py"
    if source is not None:
        target.write_bytes(source)

    result = _parse(target, _payload(target, ((0, 0, 0, 1),)))

    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].message == "diagnostic 0"
    assert result.diagnostics[0].severity == "error"
    assert result.diagnostics[0].code == "reportUndefinedVariable"
    assert result.diagnostics[0].span is None
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_parse_error"


def test_pyright_positions_read_each_file_once_and_preserve_valid_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("\U0001f600abc\n\U00010400xyz\n", encoding="utf-8")
    reads: list[Path] = []
    original_open = os.open

    def tracked_open(path: Path, flags: int) -> int:
        reads.append(path)
        return original_open(path, flags)

    monkeypatch.setattr(os, "open", tracked_open)

    result = _parse(
        target, _payload(target, ((0, 2, 0, 5), (0, 1, 0, 2), (1, 2, 1, 5)))
    )

    assert reads == [target]
    assert [diagnostic.message for diagnostic in result.diagnostics] == [
        "diagnostic 0",
        "diagnostic 2",
    ]
    assert [diagnostic.span for diagnostic in result.diagnostics] == [
        TextSpan(TextPosition(1, 1), TextPosition(1, 4)),
        TextSpan(TextPosition(2, 1), TextPosition(2, 4)),
    ]
    assert len(result.failures) == 1


def test_pyright_positions_preserve_range_less_diagnostics_without_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "missing.py"
    payload = json.loads(_payload(target, ((0, 0, 0, 1),)))
    del payload["generalDiagnostics"][0]["range"]

    def denied_open(path: Path, flags: int) -> int:
        raise AssertionError("a range-less diagnostic does not need source text")

    monkeypatch.setattr(os, "open", denied_open)

    result = _parse(target, json.dumps(payload))

    assert result.failures == ()
    assert result.diagnostics[0].message == "diagnostic 0"
    assert result.diagnostics[0].span is None


def test_pyright_positions_use_each_diagnostics_own_source_file(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("\U0001f600ab\n", encoding="utf-8")
    second.write_text("\u6f22ab\n", encoding="utf-8")
    payload = json.loads(_payload(first, ((0, 2, 0, 4),)))
    payload["generalDiagnostics"].extend(
        json.loads(_payload(second, ((0, 2, 0, 3),)))["generalDiagnostics"]
    )

    result = _parse(first, json.dumps(payload))

    assert result.failures == ()
    assert [(item.path, item.span) for item in result.diagnostics] == [
        (first, TextSpan(TextPosition(1, 1), TextPosition(1, 3))),
        (second, TextSpan(TextPosition(1, 2), TextPosition(1, 3))),
    ]


def test_pyright_positions_report_permission_failure_without_losing_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("x\n", encoding="utf-8")

    def denied_open(path: Path, flags: int) -> int:
        raise PermissionError("source access denied")

    monkeypatch.setattr(os, "open", denied_open)

    result = _parse(target, _payload(target, ((0, 0, 0, 1),)))

    assert result.diagnostics[0].span is None
    assert result.diagnostics[0].message == "diagnostic 0"
    assert result.failures[0].kind == "output_parse_error"
    assert "source access denied" in result.failures[0].message


def test_pyright_positions_enforce_field_limits_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sample.py"
    payload = json.loads(_payload(target, ((0, 0, 0, 1), (0, 0, 0, 1))))
    payload["generalDiagnostics"][1]["message"] = "x" * 4097

    def denied_open(path: Path, flags: int) -> int:
        raise AssertionError("field limits must precede source I/O")

    monkeypatch.setattr(os, "open", denied_open)

    result = PyrightAdapter(
        name="basedpyright", executable="basedpyright"
    ).parse_output(
        stdout=json.dumps(payload),
        stderr="",
        request=CheckerRequest(targets=(target,), project_root=tmp_path),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=4096),
    )

    assert result.diagnostics == ()
    assert result.failures[0].kind == "output_limit_exceeded"


@pytest.mark.skipif(
    shutil.which("basedpyright") is None, reason="basedpyright not installed"
)
def test_real_basedpyright_positions_reach_validator_json(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    prefix = '"\u754c\U0001f600e\u0301"; '
    target.write_text(prefix + "missing_symbol\n", encoding="utf-8")
    adapter = PyrightAdapter(name="basedpyright", executable="basedpyright")
    request = CheckerRequest(targets=(target,), project_root=tmp_path)

    async def execute() -> CheckerResult:
        return await CheckerExecutor(CheckerExecutionPolicy()).run(adapter, request)

    checked = asyncio.run(execute())
    diagnostic = next(
        item for item in checked.diagnostics if item.code == "reportUndefinedVariable"
    )
    assert checked.failures == ()
    assert diagnostic.span == TextSpan(
        TextPosition(1, len(prefix)),
        TextPosition(1, len(prefix) + len("missing_symbol")),
    )

    report = run_validation(
        targets=(target,),
        parser_backend=AstParserBackend(),
        checker_adapters=(adapter,),
    )
    assert report.checker_failures == ()
    assert diagnostic in report.files[0].checker_diagnostics
    projected = project_validation_report(report)
    serialized = next(
        item
        for item in projected["files"][0]["checker_diagnostics"]
        if item["code"] == "reportUndefinedVariable"
    )
    assert serialized["span"] == {
        "start": {"line": 1, "column": len(prefix)},
        "end": {"line": 1, "column": len(prefix) + len("missing_symbol")},
    }
