import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import einf.analysis.checkers.pyright as pyright_module
from einf.analysis.checkers.model import CheckerRequest, CheckerResult
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.model import TextPosition, TextSpan

_BYTE_LIMIT = 8 * 1024 * 1024


def _parse(targets: tuple[Path, ...]) -> CheckerResult:
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(target),
                    "severity": "error",
                    "message": f"diagnostic {index}",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                }
                for index, target in enumerate(targets)
            ]
        }
    )
    return PyrightAdapter(name="basedpyright", executable="basedpyright").parse_output(
        stdout=payload,
        stderr="",
        request=CheckerRequest(targets=targets, project_root=targets[0].parent),
    )


def _probe_special_source(target: Path) -> None:
    code = """import json
import sys
from pathlib import Path
from einf.analysis.checkers.model import CheckerRequest
from einf.analysis.checkers.pyright import PyrightAdapter
path = Path(sys.argv[1])
payload = json.dumps({"generalDiagnostics": [{
    "file": str(path), "severity": "error", "message": "source diagnostic",
    "range": {"start": {"line": 0, "character": 0},
              "end": {"line": 0, "character": 0}}
}]})
result = PyrightAdapter(name="pyright", executable="pyright").parse_output(
    stdout=payload, stderr="",
    request=CheckerRequest(targets=(path,), project_root=path.parent))
assert len(result.diagnostics) == 1
assert result.diagnostics[0].message == "source diagnostic"
assert result.diagnostics[0].span is None
assert result.failures[0].kind == "output_parse_error"
assert "regular file" in result.failures[0].message
"""
    subprocess.run(
        (sys.executable, "-c", code, str(target)),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_pyright_source_rejects_fifo_without_waiting_for_a_writer(
    tmp_path: Path,
) -> None:
    target = tmp_path / "source.py"
    os.mkfifo(target)

    _probe_special_source(target)


@pytest.mark.skipif(
    not Path("/dev/zero").exists(), reason="character device unavailable"
)
def test_pyright_source_rejects_character_device() -> None:
    _probe_special_source(Path("/dev/zero"))


def test_pyright_source_rejects_oversized_newline_free_file_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    with target.open("wb") as source:
        source.truncate(_BYTE_LIMIT + 1)

    def denied_read(descriptor: int, count: int) -> bytes:
        raise AssertionError("an oversized source must not be read")

    monkeypatch.setattr(os, "read", denied_read)

    result = _parse((target,))

    assert result.diagnostics[0].span is None
    assert result.failures[0].kind == "output_parse_error"
    assert "file-size limit" in result.failures[0].message


def test_pyright_source_accepts_exact_byte_limit_and_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    target.write_bytes(b"x" * _BYTE_LIMIT)
    original_read = os.read
    counts: list[int] = []

    def tracked_read(descriptor: int, count: int) -> bytes:
        assert 0 < count <= 8192
        chunk = original_read(descriptor, count)
        counts.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", tracked_read)

    result = _parse((target,))

    assert result.failures == ()
    assert result.diagnostics[0].span == TextSpan(
        TextPosition(1, 0), TextPosition(1, 0)
    )
    assert sum(counts) == _BYTE_LIMIT


def test_pyright_source_budget_is_shared_and_stops_after_one_overrun_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, third = (tmp_path / f"source_{index}.py" for index in range(3))
    first.write_bytes(b"x" * (_BYTE_LIMIT // 2 + 1))
    second.write_bytes(b"x" * (_BYTE_LIMIT // 2 + 1))
    third.write_bytes(b"x")
    original_read = os.read
    counts: list[int] = []

    def tracked_read(descriptor: int, count: int) -> bytes:
        assert 0 < count <= 8192
        chunk = original_read(descriptor, count)
        counts.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", tracked_read)

    result = _parse((first, second, third))

    assert sum(counts) == _BYTE_LIMIT + 1
    assert result.diagnostics[0].span is not None
    assert result.diagnostics[1].span is None
    assert result.diagnostics[2].span is None
    assert result.failures[0].kind == "output_parse_error"
    assert "read budget" in result.failures[0].message


def test_pyright_source_decode_failure_does_not_reset_byte_budget(
    tmp_path: Path,
) -> None:
    first = tmp_path / "invalid.py"
    second = tmp_path / "valid.py"
    first.write_bytes(b"\xff" + b"x" * (_BYTE_LIMIT // 2))
    second.write_bytes(b"x" * (_BYTE_LIMIT // 2))

    result = _parse((first, second))

    assert [diagnostic.span for diagnostic in result.diagnostics] == [None, None]
    assert result.failures[0].kind == "output_parse_error"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_pyright_source_revalidates_descriptor_when_file_is_replaced_by_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    fifo = tmp_path / "fifo"
    target.write_text("x", encoding="utf-8")
    os.mkfifo(fifo)
    original_open = os.open
    descriptors: list[int] = []

    def redirected_open(path: Path, flags: int) -> int:
        assert flags & os.O_NONBLOCK
        descriptor = original_open(fifo, flags)
        descriptors.append(descriptor)
        return descriptor

    def denied_read(descriptor: int, count: int) -> bytes:
        raise AssertionError("the opened FIFO must not be read")

    monkeypatch.setattr(os, "open", redirected_open)
    monkeypatch.setattr(os, "read", denied_read)

    result = _parse((target,))

    assert result.diagnostics[0].span is None
    assert "regular file" in result.failures[0].message
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_pyright_source_bounds_growth_after_descriptor_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    target.write_bytes(b"x")
    original_fstat = os.fstat
    original_read = os.read
    descriptors: list[int] = []
    counts: list[int] = []

    def grow_after_stat(descriptor: int) -> os.stat_result:
        metadata = original_fstat(descriptor)
        descriptors.append(descriptor)
        with target.open("ab") as source:
            source.write(b"x" * _BYTE_LIMIT)
        return metadata

    def tracked_read(descriptor: int, count: int) -> bytes:
        assert 0 < count <= 8192
        chunk = original_read(descriptor, count)
        counts.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    monkeypatch.setattr(os, "read", tracked_read)

    result = _parse((target,))

    assert sum(counts) == _BYTE_LIMIT + 1
    assert result.diagnostics[0].span is None
    assert "read budget" in result.failures[0].message
    with pytest.raises(OSError):
        original_fstat(descriptors[0])


def test_pyright_source_closes_descriptor_on_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    target.write_bytes(b"x")
    descriptors: list[int] = []

    def failed_read(descriptor: int, count: int) -> bytes:
        descriptors.append(descriptor)
        raise OSError("read failed")

    monkeypatch.setattr(os, "read", failed_read)

    result = _parse((target,))

    assert result.diagnostics[0].span is None
    assert "read failed" in result.failures[0].message
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("content", (b"x", b"\xff"))
def test_pyright_source_closes_descriptor_on_success_or_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    target = tmp_path / "source.py"
    target.write_bytes(content)
    original_open = os.open
    descriptors: list[int] = []

    def tracked_open(path: Path, flags: int) -> int:
        descriptor = original_open(path, flags)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", tracked_open)

    result = _parse((target,))

    if content == b"x":
        assert result.failures == ()
        assert result.diagnostics[0].span is not None
    else:
        assert result.failures[0].kind == "output_parse_error"
        assert result.diagnostics[0].span is None
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_pyright_source_resolves_all_offsets_in_one_unicode_line_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    source = "\U0001f600" + "x" * 1000
    target.write_text(source, encoding="utf-8")
    inspections = 0

    def tracked_ord(character: str) -> int:
        nonlocal inspections
        inspections += 1
        return ord(character)

    monkeypatch.setattr(pyright_module, "ord", tracked_ord, raising=False)
    records = [
        {
            "file": str(target),
            "severity": "error",
            "message": f"diagnostic {offset}",
            "range": {
                "start": {"line": 0, "character": offset},
                "end": {"line": 0, "character": 1002},
            },
        }
        for offset in range(2, 1002, 10)
    ]
    result = PyrightAdapter(
        name="basedpyright", executable="basedpyright"
    ).parse_output(
        stdout=json.dumps({"generalDiagnostics": records}),
        stderr="",
        request=CheckerRequest(targets=(target,), project_root=tmp_path),
    )

    assert result.failures == ()
    assert inspections == len(source)
    assert [diagnostic.span for diagnostic in result.diagnostics] == [
        TextSpan(TextPosition(1, offset - 1), TextPosition(1, len(source)))
        for offset in range(2, 1002, 10)
    ]
