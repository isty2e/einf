import errno
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from einf.analysis.checkers import CheckerAdapter, CheckerRequest, CheckerResult
from einf.analysis.parser import AstParserBackend, ParserUnavailableError
from einf.analysis.validator import (
    ValidationLimitFailure,
    ValidationLimits,
    run_validation,
)
from einf.analysis.validator.cli import build_argument_parser, main


class _RecordingChecker(CheckerAdapter):
    name = "probe"
    executable = sys.executable

    def __init__(self) -> None:
        self.requests: list[CheckerRequest] = []

    def build_command(self, request: CheckerRequest, /) -> tuple[str, ...]:
        self.requests.append(request)
        return (self.executable, "-c", "pass")

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        return CheckerResult(diagnostics=(), failures=())


@pytest.mark.parametrize(
    "field", ["max_discovery_entries", "max_file_bytes", "max_total_bytes"]
)
@pytest.mark.parametrize("value", [True, False, 1.5, "3"])
def test_limits_reject_non_integer_values(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        # Deliberately violate annotations to exercise the runtime boundary.
        replace(ValidationLimits(), **{field: cast(int, value)})


@pytest.mark.parametrize(
    "field", ["max_discovery_entries", "max_file_bytes", "max_total_bytes"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_limits_reject_nonpositive_values(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        replace(ValidationLimits(), **{field: value})


def test_unrestricted_configuration_preserves_normal_validation(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("# unrestricted\n", encoding="utf-8")
    expected = run_validation(targets=(source,), parser_backend=AstParserBackend())
    actual = run_validation(
        targets=(source,), parser_backend=AstParserBackend(), limits=ValidationLimits()
    )
    assert actual == expected
    assert actual.exit_code() == 0
    assert actual.limit_failures == ()


def test_limits_accept_positive_integer_subclasses() -> None:
    class PositiveInteger(int):
        pass

    limits = ValidationLimits(max_file_bytes=PositiveInteger(3))
    assert limits.max_file_bytes == 3


def test_discovery_limit_accepts_exact_boundary_and_rejects_next_entry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.py"
    source.write_text("#x\n", encoding="utf-8")
    accepted = run_validation(
        targets=(tmp_path,),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_discovery_entries=2),
    )
    assert accepted.exit_code() == 0
    assert len(accepted.files) == 1

    rejected = run_validation(
        targets=(tmp_path,),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_discovery_entries=1),
    )
    assert rejected.files == ()
    assert rejected.limit_failures == (
        ValidationLimitFailure("discovery_entries", 1, 2, str(tmp_path)),
    )
    assert rejected.exit_code() == 1


@pytest.mark.parametrize("directories", [False, True])
def test_discovery_limit_counts_non_python_entries_and_stops_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directories: bool
) -> None:
    for index in range(12):
        path = tmp_path / f"entry-{index}.txt"
        if directories:
            path.mkdir()
        else:
            path.touch()
    original = os.scandir
    visited: list[str] = []

    @contextmanager
    def counted_scandir(
        path: str | os.PathLike[str],
    ) -> Iterator[Iterator[os.DirEntry[str]]]:
        with original(path) as entries:

            def counted_entries() -> Iterator[os.DirEntry[str]]:
                for entry in entries:
                    visited.append(entry.name)
                    yield entry

            yield counted_entries()

    monkeypatch.setattr(os, "scandir", counted_scandir)
    report = run_validation(
        targets=(tmp_path,),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_discovery_entries=4),
    )
    assert len(visited) == 4
    assert report.files == ()
    assert report.limit_failures == (
        ValidationLimitFailure("discovery_entries", 4, 5, str(tmp_path)),
    )


def test_discovery_failure_does_not_depend_on_scandir_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("c.py", "b.py", "a.py"):
        (tmp_path / name).write_text("#x\n", encoding="utf-8")
    limits = ValidationLimits(max_discovery_entries=2)
    expected = run_validation(
        targets=(tmp_path,), parser_backend=AstParserBackend(), limits=limits
    )
    original = os.scandir

    @contextmanager
    def reversed_scandir(
        path: str | os.PathLike[str],
    ) -> Iterator[Iterator[os.DirEntry[str]]]:
        with original(path) as entries:
            yield iter(reversed(tuple(entries)))

    monkeypatch.setattr(os, "scandir", reversed_scandir)
    actual = run_validation(
        targets=(tmp_path,), parser_backend=AstParserBackend(), limits=limits
    )
    assert actual == expected


def test_discovery_budget_is_shared_across_explicit_targets(tmp_path: Path) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.touch()
    second.touch()
    report = run_validation(
        targets=(first, first, second),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_discovery_entries=2),
    )
    assert report.files == ()
    assert report.limit_failures == (
        ValidationLimitFailure("discovery_entries", 2, 3, str(second)),
    )


def test_discovery_budget_is_shared_across_roots_and_prevents_checker_launch(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        root.mkdir()
        (root / "sample.py").touch()
    checker = _RecordingChecker()
    accepted = run_validation(
        targets=roots,
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_discovery_entries=4),
    )
    assert accepted.exit_code() == 0
    assert len(checker.requests) == 1
    checker.requests.clear()

    rejected = run_validation(
        targets=roots,
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_discovery_entries=3),
    )
    assert rejected.files == ()
    assert rejected.limit_failures == (
        ValidationLimitFailure("discovery_entries", 3, 4, str(roots[1])),
    )
    assert checker.requests == []


@pytest.mark.parametrize("resource", ["file_bytes", "total_bytes"])
def test_source_limits_accept_exact_bytes_and_preflight_larger_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    source = tmp_path / "sample.py"
    source.write_bytes(b"#x\n")
    field = "max_file_bytes" if resource == "file_bytes" else "max_total_bytes"
    accepted = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        limits=replace(ValidationLimits(), **{field: 3}),
    )
    assert accepted.exit_code() == 0

    def unexpected_read(descriptor: int, count: int) -> bytes:
        raise AssertionError("oversize metadata must be rejected before reading")

    monkeypatch.setattr(os, "read", unexpected_read)
    rejected = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        limits=replace(ValidationLimits(), **{field: 2}),
    )
    assert rejected.exit_code() == 1
    assert rejected.files == ()
    assert len(rejected.limit_failures) == 1
    failure = rejected.limit_failures[0]
    assert (failure.resource, failure.limit, failure.observed, failure.path) == (
        resource,
        2,
        3,
        str(source),
    )


def test_total_byte_limit_retains_completed_reports_but_never_runs_checkers(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_bytes(b"#a\n")
    second.write_bytes(b"#b\n")
    checker = _RecordingChecker()
    accepted = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_file_bytes=3, max_total_bytes=6),
    )
    assert accepted.exit_code() == 0
    assert len(checker.requests) == 1
    assert checker.requests[0].targets == (first, second)
    checker.requests.clear()

    rejected = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_file_bytes=3, max_total_bytes=5),
    )
    assert tuple(file.path for file in rejected.files) == (str(first),)
    assert rejected.limit_failures == (
        ValidationLimitFailure("total_bytes", 5, 6, str(second)),
    )
    assert rejected.exit_code() == 1
    assert checker.requests == []


def test_total_byte_budget_accumulates_all_chunks(tmp_path: Path) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    payload = b"#" + b"x" * 65536 + b"\n"
    first.write_bytes(payload)
    second.write_bytes(b"#x\n")
    report = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_total_bytes=len(payload) + 2),
    )
    assert tuple(file.path for file in report.files) == (str(first),)
    assert report.files[0].failures == ()
    assert report.limit_failures == (
        ValidationLimitFailure(
            "total_bytes", len(payload) + 2, len(payload) + 3, str(second)
        ),
    )


def test_duplicate_and_symlinked_sources_are_read_once(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = outside / "sample.py"
    source.write_bytes(b"#x\n")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (outside / "back").symlink_to(root, target_is_directory=True)
    alias = root / "alias.py"
    alias.symlink_to(source)
    report = run_validation(
        targets=(root, outside, source, alias),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(
            max_discovery_entries=30, max_file_bytes=3, max_total_bytes=3
        ),
    )
    assert report.exit_code() == 0
    assert tuple(file.path for file in report.files) == (str(source),)


def test_total_byte_budget_charges_undecodable_source(tmp_path: Path) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_bytes(b"\xff")
    second.write_bytes(b"#x\n")
    report = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_total_bytes=3),
    )
    assert report.files[0].failures[0].kind == "read_error"
    assert report.limit_failures == (
        ValidationLimitFailure("total_bytes", 3, 4, str(second)),
    )


def test_total_byte_budget_charges_bytes_before_partial_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_bytes(b"#\n")
    second.write_bytes(b"#x\n")
    original = os.read
    reads: list[int] = []

    def fail_second_read(descriptor: int, count: int) -> bytes:
        reads.append(count)
        if len(reads) == 2:
            raise OSError(errno.EIO, "injected read failure")
        return original(descriptor, min(count, 1))

    monkeypatch.setattr(os, "read", fail_second_read)
    report = run_validation(
        targets=(first, second),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_total_bytes=3),
    )
    assert report.files[0].failures[0].kind == "read_error"
    assert report.limit_failures == (
        ValidationLimitFailure("total_bytes", 3, 4, str(second)),
    )


@pytest.mark.parametrize("resource", ["file_bytes", "total_bytes"])
def test_source_growth_is_bounded_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: Literal["file_bytes", "total_bytes"],
) -> None:
    source = tmp_path / "sample.py"
    source.write_bytes(b"#\n")
    original_read, original_stat = os.read, os.fstat
    descriptors: list[int] = []
    reads: list[int] = []

    def grow_on_read(descriptor: int, count: int) -> bytes:
        if not reads:
            with source.open("ab") as stream:
                stream.write(b"#growth\n")
        reads.append(count)
        descriptors.append(descriptor)
        return original_read(descriptor, count)

    monkeypatch.setattr(os, "read", grow_on_read)
    report = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        limits=(
            ValidationLimits(max_file_bytes=3, max_total_bytes=5)
            if resource == "file_bytes"
            else ValidationLimits(max_file_bytes=5, max_total_bytes=3)
        ),
    )
    assert reads == [4]
    assert report.limit_failures == (
        ValidationLimitFailure(resource, 3, 4, str(source)),
    )
    with pytest.raises(OSError) as closed:
        original_stat(descriptors[0])
    assert closed.value.errno == errno.EBADF


def test_descriptor_cleanup_cannot_replace_limit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.py"
    source.write_bytes(b"#x\n")
    original_close = os.close
    closed: list[int] = []
    checker = _RecordingChecker()

    def close_with_error(descriptor: int) -> None:
        original_close(descriptor)
        if not closed:
            closed.append(descriptor)
            raise OSError(errno.EIO, "injected close failure")

    monkeypatch.setattr(os, "close", close_with_error)
    report = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_file_bytes=2),
    )
    assert closed
    assert report.limit_failures == (
        ValidationLimitFailure("file_bytes", 2, 3, str(source)),
    )
    assert checker.requests == []


def test_byte_capture_preserves_utf8_and_universal_newlines(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    payload = (
        "# \u03c0\r\nfrom einf import ax, axes, reduce\r\n"
        'b, n, z = axes("b", "n", "z")\r\n'
        "reduce(ax[b, n], ax[b, z])\r"
    ).encode()
    source.write_bytes(payload)
    expected = run_validation(targets=(source,), parser_backend=AstParserBackend())
    actual = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(
            max_file_bytes=len(payload), max_total_bytes=len(payload)
        ),
    )
    assert len(actual.files[0].diagnostics) == 1
    assert actual == expected


@pytest.mark.parametrize("field", ["max_file_bytes", "max_total_bytes"])
def test_source_limits_count_utf8_bytes_not_characters(
    tmp_path: Path, field: str
) -> None:
    source = tmp_path / "sample.py"
    source.write_bytes("#\u03c0\n".encode())
    report = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        limits=replace(ValidationLimits(), **{field: 3}),
    )
    assert report.files == ()
    assert report.limit_failures == (
        ValidationLimitFailure(
            "file_bytes" if field == "max_file_bytes" else "total_bytes",
            3,
            4,
            str(source),
        ),
    )


def test_total_byte_budget_allows_empty_source_after_exact_exhaustion(
    tmp_path: Path,
) -> None:
    first, empty = tmp_path / "a.py", tmp_path / "empty.py"
    first.write_bytes(b"#x\n")
    empty.touch()
    report = run_validation(
        targets=(first, empty),
        parser_backend=AstParserBackend(),
        limits=ValidationLimits(max_total_bytes=3),
    )
    assert report.exit_code() == 0
    assert len(report.files) == 2


def test_missing_parser_cannot_bypass_byte_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.py"
    source.write_bytes(b"#x\n")
    checker = _RecordingChecker()

    def unavailable(self: AstParserBackend) -> None:
        raise ParserUnavailableError(
            backend="ast", message="injected unavailable parser"
        )

    monkeypatch.setattr(AstParserBackend, "validate_available", unavailable)
    rejected = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_file_bytes=2),
    )
    assert rejected.limit_failures == (
        ValidationLimitFailure("file_bytes", 2, 3, str(source)),
    )
    assert checker.requests == []

    allowed = run_validation(
        targets=(source,),
        parser_backend=AstParserBackend(),
        checker_adapters=(checker,),
        limits=ValidationLimits(max_file_bytes=3),
    )
    assert allowed.limit_failures == ()
    assert allowed.files[0].failures[0].kind == "parser_unavailable"
    assert len(checker.requests) == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_bounded_source_capture_rejects_fifo_without_waiting_for_writer(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "pipe.py"
    os.mkfifo(fifo)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from einf.analysis.validator.cli import main; "
                "sys.exit(main(sys.argv[1:]))"
            ),
            str(fifo),
            "--max-file-bytes",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["limit_failures"] == []
    assert payload["files"][0]["failures"][0]["kind"] == "read_error"


@pytest.mark.parametrize(
    ("flag", "resource", "maximum", "observed"),
    [
        ("--max-file-bytes", "file_bytes", 2, 3),
        ("--max-total-bytes", "total_bytes", 2, 3),
        ("--max-discovery-entries", "discovery_entries", 1, 2),
    ],
)
def test_cli_projects_limit_failure_and_defaults_are_unrestricted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    resource: str,
    maximum: int,
    observed: int,
) -> None:
    defaults = build_argument_parser().parse_args(["sample.py"])
    assert defaults.max_discovery_entries is None
    assert defaults.max_file_bytes is None
    assert defaults.max_total_bytes is None
    source = tmp_path / "sample.py"
    source.write_bytes(b"#x\n")
    assert main([str(source), str(source), flag, str(maximum)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.4"
    assert payload["limit_failures"] == [
        {
            "resource": resource,
            "limit": maximum,
            "observed": observed,
            "path": str(source),
        }
    ]


@pytest.mark.parametrize(
    "flag", ["--max-discovery-entries", "--max-file-bytes", "--max-total-bytes"]
)
@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_limit_values(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args(["sample.py", flag, value])
    assert error.value.code == 2
