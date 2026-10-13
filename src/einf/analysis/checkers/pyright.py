import io
import os
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from einf.analysis.checkers.base import (
    CheckerAdapter,
    diagnostic_count_violation,
    diagnostic_field_violation,
    field_limit_failure,
    resolve_report_path,
)
from einf.analysis.checkers.json_limited import load_list_field_limited
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
    CheckerOutputLimits,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.model import DiagnosticSeverity, TextPosition, TextSpan

_Utf16Range = tuple[int, int, int, int]
_ParsedDiagnostic = tuple[CheckerDiagnostic, _Utf16Range | None]
_SOURCE_BYTE_LIMIT = 8 * 1024 * 1024
_SOURCE_READ_CHUNK_BYTES = 8192


@dataclass(frozen=True, slots=True)
class PyrightAdapter(CheckerAdapter):
    """Adapter for pyright-compatible JSON output."""

    name: str
    executable: str

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        return (
            self.executable,
            "--outputjson",
            *[str(path) for path in request.targets],
        )

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        if not stdout.strip():
            if stderr.strip():
                return CheckerResult(
                    diagnostics=(),
                    failures=(
                        CheckerFailure(
                            tool=self.name,
                            kind="execution_error",
                            message=stderr.strip(),
                        ),
                    ),
                )
            return CheckerResult(diagnostics=(), failures=())

        max_entries = limits.max_diagnostics if limits is not None else sys.maxsize
        entries, truncated, parse_error = load_list_field_limited(
            stdout,
            field="generalDiagnostics",
            max_entries=max_entries,
        )
        if parse_error is not None:
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message=f"pyright {parse_error}",
                    ),
                ),
            )
        if truncated:
            assert limits is not None
            return CheckerResult(
                diagnostics=(),
                failures=(
                    diagnostic_count_violation(
                        tool=self.name,
                        limits=limits,
                    ),
                ),
            )

        parsed_entries: list[_ParsedDiagnostic | CheckerFailure] = []
        for index, entry in enumerate(entries):
            parsed_entry = _parse_diagnostic_entry(
                entry=entry,
                index=index,
                tool=self.name,
                request=request,
                limits=limits,
            )
            if not isinstance(parsed_entry, CheckerFailure):
                diagnostic, _ = parsed_entry
                field_violation = (
                    diagnostic_field_violation(
                        tool=self.name,
                        limits=limits,
                        diagnostic=diagnostic,
                    )
                    if limits is not None
                    else None
                )
                if field_violation is not None:
                    return CheckerResult(
                        diagnostics=(),
                        failures=(field_violation,),
                    )
            elif parsed_entry.kind == "output_limit_exceeded":
                return CheckerResult(
                    diagnostics=(),
                    failures=(parsed_entry,),
                )
            parsed_entries.append(parsed_entry)

        return _normalize_spans(parsed_entries, tool=self.name)


def _parse_diagnostic_entry(
    *,
    entry: object,
    index: int,
    tool: str,
    request: CheckerRequest,
    limits: CheckerOutputLimits | None,
) -> _ParsedDiagnostic | CheckerFailure:
    if not isinstance(entry, dict):
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} must be a JSON object",
        )

    file_path = entry.get("file")
    if not isinstance(file_path, str) or not file_path:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid file path",
        )
    if limits is not None and len(file_path) > limits.max_field_length:
        return field_limit_failure(tool=tool, limits=limits)
    path = resolve_report_path(file_path, request)
    if path is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has an invalid file path",
        )

    message = entry.get("message")
    if not isinstance(message, str):
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid message",
        )

    severity = _severity(entry.get("severity"))
    if severity is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid severity",
        )

    coordinates = None
    if "range" in entry:
        coordinates = _parse_range(entry["range"])
        if coordinates is None:
            return CheckerFailure(
                tool=tool,
                kind="output_parse_error",
                message=f"{tool} diagnostic {index} has no valid range",
            )

    rule = entry.get("rule")
    if rule is not None and not isinstance(rule, str):
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has an invalid rule",
        )

    return (
        CheckerDiagnostic(
            tool=tool,
            path=path,
            code=rule,
            message=message,
            severity=severity,
            span=None,
        ),
        coordinates,
    )


def _severity(value: object) -> DiagnosticSeverity | None:
    if value == "error":
        return "error"
    if value == "warning":
        return "warning"
    if value == "information":
        return "info"
    return None


def _parse_range(value: object) -> _Utf16Range | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    start_line = start.get("line")
    start_character = start.get("character")
    end_line = end.get("line")
    end_character = end.get("character")
    if (
        type(start_line) is not int
        or type(start_character) is not int
        or type(end_line) is not int
        or type(end_character) is not int
        or min(start_line, start_character, end_line, end_character) < 0
        or (end_line, end_character) < (start_line, start_character)
    ):
        return None
    return start_line, start_character, end_line, end_character


def _normalize_spans(
    entries: list[_ParsedDiagnostic | CheckerFailure],
    *,
    tool: str,
) -> CheckerResult:
    requested_offsets: dict[Path, dict[int, set[int]]] = {}
    for entry in entries:
        if isinstance(entry, CheckerFailure):
            continue
        diagnostic, coordinates = entry
        if coordinates is not None:
            offsets = requested_offsets.setdefault(diagnostic.path, {})
            for line, offset in (
                (coordinates[0], coordinates[1]),
                (coordinates[2], coordinates[3]),
            ):
                offsets.setdefault(line, set()).add(offset)

    source_columns: dict[Path, dict[int, dict[int, int]] | CheckerFailure] = {}
    bytes_read = 0

    def read_chunk(descriptor: int) -> bytes:
        nonlocal bytes_read
        if bytes_read > _SOURCE_BYTE_LIMIT:
            raise OSError("source normalization exceeds the 8 MiB read budget")
        chunk = os.read(
            descriptor,
            min(_SOURCE_READ_CHUNK_BYTES, _SOURCE_BYTE_LIMIT - bytes_read + 1),
        )
        bytes_read += len(chunk)
        if bytes_read > _SOURCE_BYTE_LIMIT:
            raise OSError("source normalization exceeds the 8 MiB read budget")
        return chunk

    for path, offsets in requested_offsets.items():
        try:
            lines = _read_source_lines(path, set(offsets), read_chunk=read_chunk)
            source_columns[path] = {
                line: _character_columns(content, offsets[line])
                for line, content in lines.items()
            }
        except (OSError, UnicodeError) as error:
            source_columns[path] = CheckerFailure(
                tool=tool,
                kind="output_parse_error",
                message=f"{tool} cannot normalize diagnostic positions for {path}: {error}",
            )

    diagnostics: list[CheckerDiagnostic] = []
    failure: CheckerFailure | None = None
    for entry in entries:
        if isinstance(entry, CheckerFailure):
            if failure is None:
                failure = entry
            continue

        diagnostic, coordinates = entry
        if coordinates is not None:
            columns = source_columns[diagnostic.path]
            if isinstance(columns, CheckerFailure):
                if failure is None:
                    failure = columns
            else:
                span = _span_from_utf16_range(coordinates, columns)
                if span is None:
                    if failure is None:
                        failure = CheckerFailure(
                            tool=tool,
                            kind="output_parse_error",
                            message=(
                                f"{tool} diagnostic range is not a source-character "
                                f"boundary in {diagnostic.path}"
                            ),
                        )
                    continue
                diagnostic = replace(diagnostic, span=span)
        diagnostics.append(diagnostic)

    return CheckerResult(
        diagnostics=tuple(diagnostics),
        failures=() if failure is None else (failure,),
    )


def _read_source_lines(
    path: Path,
    requested_lines: set[int],
    *,
    read_chunk: Callable[[int], bytes],
) -> dict[int, str]:
    data = _read_regular_source(path, read_chunk=read_chunk)
    lines: dict[int, str] = {}
    last_requested_line = max(requested_lines)
    last_index = -1
    ends_with_newline = True
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8") as source:
        for last_index, line in enumerate(source):
            ends_with_newline = line.endswith("\n")
            if last_index in requested_lines:
                lines[last_index] = line.removesuffix("\n")
            if last_index >= last_requested_line:
                break
        else:
            # EOF after a newline (or in an empty file) is a valid empty line.
            if ends_with_newline and last_index + 1 in requested_lines:
                lines[last_index + 1] = ""
    return lines


def _read_regular_source(
    path: Path, *, read_chunk: Callable[[int], bytes]
) -> bytearray:
    if not stat.S_ISREG(path.stat().st_mode):
        raise OSError("diagnostic source is not a regular file")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0)
    )
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("diagnostic source is not a regular file")
        if metadata.st_size > _SOURCE_BYTE_LIMIT:
            raise OSError("diagnostic source exceeds the 8 MiB file-size limit")

        data = bytearray()
        while chunk := read_chunk(descriptor):
            data.extend(chunk)
        return data
    finally:
        os.close(descriptor)


def _span_from_utf16_range(
    coordinates: _Utf16Range,
    columns: dict[int, dict[int, int]],
) -> TextSpan | None:
    start_line, start_offset, end_line, end_offset = coordinates
    if start_line not in columns or end_line not in columns:
        return None

    start_column = columns[start_line].get(start_offset)
    end_column = columns[end_line].get(end_offset)
    if start_column is None or end_column is None:
        return None
    return TextSpan(
        start=TextPosition(line=start_line + 1, column=start_column),
        end=TextPosition(line=end_line + 1, column=end_column),
    )


def _character_columns(line: str, offsets: set[int]) -> dict[int, int]:
    if line.isascii():
        return {offset: offset for offset in offsets if offset <= len(line)}

    columns = {0: 0} if 0 in offsets else {}
    last_requested_offset = max(offsets)
    if last_requested_offset == 0:
        return columns

    units = 0
    for column, character in enumerate(line):
        units += 2 if ord(character) > 0xFFFF else 1
        if units in offsets:
            columns[units] = column + 1
        if units >= last_requested_offset:
            break
    return columns


__all__ = ["PyrightAdapter"]
