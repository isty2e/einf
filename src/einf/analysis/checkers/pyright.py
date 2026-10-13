import sys
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
    requested_lines: dict[Path, set[int]] = {}
    for entry in entries:
        if isinstance(entry, CheckerFailure):
            continue
        diagnostic, coordinates = entry
        if coordinates is not None:
            lines = requested_lines.setdefault(diagnostic.path, set())
            lines.update((coordinates[0], coordinates[2]))

    source_lines: dict[Path, dict[int, str] | CheckerFailure] = {}
    for path, lines in requested_lines.items():
        try:
            source_lines[path] = _read_source_lines(path, lines)
        except (OSError, UnicodeError) as error:
            source_lines[path] = CheckerFailure(
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
            lines = source_lines[diagnostic.path]
            if isinstance(lines, CheckerFailure):
                if failure is None:
                    failure = lines
            else:
                span = _span_from_utf16_range(coordinates, lines)
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


def _read_source_lines(path: Path, requested_lines: set[int]) -> dict[int, str]:
    lines: dict[int, str] = {}
    last_requested_line = max(requested_lines)
    last_index = -1
    ends_with_newline = True
    with path.open(encoding="utf-8") as source:
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


def _span_from_utf16_range(
    coordinates: _Utf16Range,
    lines: dict[int, str],
) -> TextSpan | None:
    start_line, start_offset, end_line, end_offset = coordinates
    if start_line not in lines or end_line not in lines:
        return None

    start_column = _character_column(lines[start_line], start_offset)
    end_column = _character_column(lines[end_line], end_offset)
    if start_column is None or end_column is None:
        return None
    return TextSpan(
        start=TextPosition(line=start_line + 1, column=start_column),
        end=TextPosition(line=end_line + 1, column=end_column),
    )


def _character_column(line: str, offset: int) -> int | None:
    units = 0
    for column, character in enumerate(line):
        if units == offset:
            return column
        units += 2 if ord(character) > 0xFFFF else 1
        if units > offset:
            return None
    return len(line) if units == offset else None


__all__ = ["PyrightAdapter"]
