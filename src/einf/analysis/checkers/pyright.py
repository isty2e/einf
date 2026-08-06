import sys
from dataclasses import dataclass

from einf.analysis.checkers.base import (
    CheckerAdapter,
    diagnostic_count_violation,
    diagnostic_field_violation,
    field_limit_failure,
    line_span,
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
from einf.analysis.model import DiagnosticSeverity, TextSpan


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

        diagnostics: list[CheckerDiagnostic] = []
        failure: CheckerFailure | None = None
        for index, entry in enumerate(entries):
            parsed_entry = _parse_diagnostic_entry(
                entry=entry,
                index=index,
                tool=self.name,
                request=request,
                limits=limits,
            )
            if isinstance(parsed_entry, CheckerDiagnostic):
                field_violation = (
                    diagnostic_field_violation(
                        tool=self.name,
                        limits=limits,
                        diagnostic=parsed_entry,
                    )
                    if limits is not None
                    else None
                )
                if field_violation is not None:
                    return CheckerResult(
                        diagnostics=(),
                        failures=(field_violation,),
                    )
                diagnostics.append(parsed_entry)
            elif parsed_entry.kind == "output_limit_exceeded":
                return CheckerResult(
                    diagnostics=(),
                    failures=(parsed_entry,),
                )
            elif failure is None:
                failure = parsed_entry

        return CheckerResult(
            diagnostics=tuple(diagnostics),
            failures=() if failure is None else (failure,),
        )


def _parse_diagnostic_entry(
    *,
    entry: object,
    index: int,
    tool: str,
    request: CheckerRequest,
    limits: CheckerOutputLimits | None,
) -> CheckerDiagnostic | CheckerFailure:
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

    span = None
    if "range" in entry:
        span = _range_to_span(entry["range"])
        if span is None:
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

    return CheckerDiagnostic(
        tool=tool,
        path=path,
        code=rule,
        message=message,
        severity=severity,
        span=span,
    )


def _severity(value: object) -> DiagnosticSeverity | None:
    if value == "error":
        return "error"
    if value == "warning":
        return "warning"
    if value == "information":
        return "info"
    return None


def _range_to_span(value: object) -> TextSpan | None:
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
    ):
        return None
    return line_span(
        line=start_line + 1,
        column=start_character,
        end_line=end_line + 1,
        end_column=end_character,
        columns_are_one_based=False,
    )


__all__ = ["PyrightAdapter"]
