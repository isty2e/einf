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
class PyreflyAdapter(CheckerAdapter):
    """Adapter for pyrefly JSON output."""

    name: str = "pyrefly"
    executable: str = "pyrefly"

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        return (
            self.executable,
            "check",
            *[str(path) for path in request.targets],
            "--output-format",
            "json",
            "--summary=none",
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
            field="errors",
            max_entries=max_entries,
        )
        if parse_error is not None:
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message=f"pyrefly {parse_error}",
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

    path_text = entry.get("path")
    if not isinstance(path_text, str) or not path_text:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid file path",
        )
    if limits is not None and len(path_text) > limits.max_field_length:
        return field_limit_failure(tool=tool, limits=limits)
    path = resolve_report_path(path_text, request)
    if path is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has an invalid file path",
        )

    description = entry.get("description")
    if not isinstance(description, str):
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid description",
        )

    severity = _severity(entry.get("severity"))
    if severity is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid severity",
        )

    span = _entry_span(
        line=entry.get("line"),
        column=entry.get("column"),
        stop_line=entry.get("stop_line"),
        stop_column=entry.get("stop_column"),
    )
    if span is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has no valid range",
        )

    name = entry.get("name")
    if name is not None and not isinstance(name, str):
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic {index} has an invalid name",
        )
    if limits is not None and (
        len(description) > limits.max_field_length
        or (name is not None and len(name) > limits.max_field_length)
    ):
        return field_limit_failure(tool=tool, limits=limits)

    return CheckerDiagnostic(
        tool=tool,
        path=path,
        code=name,
        message=description,
        severity=severity,
        span=span,
    )


def _severity(value: object) -> DiagnosticSeverity | None:
    if value == "error":
        return "error"
    if value == "warn":
        return "warning"
    if value == "info":
        return "info"
    return None


def _entry_span(
    *,
    line: object,
    column: object,
    stop_line: object,
    stop_column: object,
) -> TextSpan | None:
    if (
        type(line) is not int
        or type(column) is not int
        or type(stop_line) is not int
        or type(stop_column) is not int
    ):
        return None
    return line_span(
        line=line,
        column=column,
        end_line=stop_line,
        end_column=stop_column,
        columns_are_one_based=True,
    )


__all__ = ["PyreflyAdapter"]
