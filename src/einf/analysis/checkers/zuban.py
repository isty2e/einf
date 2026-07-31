import re
from dataclasses import dataclass

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.model import DiagnosticSeverity

_ZUBAN_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+):(?P<end_line>\d+):(?P<end_column>\d+): "
    r"(?P<severity>error|warning|info|note): (?P<message>.+?)"
    r"(?:  \[(?P<code>[^\]]+)\])?$"
)
_ZUBAN_SUMMARY_LINE = re.compile(
    r"^(?:Success: no issues found in \d+ source files?|"
    r"Found \d+ errors? in \d+ files? \(checked \d+ source files?\))$"
)


@dataclass(frozen=True, slots=True)
class ZubanAdapter(CheckerAdapter):
    """Adapter for zuban text diagnostics."""

    name: str = "zuban"
    executable: str = "zuban"

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        target_arguments: list[str] = []
        for path in request.targets:
            try:
                target_arguments.append(str(path.relative_to(request.project_root)))
            except ValueError:
                target_arguments.append(str(path))
        return (
            self.executable,
            "check",
            *target_arguments,
            "--no-pretty",
            "--show-column-numbers",
            "--show-error-end",
            "--show-error-codes",
        )

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        diagnostics: list[CheckerDiagnostic] = []
        failure: CheckerFailure | None = None
        for output in (stdout, stderr):
            for raw_line in output.splitlines():
                line = raw_line.strip()
                if not line or _ZUBAN_SUMMARY_LINE.fullmatch(line):
                    continue
                match = _ZUBAN_LINE.match(line)
                if match is None:
                    if failure is None:
                        failure = CheckerFailure(
                            tool=self.name,
                            kind="output_parse_error",
                            message=(
                                f"zuban output contained an unrecognized line: {line}"
                            ),
                        )
                    continue
                parsed_line = _parse_diagnostic_line(
                    match=match,
                    tool=self.name,
                    request=request,
                )
                if isinstance(parsed_line, CheckerDiagnostic):
                    diagnostics.append(parsed_line)
                elif failure is None:
                    failure = parsed_line

        return CheckerResult(
            diagnostics=tuple(diagnostics),
            failures=() if failure is None else (failure,),
        )


def _parse_diagnostic_line(
    *,
    match: re.Match[str],
    tool: str,
    request: CheckerRequest,
) -> CheckerDiagnostic | CheckerFailure:
    try:
        line = int(match.group("line"))
        column = int(match.group("column"))
        end_line = int(match.group("end_line"))
        end_column = int(match.group("end_column"))
    except ValueError:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic has invalid coordinates",
        )

    span = line_span(
        line=line,
        column=column,
        end_line=end_line,
        end_column=end_column,
        columns_are_one_based=True,
    )
    if span is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic has invalid coordinates",
        )

    path = resolve_report_path(match.group("path"), request)
    if path is None:
        return CheckerFailure(
            tool=tool,
            kind="output_parse_error",
            message=f"{tool} diagnostic has an invalid file path",
        )

    return CheckerDiagnostic(
        tool=tool,
        path=path,
        code=match.group("code"),
        message=match.group("message"),
        severity=_severity_from_text(match.group("severity")),
        span=span,
    )


def _severity_from_text(value: str) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value in ("info", "note"):
        return "info"
    return "error"


__all__ = ["ZubanAdapter"]
