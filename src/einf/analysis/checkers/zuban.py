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
        unrecognized_line: str | None = None
        for output in (stdout, stderr):
            for raw_line in output.splitlines():
                line = raw_line.strip()
                if not line or _ZUBAN_SUMMARY_LINE.fullmatch(line):
                    continue
                match = _ZUBAN_LINE.match(line)
                if match is None:
                    if unrecognized_line is None:
                        unrecognized_line = line
                    continue
                severity = match.group("severity")
                diagnostics.append(
                    CheckerDiagnostic(
                        tool=self.name,
                        path=resolve_report_path(match.group("path"), request),
                        code=match.group("code"),
                        message=match.group("message"),
                        severity=_severity_from_text(severity),
                        span=line_span(
                            line=int(match.group("line")),
                            column=int(match.group("column")),
                            end_line=int(match.group("end_line")),
                            end_column=int(match.group("end_column")),
                            columns_are_one_based=True,
                        ),
                    )
                )

        failures = (
            ()
            if unrecognized_line is None
            else (
                CheckerFailure(
                    tool=self.name,
                    kind="output_parse_error",
                    message=(
                        "zuban output contained an unrecognized line: "
                        f"{unrecognized_line}"
                    ),
                ),
            )
        )
        return CheckerResult(diagnostics=tuple(diagnostics), failures=failures)


def _severity_from_text(value: str) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value in ("info", "note"):
        return "info"
    return "error"


__all__ = ["ZubanAdapter"]
