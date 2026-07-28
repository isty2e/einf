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

_TY_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): "
    r"(?P<severity>error|warning|info)"
    r"(?:\[(?P<code>[^\]]+)\])? (?P<message>.+)$"
)
_TY_SUMMARY_LINE = re.compile(r"^(?:Found \d+ diagnostics?|All checks passed!)$")


@dataclass(frozen=True, slots=True)
class TyAdapter(CheckerAdapter):
    """Adapter for ty concise text output."""

    name: str = "ty"
    executable: str = "ty"

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
            "concise",
            "--no-progress",
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
                if not line or _TY_SUMMARY_LINE.fullmatch(line):
                    continue
                match = _TY_LINE.match(line)
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
                        f"ty output contained an unrecognized line: {unrecognized_line}"
                    ),
                ),
            )
        )
        return CheckerResult(diagnostics=tuple(diagnostics), failures=failures)


def _severity_from_text(value: str) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value == "info":
        return "info"
    return "error"


__all__ = ["TyAdapter"]
