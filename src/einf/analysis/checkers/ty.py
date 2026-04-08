import re
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import CheckerDiagnostic, CheckerResult
from einf.analysis.model import DiagnosticSeverity

_TY_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): "
    r"(?P<severity>error|warning|info)"
    r"(?:\[(?P<code>[^\]]+)\])? (?P<message>.+)$"
)


@dataclass(frozen=True, slots=True)
class TyAdapter(CheckerAdapter):
    """Adapter for ty concise text output."""

    name: str = "ty"
    executable: str = "ty"

    def build_command(
        self,
        *,
        targets: tuple[Path, ...],
        project_root: Path,
    ) -> list[str]:
        return [
            self.executable,
            "check",
            *[str(path) for path in targets],
            "--output-format",
            "concise",
            "--no-progress",
        ]

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        project_root: Path,
    ) -> CheckerResult:
        diagnostics: list[CheckerDiagnostic] = []
        for line in stdout.splitlines():
            match = _TY_LINE.match(line.strip())
            if match is None:
                continue
            severity = match.group("severity")
            diagnostics.append(
                CheckerDiagnostic(
                    tool=self.name,
                    path=resolve_report_path(match.group("path"), project_root),
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

        return CheckerResult(diagnostics=tuple(diagnostics), failures=())


def _severity_from_text(value: str) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value == "info":
        return "info"
    return "error"


__all__ = ["TyAdapter"]
