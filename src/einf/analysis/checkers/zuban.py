import re
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import CheckerDiagnostic, CheckerResult
from einf.analysis.model import DiagnosticSeverity

_ZUBAN_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+):(?P<end_line>\d+):(?P<end_column>\d+): "
    r"(?P<severity>error|warning|info): (?P<message>.+?)"
    r"(?:  \[(?P<code>[^\]]+)\])?$"
)


@dataclass(frozen=True, slots=True)
class ZubanAdapter(CheckerAdapter):
    """Adapter for zuban text diagnostics."""

    name: str = "zuban"
    executable: str = "zuban"

    def build_command(
        self,
        *,
        targets: tuple[Path, ...],
        project_root: Path,
    ) -> list[str]:
        return [
            self.executable,
            "check",
            *[_target_arg(path=path, project_root=project_root) for path in targets],
            "--no-pretty",
            "--show-column-numbers",
            "--show-error-end",
            "--show-error-codes",
        ]

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        project_root: Path,
    ) -> CheckerResult:
        diagnostics: list[CheckerDiagnostic] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            match = _ZUBAN_LINE.match(line)
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
                        end_line=int(match.group("end_line")),
                        end_column=int(match.group("end_column")),
                        columns_are_one_based=True,
                    ),
                )
            )

        return CheckerResult(diagnostics=tuple(diagnostics), failures=())


def _target_arg(*, path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _severity_from_text(value: str) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value == "info":
        return "info"
    return "error"


__all__ = ["ZubanAdapter"]
