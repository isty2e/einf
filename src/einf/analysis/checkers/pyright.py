import json
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
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
        *,
        targets: tuple[Path, ...],
        project_root: Path,
    ) -> list[str]:
        return [self.executable, "--outputjson", *[str(path) for path in targets]]

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        project_root: Path,
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

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message=str(error),
                    ),
                ),
            )

        if not isinstance(payload, dict):
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message="pyright output must be a JSON object",
                    ),
                ),
            )

        diagnostics_field = payload.get("generalDiagnostics")
        if not isinstance(diagnostics_field, list):
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message="pyright output missing generalDiagnostics",
                    ),
                ),
            )

        diagnostics: list[CheckerDiagnostic] = []
        for entry in diagnostics_field:
            if not isinstance(entry, dict):
                continue
            file_path = entry.get("file")
            severity = _severity(entry.get("severity"))
            message = entry.get("message")
            if not isinstance(file_path, str) or not isinstance(message, str):
                continue
            rule = entry.get("rule")
            code = rule if isinstance(rule, str) else None
            diagnostics.append(
                CheckerDiagnostic(
                    tool=self.name,
                    path=resolve_report_path(file_path, project_root),
                    code=code,
                    message=message,
                    severity=severity,
                    span=_range_to_span(entry.get("range")),
                )
            )

        return CheckerResult(diagnostics=tuple(diagnostics), failures=())


def _severity(value: object) -> DiagnosticSeverity:
    if value == "warning":
        return "warning"
    if value == "information":
        return "info"
    return "error"


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
