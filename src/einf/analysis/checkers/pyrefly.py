import json
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
    CheckerResult,
)
from einf.analysis.model import TextSpan


@dataclass(frozen=True, slots=True)
class PyreflyAdapter(CheckerAdapter):
    """Adapter for pyrefly JSON output."""

    name: str = "pyrefly"
    executable: str = "pyrefly"

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
            "json",
            "--summary=none",
        ]

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
                        message="pyrefly output must be a JSON object",
                    ),
                ),
            )

        errors = payload.get("errors")
        if not isinstance(errors, list):
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message="pyrefly output missing errors list",
                    ),
                ),
            )

        diagnostics: list[CheckerDiagnostic] = []
        for entry in errors:
            if not isinstance(entry, dict):
                continue
            path_text = entry.get("path")
            description = entry.get("description")
            if not isinstance(path_text, str) or not isinstance(description, str):
                continue
            line = entry.get("line")
            column = entry.get("column")
            stop_line = entry.get("stop_line")
            stop_column = entry.get("stop_column")
            name = entry.get("name")
            diagnostics.append(
                CheckerDiagnostic(
                    tool=self.name,
                    path=resolve_report_path(path_text, project_root),
                    code=name if isinstance(name, str) else None,
                    message=description,
                    severity="error",
                    span=_entry_span(
                        line=line,
                        column=column,
                        stop_line=stop_line,
                        stop_column=stop_column,
                    ),
                )
            )

        return CheckerResult(diagnostics=tuple(diagnostics), failures=())


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
