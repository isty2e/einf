import json
from dataclasses import dataclass

from einf.analysis.checkers.base import CheckerAdapter, line_span, resolve_report_path
from einf.analysis.checkers.model import (
    CheckerDiagnostic,
    CheckerFailure,
    CheckerRequest,
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
        except (RecursionError, ValueError) as error:
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
        failure: CheckerFailure | None = None
        for index, entry in enumerate(errors):
            parsed_entry = _parse_diagnostic_entry(
                entry=entry,
                index=index,
                tool=self.name,
                request=request,
            )
            if isinstance(parsed_entry, CheckerDiagnostic):
                diagnostics.append(parsed_entry)
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

    return CheckerDiagnostic(
        tool=tool,
        path=path,
        code=name,
        message=description,
        severity="error",
        span=span,
    )


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
