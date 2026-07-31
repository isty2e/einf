from abc import ABC, abstractmethod
from pathlib import Path

from einf.analysis.checkers.model import (
    CheckerFailure,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.model import TextPosition, TextSpan


class CheckerAdapter(ABC):
    """Translate between checker-neutral requests and tool-specific output."""

    name: str
    executable: str

    @abstractmethod
    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        """Build one subprocess command for the checker."""

    @abstractmethod
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        """Parse checker stdout/stderr into canonical result objects."""

    def normalize_output(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        """Normalize one completed checker process into canonical results."""
        result = self.parse_output(stdout=stdout, stderr=stderr, request=request)
        if returncode == 1 and not result.diagnostics and not result.failures:
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="output_parse_error",
                        message=(
                            f"{self.name} exited with code 1 without "
                            "recognized diagnostics"
                        ),
                    ),
                ),
            )
        if returncode not in (0, 1) and not result.failures:
            message = stderr.strip() or stdout.strip()
            return CheckerResult(
                diagnostics=result.diagnostics,
                failures=result.failures
                + (
                    CheckerFailure(
                        tool=self.name,
                        kind="execution_error",
                        message=message or f"{self.name} exited with code {returncode}",
                    ),
                ),
            )
        return result


def resolve_report_path(path_text: str, request: CheckerRequest) -> Path | None:
    """Resolve one checker-reported path, returning None for invalid input."""
    if not path_text:
        return None
    try:
        reported = Path(path_text)
        if reported.is_absolute():
            return reported.resolve(strict=False)
        return (request.project_root / reported).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def line_span(
    *,
    line: int,
    column: int,
    end_line: int | None = None,
    end_column: int | None = None,
    columns_are_one_based: bool,
) -> TextSpan | None:
    """Build one canonical text span from checker coordinates."""
    if type(line) is not int or type(column) is not int:
        return None
    if line < 1 or column < 0:
        return None

    start_column = column - 1 if columns_are_one_based else column
    if start_column < 0:
        return None

    start = TextPosition(line=line, column=start_column)
    if end_line is None and end_column is None:
        return TextSpan(
            start=start,
            end=TextPosition(line=line, column=start_column + 1),
        )
    if type(end_line) is not int or type(end_column) is not int:
        return None

    normalized_end_column = end_column - 1 if columns_are_one_based else end_column
    if end_line < line or normalized_end_column < 0:
        return None
    if end_line == line and normalized_end_column < start_column:
        return None

    canonical_end_column = normalized_end_column
    if end_line == line:
        canonical_end_column = max(start_column + 1, normalized_end_column)

    return TextSpan(
        start=start,
        end=TextPosition(
            line=end_line,
            column=canonical_end_column,
        ),
    )


__all__ = ["CheckerAdapter", "line_span", "resolve_report_path"]
