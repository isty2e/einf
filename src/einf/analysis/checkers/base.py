import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from einf.analysis.checkers.model import CheckerFailure, CheckerResult
from einf.analysis.model import TextPosition, TextSpan


class CheckerAdapter(ABC):
    """Abstract subprocess-backed external checker adapter."""

    name: str
    executable: str

    def run(
        self,
        *,
        targets: tuple[Path, ...],
        project_root: Path,
    ) -> CheckerResult:
        """Execute one checker and normalize its diagnostics."""
        if shutil.which(self.executable) is None:
            return CheckerResult(
                diagnostics=(),
                failures=(
                    CheckerFailure(
                        tool=self.name,
                        kind="unavailable",
                        message=f"checker executable not found: {self.executable}",
                    ),
                ),
            )

        process = subprocess.run(
            self.build_command(targets=targets, project_root=project_root),
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        result = self.parse_output(
            stdout=process.stdout,
            stderr=process.stderr,
            project_root=project_root,
        )
        if process.returncode == 1 and not result.diagnostics and not result.failures:
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
        if process.returncode not in (0, 1) and not result.failures:
            message = process.stderr.strip() or process.stdout.strip()
            return CheckerResult(
                diagnostics=result.diagnostics,
                failures=result.failures
                + (
                    CheckerFailure(
                        tool=self.name,
                        kind="execution_error",
                        message=message
                        or f"{self.name} exited with code {process.returncode}",
                    ),
                ),
            )
        return result

    @abstractmethod
    def build_command(
        self,
        *,
        targets: tuple[Path, ...],
        project_root: Path,
    ) -> list[str]:
        """Build one subprocess command for the checker."""

    @abstractmethod
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        project_root: Path,
    ) -> CheckerResult:
        """Parse checker stdout/stderr into canonical result objects."""


def resolve_report_path(path_text: str, project_root: Path) -> Path:
    """Resolve one checker-reported file path against the project root."""
    reported = Path(path_text)
    if reported.is_absolute():
        return reported.resolve(strict=False)
    return (project_root / reported).resolve(strict=False)


def line_span(
    *,
    line: int,
    column: int,
    end_line: int | None = None,
    end_column: int | None = None,
    columns_are_one_based: bool,
) -> TextSpan | None:
    """Build one canonical text span from checker coordinates."""
    if line < 1 or column < 0:
        return None

    start_column = column - 1 if columns_are_one_based else column
    if start_column < 0:
        return None

    start = TextPosition(line=line, column=start_column)
    if end_line is None or end_column is None:
        return TextSpan(
            start=start,
            end=TextPosition(line=line, column=start_column + 1),
        )

    normalized_end_column = end_column - 1 if columns_are_one_based else end_column
    if end_line < 1 or normalized_end_column < start_column:
        return TextSpan(
            start=start,
            end=TextPosition(line=line, column=start_column + 1),
        )

    return TextSpan(
        start=start,
        end=TextPosition(
            line=end_line,
            column=max(start_column + 1, normalized_end_column),
        ),
    )


__all__ = ["CheckerAdapter", "line_span", "resolve_report_path"]
