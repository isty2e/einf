import os
from pathlib import Path

from einf.analysis.checkers import CheckerAdapter, CheckerDiagnostic, CheckerFailure
from einf.analysis.engine import analyze_module
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import AstParserBackend, LibCstParserBackend, ParserBackend
from einf.analysis.validator.model import (
    ValidationFailure,
    ValidationFileReport,
    ValidationReport,
)

SCHEMA_VERSION = "0.1"
SUPPORTED_PARSER_NAMES = ("ast", "libcst")


def build_parser_backend(parser_name: str) -> ParserBackend:
    """Build one parser backend from validator CLI configuration."""
    match parser_name:
        case "ast":
            return AstParserBackend()
        case "libcst":
            return LibCstParserBackend()
        case _:
            raise ValueError(f"unsupported parser backend: {parser_name}")


def run_validation(
    *,
    targets: tuple[Path, ...],
    parser_backend: ParserBackend,
    checker_adapters: tuple[CheckerAdapter, ...] = (),
) -> ValidationReport:
    """Analyze Python targets and return one stable validation report."""
    resolved_targets = _resolve_python_targets(targets)
    project_root = _infer_project_root(resolved_targets)
    checker_targets = tuple(path for path in resolved_targets if path.is_file())
    analyzer_reports = {
        path: analyze_path(path=path, parser_backend=parser_backend)
        for path in resolved_targets
    }
    checker_failures, checker_diagnostics_by_path = run_checker_adapters(
        targets=checker_targets,
        checker_adapters=checker_adapters,
        project_root=project_root,
    )

    return ValidationReport(
        schema_version=SCHEMA_VERSION,
        parser_backend=parser_backend.name,
        checker_failures=checker_failures,
        files=tuple(
            _merge_file_report(
                path=path,
                analyzer_report=analyzer_reports.get(path),
                checker_diagnostics=checker_diagnostics_by_path.get(path, ()),
            )
            for path in sorted(
                set(analyzer_reports) | set(checker_diagnostics_by_path),
            )
        ),
    )


def _resolve_python_targets(targets: tuple[Path, ...]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for target in targets:
        if target.is_dir():
            candidates = tuple(
                sorted(path for path in target.rglob("*.py") if path.is_file())
            )
        else:
            candidates = (target,)
        for candidate in candidates:
            normalized = candidate.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(normalized)

    return tuple(resolved)


def _infer_project_root(targets: tuple[Path, ...]) -> Path:
    if not targets:
        return Path.cwd()
    roots = [path if path.is_dir() else path.parent for path in targets]
    return Path(os.path.commonpath([str(root) for root in roots]))


def run_checker_adapters(
    *,
    targets: tuple[Path, ...],
    checker_adapters: tuple[CheckerAdapter, ...],
    project_root: Path,
) -> tuple[
    tuple[CheckerFailure, ...],
    dict[Path, tuple[CheckerDiagnostic, ...]],
]:
    if not targets or not checker_adapters:
        return (), {}

    checker_failures: list[CheckerFailure] = []
    checker_diagnostics_by_path: dict[Path, list[CheckerDiagnostic]] = {}
    for checker_adapter in checker_adapters:
        checker_result = checker_adapter.run(
            targets=targets,
            project_root=project_root,
        )
        checker_failures.extend(checker_result.failures)
        for checker_diagnostic in checker_result.diagnostics:
            checker_diagnostics_by_path.setdefault(
                checker_diagnostic.path,
                [],
            ).append(checker_diagnostic)
    return (
        tuple(checker_failures),
        {
            path: _sort_checker_diagnostics(tuple(diagnostics))
            for path, diagnostics in checker_diagnostics_by_path.items()
        },
    )


def _sort_checker_diagnostics(
    checker_diagnostics: tuple[CheckerDiagnostic, ...],
) -> tuple[CheckerDiagnostic, ...]:
    return tuple(
        sorted(
            checker_diagnostics,
            key=lambda checker_diagnostic: (
                checker_diagnostic.span.start.line
                if checker_diagnostic.span is not None
                else -1,
                checker_diagnostic.span.start.column
                if checker_diagnostic.span is not None
                else -1,
                checker_diagnostic.tool,
                checker_diagnostic.code or "",
                checker_diagnostic.message,
            ),
        )
    )


def _merge_file_report(
    *,
    path: Path,
    analyzer_report: ValidationFileReport | None,
    checker_diagnostics: tuple[CheckerDiagnostic, ...],
) -> ValidationFileReport:
    if analyzer_report is None:
        return ValidationFileReport(
            path=str(path),
            diagnostics=(),
            checker_diagnostics=checker_diagnostics,
            axis_tokens=(),
            failures=(),
        )
    return ValidationFileReport(
        path=analyzer_report.path,
        diagnostics=analyzer_report.diagnostics,
        checker_diagnostics=checker_diagnostics,
        axis_tokens=analyzer_report.axis_tokens,
        failures=analyzer_report.failures,
    )


def analyze_path(
    *,
    path: Path,
    parser_backend: ParserBackend,
) -> ValidationFileReport:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ValidationFileReport(
            path=str(path),
            diagnostics=(),
            checker_diagnostics=(),
            axis_tokens=(),
            failures=(
                ValidationFailure(
                    kind="read_error",
                    message=str(error),
                    span=None,
                ),
            ),
        )

    return analyze_source(source=source, path=path, parser_backend=parser_backend)


def analyze_source(
    *,
    source: str,
    path: Path,
    parser_backend: ParserBackend,
) -> ValidationFileReport:
    """Analyze one in-memory source string as a single file report."""
    try:
        output = analyze_module(source=source, path=path, parser_backend=parser_backend)
    except SyntaxError as error:
        return _parse_error_report(path=path, error=error)

    return ValidationFileReport(
        path=str(path),
        diagnostics=output.diagnostics,
        checker_diagnostics=(),
        axis_tokens=output.axis_tokens,
        failures=(),
    )


def _parse_error_report(
    *,
    path: Path,
    error: SyntaxError,
) -> ValidationFileReport:
    return ValidationFileReport(
        path=str(path),
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=(
            ValidationFailure(
                kind="parse_error",
                message=str(error),
                span=_syntax_error_span(error),
            ),
        ),
    )


def _syntax_error_span(error: SyntaxError) -> TextSpan | None:
    line = error.lineno
    column = error.offset
    if line is None or column is None or line < 1 or column < 1:
        return None

    start = TextPosition(line=line, column=column - 1)
    end_line = error.end_lineno if error.end_lineno is not None else line
    end_column = error.end_offset if error.end_offset is not None else column + 1
    if end_line < 1 or end_column < 1:
        return TextSpan(
            start=start,
            end=TextPosition(line=start.line, column=start.column + 1),
        )

    end = TextPosition(
        line=end_line,
        column=max(start.column + 1, end_column - 1),
    )
    return TextSpan(start=start, end=end)


__all__ = ["SCHEMA_VERSION", "build_parser_backend", "run_validation"]
