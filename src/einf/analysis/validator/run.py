import asyncio
import os
from pathlib import Path

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerDiagnostic,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.engine import analyze_module
from einf.analysis.parser import (
    AstParserBackend,
    LibCstParserBackend,
    ParserBackend,
    ParserSyntaxError,
    ParserUnavailableError,
)
from einf.analysis.validator.model import (
    ValidationDiscoveryFailure,
    ValidationFailure,
    ValidationFileReport,
    ValidationReport,
)

SCHEMA_VERSION = "0.2"
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
    checker_execution_policy: CheckerExecutionPolicy | None = None,
) -> ValidationReport:
    """Analyze Python targets and return one stable validation report."""
    resolved_targets, discovery_failures = _resolve_python_targets(targets)
    project_root = _infer_project_root(resolved_targets)
    checker_targets = tuple(path for path in resolved_targets if path.is_file())
    try:
        parser_backend.validate_available()
    except ParserUnavailableError as error:
        analyzer_reports = {
            path: _parser_unavailable_report(path=path, error=error)
            for path in resolved_targets
        }
    else:
        analyzer_reports = {
            path: analyze_path(path=path, parser_backend=parser_backend)
            for path in resolved_targets
        }
    checker_result = _run_checker_adapters(
        targets=checker_targets,
        checker_adapters=checker_adapters,
        project_root=project_root,
        execution_policy=checker_execution_policy or CheckerExecutionPolicy(),
    )

    return ValidationReport(
        schema_version=SCHEMA_VERSION,
        parser_backend=parser_backend.name,
        checker_failures=checker_result.failures,
        discovery_failures=discovery_failures,
        files=tuple(
            _merge_file_report(
                path=path,
                analyzer_report=analyzer_reports.get(path),
                checker_diagnostics=checker_result.diagnostics_for(path),
            )
            for path in sorted(
                set(analyzer_reports)
                | {diagnostic.path for diagnostic in checker_result.diagnostics},
            )
        ),
    )


def _resolve_python_targets(
    targets: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[ValidationDiscoveryFailure, ...]]:
    resolved: list[Path] = []
    discovery_failures: list[ValidationDiscoveryFailure] = []
    seen: set[Path] = set()

    for target in targets:
        if target.is_dir():
            candidates, target_failures = _walk_python_files(target)
            discovery_failures.extend(target_failures)
        else:
            candidates = (target,)
        for candidate in candidates:
            normalized = candidate.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(normalized)

    return (
        tuple(resolved),
        tuple(
            sorted(
                set(discovery_failures),
                key=lambda failure: (failure.path, failure.kind, failure.message),
            )
        ),
    )


def _walk_python_files(
    target: Path,
) -> tuple[tuple[Path, ...], tuple[ValidationDiscoveryFailure, ...]]:
    candidates: list[Path] = []
    failures: list[ValidationDiscoveryFailure] = []
    pending_directories = [target]

    def record_failure(error: OSError, *, fallback_path: Path) -> None:
        failed_path = (
            fallback_path
            if error.filename is None
            else Path(os.fsdecode(error.filename))
        )
        failures.append(
            ValidationDiscoveryFailure(
                path=str(failed_path.resolve(strict=False)),
                kind="directory_traversal_error",
                message=str(error),
            )
        )

    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                directory_entries = sorted(entries, key=lambda entry: entry.name)
        except OSError as error:
            record_failure(error, fallback_path=directory)
            continue

        child_directories: list[Path] = []
        for entry in directory_entries:
            entry_path = Path(entry.path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as error:
                record_failure(error, fallback_path=entry_path)
                continue
            if is_directory:
                child_directories.append(entry_path)
                continue
            if not entry.name.endswith(".py"):
                continue
            try:
                is_file = entry.is_file()
            except OSError as error:
                record_failure(error, fallback_path=entry_path)
                continue
            if is_file:
                candidates.append(entry_path)

        pending_directories.extend(reversed(child_directories))

    return tuple(sorted(candidates)), tuple(failures)


def _infer_project_root(targets: tuple[Path, ...]) -> Path:
    if not targets:
        return Path.cwd()
    roots = [path if path.is_dir() else path.parent for path in targets]
    return Path(os.path.commonpath([str(root) for root in roots]))


def _run_checker_adapters(
    *,
    targets: tuple[Path, ...],
    checker_adapters: tuple[CheckerAdapter, ...],
    project_root: Path,
    execution_policy: CheckerExecutionPolicy,
) -> CheckerResult:
    if not targets or not checker_adapters:
        return CheckerResult(diagnostics=(), failures=())

    request = CheckerRequest(targets=targets, project_root=project_root)

    async def execute() -> CheckerResult:
        executor = CheckerExecutor(execution_policy)
        return await executor.run_all(checker_adapters, request)

    return asyncio.run(execute())


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
    except ParserSyntaxError as error:
        return _parse_error_report(path=path, error=error)
    except ParserUnavailableError as error:
        return _parser_unavailable_report(path=path, error=error)

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
    error: ParserSyntaxError,
) -> ValidationFileReport:
    return ValidationFileReport(
        path=str(path),
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=(
            ValidationFailure(
                kind="parse_error",
                message=error.message,
                span=error.span,
            ),
        ),
    )


def _parser_unavailable_report(
    *,
    path: Path,
    error: ParserUnavailableError,
) -> ValidationFileReport:
    return ValidationFileReport(
        path=str(path),
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=(
            ValidationFailure(
                kind="parser_unavailable",
                message=error.message,
                span=None,
            ),
        ),
    )


__all__ = ["SCHEMA_VERSION", "build_parser_backend", "run_validation"]
