import asyncio
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerDiagnostic,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.engine import analyze_source
from einf.analysis.parser import ParserBackend, ParserUnavailableError
from einf.analysis.report import (
    AnalysisFailure,
    AnalysisFileReport,
    parser_unavailable_report,
)
from einf.analysis.validator.limits import ValidationLimitFailure, ValidationLimits
from einf.analysis.validator.model import (
    ValidationDiscoveryFailure,
    ValidationReport,
)

SCHEMA_VERSION = "0.4"


class _ValidationLimitExceeded(Exception):
    def __init__(self, failure: ValidationLimitFailure) -> None:
        self.failure = failure
        super().__init__(
            f"{failure.resource} limit exceeded: {failure.observed} > {failure.limit}"
        )


@dataclass(slots=True)
class _ValidationBudget:
    limits: ValidationLimits
    entries: int = 0
    source_bytes: int = 0

    def visit_entry(self, path: str) -> None:
        self.entries += 1
        maximum = self.limits.max_discovery_entries
        if maximum is not None and self.entries > maximum:
            raise _ValidationLimitExceeded(
                ValidationLimitFailure("discovery_entries", maximum, self.entries, path)
            )

    def source_allowance(self) -> int | None:
        allowance = self.limits.max_file_bytes
        total = self.limits.max_total_bytes
        if total is not None:
            remaining = total - self.source_bytes
            allowance = remaining if allowance is None else min(allowance, remaining)
        return allowance

    def check_source_size(self, size: int, path: str) -> None:
        self._check_bytes(size, self.source_bytes + size, path)

    def record_chunk(self, size: int, *, file_bytes: int, path: str) -> None:
        total = self.source_bytes + size
        self._check_bytes(file_bytes, total, path)
        self.source_bytes = total

    def _check_bytes(self, file_bytes: int, total_bytes: int, path: str) -> None:
        for resource, maximum, observed in (
            ("file_bytes", self.limits.max_file_bytes, file_bytes),
            ("total_bytes", self.limits.max_total_bytes, total_bytes),
        ):
            if maximum is not None and observed > maximum:
                raise _ValidationLimitExceeded(
                    ValidationLimitFailure(resource, maximum, observed, path)
                )


def run_validation(
    *,
    targets: tuple[Path, ...],
    parser_backend: ParserBackend,
    checker_adapters: tuple[CheckerAdapter, ...] = (),
    checker_execution_policy: CheckerExecutionPolicy | None = None,
    limits: ValidationLimits | None = None,
) -> ValidationReport:
    """Analyze Python targets and return one stable validation report.

    Parameters
    ----------
    targets : tuple[Path, ...]
        Source files and recursively traversed directories.
    parser_backend : ParserBackend
        Parser used for semantic analysis.
    checker_adapters : tuple[CheckerAdapter, ...], optional
        External checker integrations to run after source admission.
    checker_execution_policy : CheckerExecutionPolicy or None, optional
        Subprocess and output limits for external checkers.
    limits : ValidationLimits or None, optional
        Opt-in discovery and source-capture quotas. None is unrestricted.

    Returns
    -------
    ValidationReport
        Diagnostics and failures, including any quota that stopped validation.

    Raises
    ------
    TypeError
        If limits is neither ValidationLimits nor None.

    Notes
    -----
    A quota failure stops further work and prevents checker execution. Completed
    file reports can remain in the result; its exit code is still 1.
    """
    if limits is not None and not isinstance(limits, ValidationLimits):
        raise TypeError("limits must be ValidationLimits or None")
    budget = _ValidationBudget(limits if limits is not None else ValidationLimits())
    try:
        resolved_targets, discovery_failures = _resolve_python_targets(targets, budget)
    except _ValidationLimitExceeded as error:
        return ValidationReport(
            schema_version=SCHEMA_VERSION,
            parser_backend=parser_backend.name,
            checker_failures=(),
            discovery_failures=(),
            files=(),
            limit_failures=(error.failure,),
        )
    project_root = _infer_project_root(resolved_targets)
    checker_targets = tuple(path for path in resolved_targets if path.is_file())
    unavailable: ParserUnavailableError | None = None
    try:
        parser_backend.validate_available()
    except ParserUnavailableError as error:
        unavailable = error

    analyzer_reports: dict[Path, AnalysisFileReport] = {}
    limit_failures: tuple[ValidationLimitFailure, ...] = ()
    for path in resolved_targets:
        try:
            analyzer_reports[path] = _analyze_path(
                path=path,
                parser_backend=parser_backend,
                budget=budget,
                unavailable=unavailable,
            )
        except _ValidationLimitExceeded as error:
            limit_failures = (error.failure,)
            break

    checker_result = (
        CheckerResult(diagnostics=(), failures=())
        if limit_failures
        else _run_checker_adapters(
            targets=checker_targets,
            checker_adapters=checker_adapters,
            project_root=project_root,
            execution_policy=checker_execution_policy or CheckerExecutionPolicy(),
        )
    )

    return ValidationReport(
        schema_version=SCHEMA_VERSION,
        parser_backend=parser_backend.name,
        checker_failures=checker_result.failures,
        discovery_failures=discovery_failures,
        limit_failures=limit_failures,
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
    budget: _ValidationBudget,
) -> tuple[tuple[Path, ...], tuple[ValidationDiscoveryFailure, ...]]:
    resolved: list[Path] = []
    discovery_failures: list[ValidationDiscoveryFailure] = []
    seen: set[Path] = set()

    for target in targets:
        budget.visit_entry(str(target.absolute()))
        if target.is_dir():
            candidates, target_failures = _walk_python_files(target, budget)
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
    budget: _ValidationBudget,
) -> tuple[tuple[Path, ...], tuple[ValidationDiscoveryFailure, ...]]:
    candidates: list[Path] = []
    failures: list[ValidationDiscoveryFailure] = []
    pending_directories = [target]
    visited_directories: set[tuple[int, int]] = set()
    discovery_root = str(target.absolute())

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
        # Follow directory symlinks for rglob parity while bounding cycles by identity.
        try:
            directory_stat = directory.stat()
        except OSError as error:
            record_failure(error, fallback_path=directory)
            continue
        directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
        if directory_identity in visited_directories:
            continue
        visited_directories.add(directory_identity)

        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    budget.visit_entry(discovery_root)
                    entry_path = Path(entry.path)
                    try:
                        is_directory = entry.is_dir()
                    except OSError as error:
                        record_failure(error, fallback_path=entry_path)
                        continue
                    if is_directory:
                        pending_directories.append(entry_path)
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
        except OSError as error:
            record_failure(error, fallback_path=directory)

    candidates.sort()
    return tuple(candidates), tuple(failures)


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
    analyzer_report: AnalysisFileReport | None,
    checker_diagnostics: tuple[CheckerDiagnostic, ...],
) -> AnalysisFileReport:
    if analyzer_report is None:
        return AnalysisFileReport(
            path=str(path),
            diagnostics=(),
            checker_diagnostics=checker_diagnostics,
            axis_tokens=(),
            failures=(),
        )
    return AnalysisFileReport(
        path=analyzer_report.path,
        diagnostics=analyzer_report.diagnostics,
        checker_diagnostics=checker_diagnostics,
        axis_tokens=analyzer_report.axis_tokens,
        failures=analyzer_report.failures,
    )


def _read_source(path: Path, budget: _ValidationBudget) -> str:
    allowance = budget.source_allowance()
    if allowance is None:
        return path.read_text(encoding="utf-8")

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("source byte limits require a regular file")
        budget.check_source_size(metadata.st_size, str(path))

        source = bytearray()
        while True:
            chunk = os.read(descriptor, min(65536, allowance - len(source) + 1))
            if not chunk:
                break
            budget.record_chunk(
                len(chunk), file_bytes=len(source) + len(chunk), path=str(path)
            )
            source.extend(chunk)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            # Cleanup must not replace the original admission or read failure.
            pass
        raise
    else:
        os.close(descriptor)

    return source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _analyze_path(
    *,
    path: Path,
    parser_backend: ParserBackend,
    budget: _ValidationBudget,
    unavailable: ParserUnavailableError | None,
) -> AnalysisFileReport:
    if unavailable is not None and budget.source_allowance() is None:
        return parser_unavailable_report(path=path, error=unavailable)
    try:
        source = _read_source(path, budget)
    except (OSError, UnicodeError) as error:
        return AnalysisFileReport(
            path=str(path),
            diagnostics=(),
            checker_diagnostics=(),
            axis_tokens=(),
            failures=(
                AnalysisFailure(
                    kind="read_error",
                    message=str(error),
                    span=None,
                ),
            ),
        )

    if unavailable is not None:
        return parser_unavailable_report(path=path, error=unavailable)
    return analyze_source(source=source, path=path, parser_backend=parser_backend)


__all__ = ["SCHEMA_VERSION", "run_validation"]
