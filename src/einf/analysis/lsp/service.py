from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerDiagnostic,
    CheckerFailure,
    build_checker_adapters,
)
from einf.analysis.parser import ParserBackend
from einf.analysis.validator.model import ValidationFileReport
from einf.analysis.validator.run import (
    analyze_source,
    build_parser_backend,
    run_checker_adapters,
)

from .config import LspConfig


@dataclass(frozen=True, slots=True)
class LspDocumentState:
    """Current in-memory LSP analysis state for one document URI."""

    uri: str
    path: Path | None
    version: int | None
    report: ValidationFileReport
    checker_failures: tuple[CheckerFailure, ...]
    checker_fresh: bool


class LspService:
    """Pure document-analysis service used by the pygls transport layer.

    `einf` semantic analysis is the primary responsibility. External checker
    execution is an optional save-boundary fallback for single-server editor
    setups, not the preferred integration path.
    """

    def __init__(self, config: LspConfig) -> None:
        self._config = config
        self._parser_backend: ParserBackend = build_parser_backend(config.parser)
        self._checker_adapters: tuple[CheckerAdapter, ...] = build_checker_adapters(
            config.checkers
        )
        self._states: dict[str, LspDocumentState] = {}

    @property
    def config(self) -> LspConfig:
        """Return the immutable session configuration."""
        return self._config

    def open_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Analyze an opened document without running external checkers."""
        return self._update_document(
            uri=uri,
            source=source,
            version=version,
            run_checkers=False,
        )

    def change_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Reanalyze one changed document and clear stale checker state."""
        return self._update_document(
            uri=uri,
            source=source,
            version=version,
            run_checkers=False,
        )

    def save_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Reanalyze one saved document and refresh fallback checker state."""
        return self._update_document(
            uri=uri,
            source=source,
            version=version,
            run_checkers=True,
        )

    def close_document(self, *, uri: str) -> None:
        """Drop cached state for one closed document."""
        self._states.pop(uri, None)

    def get_document_state(self, *, uri: str) -> LspDocumentState | None:
        """Return the cached document state, if available."""
        return self._states.get(uri)

    def _update_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
        run_checkers: bool,
    ) -> LspDocumentState:
        path = path_from_uri(uri)
        report = self._analyze_document(path=path, source=source)
        checker_failures: tuple[CheckerFailure, ...] = ()
        checker_diagnostics: tuple[CheckerDiagnostic, ...] = ()
        checker_fresh = False

        if run_checkers and path is not None and self._checker_adapters:
            checker_failures, checker_diagnostics = self._run_document_checkers(
                path=path
            )
            checker_fresh = True

        merged_report = ValidationFileReport(
            path=report.path,
            diagnostics=report.diagnostics,
            checker_diagnostics=checker_diagnostics,
            axis_tokens=report.axis_tokens,
            failures=report.failures,
        )
        state = LspDocumentState(
            uri=uri,
            path=path,
            version=version,
            report=merged_report,
            checker_failures=checker_failures,
            checker_fresh=checker_fresh,
        )
        self._states[uri] = state
        return state

    def _analyze_document(
        self,
        *,
        path: Path | None,
        source: str,
    ) -> ValidationFileReport:
        if path is None:
            return ValidationFileReport(
                path="",
                diagnostics=(),
                checker_diagnostics=(),
                axis_tokens=(),
                failures=(),
            )
        return analyze_source(
            source=source,
            path=path,
            parser_backend=self._parser_backend,
        )

    def _run_document_checkers(
        self,
        *,
        path: Path,
    ) -> tuple[tuple[CheckerFailure, ...], tuple[CheckerDiagnostic, ...]]:
        checker_failures, checker_diagnostics_by_path = run_checker_adapters(
            targets=(path,),
            checker_adapters=self._checker_adapters,
            project_root=path.parent,
        )
        return checker_failures, checker_diagnostics_by_path.get(path, ())


def path_from_uri(uri: str) -> Path | None:
    """Resolve a file URI into a local path when possible."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None

    path_text = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path_text = f"//{parsed.netloc}{path_text}"
    return Path(path_text).resolve(strict=False)


__all__ = ["LspDocumentState", "LspService", "path_from_uri"]
