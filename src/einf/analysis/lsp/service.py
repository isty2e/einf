from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from einf.analysis.checkers import CheckerResult
from einf.analysis.engine import analyze_source
from einf.analysis.parser import ParserBackend, ParserUnavailableError
from einf.analysis.parser.factory import build_parser_backend
from einf.analysis.report import (
    AnalysisFailure,
    AnalysisFileReport,
    parser_unavailable_report,
)


@dataclass(frozen=True, slots=True)
class LspDocumentState:
    """Current in-memory LSP analysis snapshot for one document URI."""

    uri: str
    path: Path | None
    version: int | None
    source: str
    semantic_report: AnalysisFileReport
    checker_result: CheckerResult | None = None
    _source_lines: tuple[str, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.semantic_report.checker_diagnostics:
            raise ValueError("semantic report cannot contain checker diagnostics")

    @property
    def source_lines(self) -> tuple[str, ...]:
        """Return lazily indexed lines for this immutable source snapshot."""
        existing = self._source_lines
        if existing is not None:
            return existing
        lines = tuple(self.source.splitlines(keepends=True))
        object.__setattr__(self, "_source_lines", lines)
        return lines

    @property
    def report(self) -> AnalysisFileReport:
        """Project semantic and checker state into one LSP-facing report."""
        if self.checker_result is None or self.path is None:
            return self.semantic_report

        checker_diagnostics = self.checker_result.diagnostics_for(self.path)
        if not checker_diagnostics:
            return self.semantic_report
        return replace(
            self.semantic_report,
            checker_diagnostics=checker_diagnostics,
        )

    def with_checker_result(self, result: CheckerResult) -> "LspDocumentState":
        """Return this document state with one fresh checker result."""
        return replace(self, checker_result=result)


class LspService:
    """Pure document-analysis service used by the pygls transport layer.

    `einf` semantic analysis is the primary responsibility. External checker
    execution is an optional save-boundary fallback for single-server editor
    setups, not the preferred integration path.
    """

    def __init__(self, parser: str = "ast") -> None:
        self._parser = parser
        self._parser_backend: ParserBackend = build_parser_backend(parser)
        try:
            self._parser_backend.validate_available()
        except ParserUnavailableError as error:
            self._parser_unavailable_error: ParserUnavailableError | None = error
        else:
            self._parser_unavailable_error = None
        self._states: dict[str, LspDocumentState] = {}

    @property
    def parser(self) -> str:
        """Return the configured semantic parser name."""
        return self._parser

    def open_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Analyze an opened document without running external checkers."""
        state = self.analyze_document(
            uri=uri,
            source=source,
            version=version,
        )
        self.commit_document_state(state)
        return state

    def change_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Reanalyze one changed document and clear stale checker state."""
        state = self.analyze_document(
            uri=uri,
            source=source,
            version=version,
        )
        self.commit_document_state(state)
        return state

    def analyze_document(
        self,
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> LspDocumentState:
        """Analyze one document without making the result session-visible."""
        path = path_from_uri(uri)
        report = self._analyze_document(path=path, source=source)
        return LspDocumentState(
            uri=uri,
            path=path,
            version=version,
            source=source,
            semantic_report=report,
        )

    def commit_document_state(self, state: LspDocumentState) -> None:
        """Make one completed analysis result visible to session readers."""
        self._states[state.uri] = state

    def close_document(self, *, uri: str) -> None:
        """Drop cached state for one closed document."""
        self._states.pop(uri, None)

    def get_document_state(self, *, uri: str) -> LspDocumentState | None:
        """Return the cached document state, if available."""
        return self._states.get(uri)

    def _analyze_document(
        self,
        *,
        path: Path | None,
        source: str,
    ) -> AnalysisFileReport:
        if path is None:
            path = _IN_MEMORY_PATH
        if self._parser_unavailable_error is not None:
            return parser_unavailable_report(
                path=path,
                error=self._parser_unavailable_error,
            )
        if not _source_may_contain_einf_calls(source):
            return _empty_file_report(path=str(path))
        return analyze_source(
            source=source,
            path=path,
            parser_backend=self._parser_backend,
        )


_IN_MEMORY_PATH = Path("<in-memory>")


def path_from_uri(uri: str) -> Path | None:
    """Resolve a file URI into a local path when possible."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None

    path_text = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path_text = f"//{parsed.netloc}{path_text}"
    return Path(path_text).resolve(strict=False)


def _source_may_contain_einf_calls(source: str) -> bool:
    """Return whether source has the lexical marker used by supported syntax."""
    return "einf" in source


def _empty_file_report(
    *,
    path: str,
    failures: tuple[AnalysisFailure, ...] = (),
) -> AnalysisFileReport:
    return AnalysisFileReport(
        path=path,
        diagnostics=(),
        checker_diagnostics=(),
        axis_tokens=(),
        failures=failures,
    )


__all__ = ["LspDocumentState", "LspService", "path_from_uri"]
