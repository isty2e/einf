import asyncio

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from einf.analysis.checkers import (
    CheckerExecutor,
    CheckerFailure,
    CheckerResult,
    build_checker_adapters,
)
from einf.analysis.model import DiagnosticSeverity, TextSpan
from einf.analysis.validator.model import ValidationFileReport

from .analysis_queue import DocumentAnalysisRequest, LspAnalysisQueue
from .change_debounce import LspChangeDebouncer, PendingDocumentChange
from .checker_coordinator import DocumentCheckerRequest, LspCheckerCoordinator
from .config import InitializeOptions, LspConfig
from .hover import build_hover
from .inlay_hints import build_inlay_hints
from .position_codec import LspPositionCodec
from .semantic_tokens import TOKEN_MODIFIERS, TOKEN_TYPES, encode_semantic_tokens
from .service import LspDocumentState, LspService

_DEFAULT_CHANGE_DEBOUNCE_SECONDS = 0.15
_DEFAULT_ANALYSIS_WORKER_COUNT = 2
_DEFAULT_ANALYSIS_PENDING_LIMIT = 32


class EinfLanguageServer(LanguageServer):
    """Minimal pygls-based sidecar for einf static analysis."""

    def __init__(self) -> None:
        super().__init__(
            name="einf-lsp",
            version="0.1",
            text_document_sync_kind=lsp.TextDocumentSyncKind.Incremental,
        )
        self.einf_config = LspConfig()
        self.einf_service = LspService(self.einf_config.parser)
        self.einf_change_debouncer = LspChangeDebouncer(
            delay_seconds=_DEFAULT_CHANGE_DEBOUNCE_SECONDS,
            report_failure=self._report_debounced_analysis_failure,
        )
        self.einf_analysis_queue = LspAnalysisQueue(
            worker_count=_DEFAULT_ANALYSIS_WORKER_COUNT,
            pending_limit=_DEFAULT_ANALYSIS_PENDING_LIMIT,
        )
        self.einf_checker_coordinator = _build_checker_coordinator(self.einf_config)
        self.einf_position_encoding: lsp.PositionEncodingKind | str = (
            lsp.PositionEncodingKind.Utf16
        )

    def _report_debounced_analysis_failure(
        self,
        change: PendingDocumentChange,
        error: Exception,
    ) -> None:
        self.window_log_message(
            lsp.LogMessageParams(
                type=lsp.MessageType.Error,
                message=(
                    f"Background analysis failed for {change.uri} "
                    f"at version {change.version}: "
                    f"{type(error).__name__}: {error}"
                ),
            )
        )

    def configure(self, config: LspConfig) -> None:
        """Configure semantic and checker subsystems before document traffic."""
        self.einf_config = config
        self.einf_service = LspService(config.parser)
        self.einf_checker_coordinator = _build_checker_coordinator(config)

    def position_codec(self, state: LspDocumentState) -> LspPositionCodec:
        """Bind the negotiated wire encoding to one document source snapshot."""
        return LspPositionCodec(
            lines=state.source_lines,
            encoding=self.einf_position_encoding,
        )


SEMANTIC_TOKENS_LEGEND = lsp.SemanticTokensLegend(
    token_types=list(TOKEN_TYPES),
    token_modifiers=list(TOKEN_MODIFIERS),
)


def build_server() -> EinfLanguageServer:
    """Build the minimal einf language server instance."""
    server = EinfLanguageServer()

    @server.feature(lsp.INITIALIZE)
    def initialize(ls: EinfLanguageServer, params: lsp.InitializeParams) -> None:
        init_options = _coerce_initialize_options(params.initialization_options)
        ls.configure(LspConfig.from_initialize_options(init_options))
        position_encoding = ls.workspace.position_encoding
        ls.einf_position_encoding = (
            position_encoding
            if position_encoding is not None
            else lsp.PositionEncodingKind.Utf16
        )

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    async def did_open(
        ls: EinfLanguageServer,
        params: lsp.DidOpenTextDocumentParams,
    ) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        await ls.einf_checker_coordinator.cancel(uri=text_document.uri)
        await _analyze_document_request(
            ls,
            request=DocumentAnalysisRequest(
                uri=text_document.uri,
                source=text_document.source,
                version=text_document.version,
            ),
        )

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    async def did_change(
        ls: EinfLanguageServer, params: lsp.DidChangeTextDocumentParams
    ) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        await ls.einf_checker_coordinator.cancel(uri=text_document.uri)
        change = PendingDocumentChange(
            uri=text_document.uri,
            source=text_document.source,
            version=text_document.version,
        )

        async def analyze(pending_change: PendingDocumentChange) -> None:
            await _analyze_document_request(
                ls,
                request=DocumentAnalysisRequest(
                    uri=pending_change.uri,
                    source=pending_change.source,
                    version=pending_change.version,
                ),
            )

        ls.einf_change_debouncer.schedule(change, analyze=analyze)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    async def did_save(
        ls: EinfLanguageServer,
        params: lsp.DidSaveTextDocumentParams,
    ) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        ls.einf_change_debouncer.cancel(text_document.uri)
        await ls.einf_checker_coordinator.cancel(uri=text_document.uri)
        state = await _analyze_document_request(
            ls,
            request=DocumentAnalysisRequest(
                uri=text_document.uri,
                source=text_document.source,
                version=text_document.version,
            ),
        )
        if state is not None:
            await _check_document_state(ls, state=state)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    async def did_close(
        ls: EinfLanguageServer, params: lsp.DidCloseTextDocumentParams
    ) -> None:
        ls.einf_change_debouncer.cancel(params.text_document.uri)
        await asyncio.gather(
            ls.einf_analysis_queue.cancel(uri=params.text_document.uri),
            ls.einf_checker_coordinator.cancel(uri=params.text_document.uri),
        )
        ls.einf_service.close_document(uri=params.text_document.uri)
        ls.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(
                uri=params.text_document.uri,
                diagnostics=[],
            )
        )

    @server.feature(lsp.SHUTDOWN)
    async def shutdown(ls: EinfLanguageServer, *args: object) -> None:
        _ = args
        await ls.einf_change_debouncer.close()
        await asyncio.gather(
            ls.einf_analysis_queue.close(),
            ls.einf_checker_coordinator.close(),
        )

    @server.feature(lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL, SEMANTIC_TOKENS_LEGEND)
    async def semantic_tokens_full(
        ls: EinfLanguageServer,
        params: lsp.SemanticTokensParams,
    ) -> lsp.SemanticTokens:
        state = await _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return lsp.SemanticTokens(data=[])
        position_codec = ls.position_codec(state)
        return lsp.SemanticTokens(
            data=encode_semantic_tokens(
                state.semantic_report.axis_tokens,
                position_codec=position_codec,
            )
        )

    @server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
    async def inlay_hint(
        ls: EinfLanguageServer,
        params: lsp.InlayHintParams,
    ) -> list[lsp.InlayHint]:
        state = await _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return []
        position_codec = ls.position_codec(state)
        return build_inlay_hints(
            axis_tokens=state.semantic_report.axis_tokens,
            visible_range=position_codec.from_lsp_range(params.range),
            position_codec=position_codec,
        )

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    async def hover(
        ls: EinfLanguageServer,
        params: lsp.HoverParams,
    ) -> lsp.Hover | None:
        state = await _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return None
        position_codec = ls.position_codec(state)
        return build_hover(
            axis_tokens=state.semantic_report.axis_tokens,
            position=position_codec.from_lsp_position(params.position),
        )

    return server


def _coerce_initialize_options(
    initialize_options: object,
) -> InitializeOptions | None:
    if initialize_options is None:
        return None
    if not isinstance(initialize_options, dict):
        return None

    options: dict[str, str | int | float | list[str] | None] = {}
    parser_value = initialize_options.get("parser")
    if parser_value is None or isinstance(parser_value, str):
        options["parser"] = parser_value

    checker_value = initialize_options.get("checkers")
    if isinstance(checker_value, list):
        string_values = [value for value in checker_value if isinstance(value, str)]
        options["checkers"] = string_values
    elif checker_value is None or isinstance(checker_value, str):
        options["checkers"] = checker_value

    timeout_value = initialize_options.get("checkerTimeoutSeconds")
    if isinstance(timeout_value, (int, float)) and not isinstance(timeout_value, bool):
        options["checkerTimeoutSeconds"] = timeout_value

    concurrency_value = initialize_options.get("checkerMaxConcurrency")
    if type(concurrency_value) is int:
        options["checkerMaxConcurrency"] = concurrency_value
    return options


def _publish_document_state(
    ls: EinfLanguageServer,
    state: LspDocumentState,
) -> None:
    report = state.report
    diagnostics = (
        _build_diagnostics(
            report,
            position_codec=ls.position_codec(state),
        )
        if report.has_errors()
        else []
    )
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=state.uri,
            diagnostics=diagnostics,
            version=state.version,
        )
    )


async def _get_or_open_document_state(
    ls: EinfLanguageServer,
    *,
    uri: str,
) -> LspDocumentState | None:
    pending_change = ls.einf_change_debouncer.take_pending(uri=uri)
    if pending_change is not None:
        return await _analyze_document_request(
            ls,
            request=DocumentAnalysisRequest(
                uri=pending_change.uri,
                source=pending_change.source,
                version=pending_change.version,
            ),
        )

    latest_state = await ls.einf_analysis_queue.wait_for_latest(uri=uri)
    if latest_state is not None:
        return latest_state

    state = ls.einf_service.get_document_state(uri=uri)
    if state is not None:
        return state
    text_document = ls.workspace.get_text_document(uri)
    return await _analyze_document_request(
        ls,
        request=DocumentAnalysisRequest(
            uri=text_document.uri,
            source=text_document.source,
            version=text_document.version,
        ),
    )


async def _analyze_document_request(
    ls: EinfLanguageServer,
    *,
    request: DocumentAnalysisRequest,
) -> LspDocumentState | None:
    service = ls.einf_service

    def analyze(analysis_request: DocumentAnalysisRequest) -> LspDocumentState:
        return service.analyze_document(
            uri=analysis_request.uri,
            source=analysis_request.source,
            version=analysis_request.version,
        )

    def commit(state: LspDocumentState) -> None:
        service.commit_document_state(state)
        _publish_document_state(ls, state)

    return await ls.einf_analysis_queue.analyze(
        request,
        analyze=analyze,
        commit=commit,
    )


async def _check_document_state(
    ls: EinfLanguageServer,
    *,
    state: LspDocumentState,
) -> CheckerResult | None:
    if state.path is None:
        return None

    service = ls.einf_service

    def commit(request: DocumentCheckerRequest, result: CheckerResult) -> bool:
        current = service.get_document_state(uri=request.uri)
        if (
            current is None
            or current.path != request.path
            or current.version != request.version
        ):
            return False
        updated = current.with_checker_result(result)
        service.commit_document_state(updated)
        _publish_document_state(ls, updated)
        _report_checker_failures(ls, result.failures)
        return True

    return await ls.einf_checker_coordinator.check(
        DocumentCheckerRequest(
            uri=state.uri,
            path=state.path,
            version=state.version,
        ),
        commit=commit,
    )


def _build_checker_coordinator(config: LspConfig) -> LspCheckerCoordinator:
    return LspCheckerCoordinator(
        adapters=build_checker_adapters(config.checkers),
        executor=CheckerExecutor(config.checker_execution_policy),
    )


def _build_diagnostics(
    report: ValidationFileReport,
    *,
    position_codec: LspPositionCodec,
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for diagnostic in report.diagnostics:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(
                    diagnostic.span,
                    position_codec=position_codec,
                ),
                message=diagnostic.message,
                severity=_diagnostic_severity(diagnostic.severity),
                code=diagnostic.code,
                source="einf",
            )
        )
    for checker_diagnostic in report.checker_diagnostics:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(
                    checker_diagnostic.span,
                    position_codec=position_codec,
                ),
                message=checker_diagnostic.message,
                severity=_diagnostic_severity(checker_diagnostic.severity),
                code=checker_diagnostic.code,
                source=checker_diagnostic.tool,
            )
        )
    for failure in report.failures:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(
                    failure.span,
                    position_codec=position_codec,
                ),
                message=failure.message,
                severity=lsp.DiagnosticSeverity.Error,
                code=failure.kind,
                source="einf-validator",
            )
        )
    return diagnostics


def _diagnostic_severity(severity: DiagnosticSeverity) -> lsp.DiagnosticSeverity:
    match severity:
        case "warning":
            return lsp.DiagnosticSeverity.Warning
        case "info":
            return lsp.DiagnosticSeverity.Information
        case _:
            return lsp.DiagnosticSeverity.Error


def _range_from_span(
    span: TextSpan | None,
    *,
    position_codec: LspPositionCodec,
) -> lsp.Range:
    if span is None:
        return lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=0, character=1),
        )
    return position_codec.to_lsp_range(span)


def _report_checker_failures(
    ls: EinfLanguageServer,
    checker_failures: tuple[CheckerFailure, ...],
) -> None:
    if not checker_failures:
        return

    summaries: list[str] = []
    for checker_failure in checker_failures:
        ls.window_log_message(
            lsp.LogMessageParams(
                type=lsp.MessageType.Warning,
                message=(
                    f"[{checker_failure.tool}] "
                    f"{checker_failure.kind}: {checker_failure.message}"
                ),
            )
        )
        summaries.append(f"[{checker_failure.tool}] {checker_failure.kind}")
    ls.window_show_message(
        lsp.ShowMessageParams(
            type=lsp.MessageType.Warning,
            message=(
                "Fallback checker coverage is incomplete:\n"
                + "\n".join(summaries)
                + "\nSee the einf LSP logs for details."
            ),
        )
    )


__all__ = ["SEMANTIC_TOKENS_LEGEND", "EinfLanguageServer", "build_server"]
