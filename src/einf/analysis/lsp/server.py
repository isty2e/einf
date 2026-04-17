from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from einf.analysis.checkers import CheckerFailure
from einf.analysis.model import DiagnosticSeverity, TextPosition, TextSpan
from einf.analysis.validator.model import ValidationFileReport

from .config import InitializeOptions, LspConfig
from .hover import build_hover
from .inlay_hints import build_inlay_hints
from .semantic_tokens import TOKEN_MODIFIERS, TOKEN_TYPES, encode_semantic_tokens
from .service import LspDocumentState, LspService


class EinfLanguageServer(LanguageServer):
    """Minimal pygls-based sidecar for einf static analysis."""

    def __init__(self) -> None:
        super().__init__(
            name="einf-lsp",
            version="0.1",
            text_document_sync_kind=lsp.TextDocumentSyncKind.Incremental,
        )
        self.einf_service = LspService(LspConfig())


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
        ls.einf_service = LspService(LspConfig.from_initialize_options(init_options))

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls: EinfLanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        state = ls.einf_service.open_document(
            uri=text_document.uri,
            source=text_document.source,
            version=text_document.version,
        )
        _publish_document_state(ls, state)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(
        ls: EinfLanguageServer, params: lsp.DidChangeTextDocumentParams
    ) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        state = ls.einf_service.change_document(
            uri=text_document.uri,
            source=text_document.source,
            version=text_document.version,
        )
        _publish_document_state(ls, state)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def did_save(ls: EinfLanguageServer, params: lsp.DidSaveTextDocumentParams) -> None:
        text_document = ls.workspace.get_text_document(params.text_document.uri)
        state = ls.einf_service.save_document(
            uri=text_document.uri,
            source=text_document.source,
            version=text_document.version,
        )
        _publish_document_state(ls, state)
        _log_checker_failures(ls, state.checker_failures)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(
        ls: EinfLanguageServer, params: lsp.DidCloseTextDocumentParams
    ) -> None:
        ls.einf_service.close_document(uri=params.text_document.uri)
        ls.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(
                uri=params.text_document.uri,
                diagnostics=[],
            )
        )

    @server.feature(lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL, SEMANTIC_TOKENS_LEGEND)
    def semantic_tokens_full(
        ls: EinfLanguageServer,
        params: lsp.SemanticTokensParams,
    ) -> lsp.SemanticTokens:
        state = _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return lsp.SemanticTokens(data=[])
        return lsp.SemanticTokens(data=encode_semantic_tokens(state.report.axis_tokens))

    @server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
    def inlay_hint(
        ls: EinfLanguageServer,
        params: lsp.InlayHintParams,
    ) -> list[lsp.InlayHint]:
        state = _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return []
        return build_inlay_hints(
            axis_tokens=state.report.axis_tokens,
            visible_range=_span_from_lsp_range(params.range),
        )

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(
        ls: EinfLanguageServer,
        params: lsp.HoverParams,
    ) -> lsp.Hover | None:
        state = _get_or_open_document_state(ls, uri=params.text_document.uri)
        if state is None:
            return None
        return build_hover(
            axis_tokens=state.report.axis_tokens,
            position=_text_position_from_lsp_position(params.position),
        )

    return server


def _coerce_initialize_options(
    initialize_options: object,
) -> InitializeOptions | None:
    if initialize_options is None:
        return None
    if not isinstance(initialize_options, dict):
        return None

    options: dict[str, str | list[str] | None] = {}
    parser_value = initialize_options.get("parser")
    if parser_value is None or isinstance(parser_value, str):
        options["parser"] = parser_value

    checker_value = initialize_options.get("checkers")
    if isinstance(checker_value, list):
        string_values = [value for value in checker_value if isinstance(value, str)]
        options["checkers"] = string_values
    elif checker_value is None or isinstance(checker_value, str):
        options["checkers"] = checker_value
    return options


def _publish_document_state(
    ls: EinfLanguageServer,
    state: LspDocumentState,
) -> None:
    diagnostics = _build_diagnostics(state.report)
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=state.uri,
            diagnostics=diagnostics,
            version=state.version,
        )
    )


def _get_or_open_document_state(
    ls: EinfLanguageServer,
    *,
    uri: str,
) -> LspDocumentState | None:
    state = ls.einf_service.get_document_state(uri=uri)
    if state is not None:
        return state
    try:
        text_document = ls.workspace.get_text_document(uri)
    except Exception:
        return None
    return ls.einf_service.open_document(
        uri=text_document.uri,
        source=text_document.source,
        version=text_document.version,
    )


def _build_diagnostics(report: ValidationFileReport) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for diagnostic in report.diagnostics:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(diagnostic.span),
                message=diagnostic.message,
                severity=_diagnostic_severity(diagnostic.severity),
                code=diagnostic.code,
                source="einf",
            )
        )
    for checker_diagnostic in report.checker_diagnostics:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(checker_diagnostic.span),
                message=checker_diagnostic.message,
                severity=_diagnostic_severity(checker_diagnostic.severity),
                code=checker_diagnostic.code,
                source=checker_diagnostic.tool,
            )
        )
    for failure in report.failures:
        diagnostics.append(
            lsp.Diagnostic(
                range=_range_from_span(failure.span),
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


def _range_from_span(span: TextSpan | None) -> lsp.Range:
    if span is None:
        return lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=0, character=1),
        )
    return lsp.Range(
        start=lsp.Position(
            line=span.start.line - 1,
            character=span.start.column,
        ),
        end=lsp.Position(
            line=span.end.line - 1,
            character=span.end.column,
        ),
    )


def _text_position_from_lsp_position(position: lsp.Position) -> TextPosition:
    return TextPosition(line=position.line + 1, column=position.character)


def _span_from_lsp_range(lsp_range: lsp.Range) -> TextSpan:
    return TextSpan(
        start=_text_position_from_lsp_position(lsp_range.start),
        end=_text_position_from_lsp_position(lsp_range.end),
    )


def _log_checker_failures(
    ls: EinfLanguageServer,
    checker_failures: tuple[CheckerFailure, ...],
) -> None:
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


__all__ = ["EinfLanguageServer", "SEMANTIC_TOKENS_LEGEND", "build_server"]
