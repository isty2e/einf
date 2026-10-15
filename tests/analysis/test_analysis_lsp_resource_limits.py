import asyncio
from pathlib import Path

import pytest

lsprotocol = pytest.importorskip("lsprotocol")
pygls = pytest.importorskip("pygls")
_ = lsprotocol, pygls
from lsprotocol import types as lsp
from pygls.exceptions import JsonRpcInvalidParams

from einf.analysis.checkers import CheckerResult
from einf.analysis.lsp import server as lsp_server
from einf.analysis.lsp.analysis_limits import LspAnalysisLimitExceeded
from einf.analysis.lsp.analysis_process import LspAnalysisProcess
from einf.analysis.lsp.analysis_queue import DocumentAnalysisRequest
from einf.analysis.lsp.server import EinfLanguageServer, build_server
from einf.analysis.lsp.service import LspDocumentState, LspService


def _initialize(options: object = None) -> EinfLanguageServer:
    server = build_server()
    initialization = server.protocol.lsp_initialize(
        lsp.InitializeParams(
            capabilities=lsp.ClientCapabilities(),
            root_uri="file:///workspace",
            initialization_options=options,
        )
    )
    handler, args, kwargs = next(initialization)
    handler(*args, **(kwargs or {}))
    with pytest.raises(StopIteration):
        next(initialization)
    return server


async def _open(server: EinfLanguageServer, uri: str, source: str) -> None:
    params = lsp.DidOpenTextDocumentParams(
        text_document=lsp.TextDocumentItem(
            uri=uri, language_id="python", version=1, text=source
        )
    )
    server.workspace.put_text_document(params.text_document)
    await server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN](params)


async def _change(
    server: EinfLanguageServer, uri: str, source: str, version: int
) -> None:
    change = lsp.TextDocumentContentChangeWholeDocument(text=source)
    params = lsp.DidChangeTextDocumentParams(
        text_document=lsp.VersionedTextDocumentIdentifier(uri=uri, version=version),
        content_changes=[change],
    )
    server.workspace.update_text_document(params.text_document, change)
    await server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_CHANGE](params)


async def _close(server: EinfLanguageServer, uri: str) -> None:
    server.workspace.remove_text_document(uri)
    await server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_CLOSE](
        lsp.DidCloseTextDocumentParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri)
        )
    )


async def _shutdown(server: EinfLanguageServer) -> None:
    await server.protocol.fm.features[lsp.SHUTDOWN]()


@pytest.fixture
def analyzed(monkeypatch: pytest.MonkeyPatch) -> list[DocumentAnalysisRequest]:
    requests: list[DocumentAnalysisRequest] = []

    async def analyze(
        self: LspAnalysisProcess, request: DocumentAnalysisRequest, *, parser: str
    ) -> LspDocumentState:
        requests.append(request)
        return LspService(parser).analyze_document(
            uri=request.uri, source=request.source, version=request.version
        )

    monkeypatch.setattr(LspAnalysisProcess, "analyze", analyze)
    return requests


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[lsp.PublishDiagnosticsParams]:
    notifications: list[lsp.PublishDiagnosticsParams] = []
    monkeypatch.setattr(
        EinfLanguageServer,
        "text_document_publish_diagnostics",
        lambda self, params: notifications.append(params),
    )
    return notifications


@pytest.mark.parametrize(
    "option",
    ["analysisMaxDocuments", "analysisMaxDocumentBytes", "analysisMaxSourceBytes"],
)
@pytest.mark.parametrize("value", [False, True, 0, -1, 1.0, "3", [], {}])
def test_initialize_rejects_invalid_explicit_analysis_limits(
    option: str, value: object
) -> None:
    with pytest.raises(JsonRpcInvalidParams, match=option):
        _initialize({option: value})


def test_default_server_never_reserves_or_scans_source(
    monkeypatch: pytest.MonkeyPatch,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize()
        monkeypatch.setattr(
            server.einf_source_budget,
            "reserve",
            lambda **kwargs: pytest.fail("default admission must not reserve"),
        )
        try:
            await _open(server, "untitled:default", "# é🙂\n")
            await _change(server, "untitled:default", "# changed\n", 2)
            state = await lsp_server._get_or_open_document_state(
                server, uri="untitled:default"
            )
            assert state is not None and state.version == 2
            assert len(analyzed) == 2
            assert published[-1].diagnostics == []
        finally:
            await _shutdown(server)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "option", ["analysisMaxDocumentBytes", "analysisMaxSourceBytes"]
)
def test_utf8_refusal_clears_state_blocks_save_and_recovers(
    option: str,
    monkeypatch: pytest.MonkeyPatch,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({option: 8})
        uri = "file:///outside/sample.py"
        checks: list[LspDocumentState] = []

        async def check(ls: EinfLanguageServer, *, state: LspDocumentState) -> None:
            checks.append(state)

        monkeypatch.setattr(lsp_server, "_check_document_state", check)
        try:
            await _open(server, uri, "# é🙂")  # Eight UTF-8 bytes.
            assert server.einf_service.get_document_state(uri=uri) is not None
            await _change(server, uri, "# é🙂x", 2)
            assert server.workspace.get_text_document(uri).source == "# é🙂x"
            assert server.einf_service.get_document_state(uri=uri) is None
            assert len(analyzed) == 1
            refusal = published[-1]
            assert refusal.version == 2
            assert [item.code for item in refusal.diagnostics] == [
                "analysis_unavailable"
            ]
            assert "9 > 8" in refusal.diagnostics[0].message
            assert refusal.diagnostics[0].severity == lsp.DiagnosticSeverity.Warning
            await server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_SAVE](
                lsp.DidSaveTextDocumentParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri)
                )
            )
            assert checks == [] and len(analyzed) == 1
            await _change(server, uri, "# é", 3)
            state = await lsp_server._get_or_open_document_state(server, uri=uri)
            assert state is not None and state.version == 3
            assert published[-1].diagnostics == []
            assert len(analyzed) == 2
        finally:
            await _shutdown(server)
        server.einf_source_budget.reserve(
            uri="untitled:new", source="12345678"
        ).release()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "option,maximum", [("analysisMaxDocuments", 1), ("analysisMaxSourceBytes", 6)]
)
def test_limits_do_not_evict_other_documents_and_close_releases_capacity(
    option: str,
    maximum: int,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({option: maximum})
        first, second = "untitled:one", "file:///outside/two.py"
        try:
            await _open(server, first, "# 1234")
            original = server.einf_service.get_document_state(uri=first)
            await _open(server, second, "# 5678")
            assert len(analyzed) == 1
            assert published[-1].diagnostics[0].code == "analysis_unavailable"
            assert server.einf_service.get_document_state(uri=first) is original
            await _close(server, first)
            state = await lsp_server._get_or_open_document_state(server, uri=second)
            assert state is not None and state.uri == second
            assert len(analyzed) == 2
        finally:
            await _shutdown(server)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method",
    [
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.TEXT_DOCUMENT_INLAY_HINT,
        lsp.TEXT_DOCUMENT_HOVER,
    ],
)
def test_editor_request_flushes_pending_capture_without_double_charge(
    method: str,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        source = "from einf import ax, rearrange\nrearrange(ax[a], ax[a])\n"
        server = _initialize({"analysisMaxSourceBytes": len(source)})
        uri = "untitled:flush"
        try:
            await _open(server, uri, "# old")
            await _change(server, uri, source, 2)
            identifier = lsp.TextDocumentIdentifier(uri=uri)
            if method == lsp.TEXT_DOCUMENT_HOVER:
                params = lsp.HoverParams(
                    text_document=identifier,
                    position=lsp.Position(line=1, character=13),
                )
            elif method == lsp.TEXT_DOCUMENT_INLAY_HINT:
                params = lsp.InlayHintParams(
                    text_document=identifier,
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=3, character=0),
                    ),
                )
            else:
                params = lsp.SemanticTokensParams(text_document=identifier)
            await server.protocol.fm.features[method](params)
            assert len(analyzed) == 2
            state = server.einf_service.get_document_state(uri=uri)
            assert state is not None and state.source == source
            assert server.einf_change_debouncer.has_pending(uri=uri) is False
            with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
                server.einf_source_budget.reserve(uri="untitled:other", source="x")
        finally:
            await _shutdown(server)
        server.einf_source_budget.reserve(uri="untitled:other", source=source).release()

    asyncio.run(scenario())


def test_unopened_file_request_obeys_limits_without_workspace_containment(
    tmp_path: Path,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxDocumentBytes": 6})
        target = tmp_path / "outside.py"
        target.write_text("# large file", encoding="utf-8")
        try:
            assert (
                await lsp_server._get_or_open_document_state(
                    server, uri=target.as_uri()
                )
                is None
            )
            assert (
                not analyzed
                and published[-1].diagnostics[0].code == "analysis_unavailable"
            )
            target.write_text("# ok", encoding="utf-8")
            state = await lsp_server._get_or_open_document_state(
                server, uri=target.as_uri()
            )
            assert state is not None and state.source == "# ok"
            assert len(analyzed) == 1
        finally:
            await _shutdown(server)

    asyncio.run(scenario())


def test_refused_edit_cannot_be_overwritten_by_running_analysis(
    monkeypatch: pytest.MonkeyPatch,
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxDocumentBytes": 6})
        started, retiring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        uri = "untitled:stale"

        async def execute(
            self: LspAnalysisProcess, request: DocumentAnalysisRequest, *, parser: str
        ) -> LspDocumentState:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                retiring.set()
                await release.wait()
            return LspService(parser).analyze_document(
                uri=request.uri, source=request.source, version=request.version
            )

        monkeypatch.setattr(LspAnalysisProcess, "analyze", execute)
        opened = asyncio.create_task(_open(server, uri, "# old"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await _change(server, uri, "# too large", 2)
            await asyncio.wait_for(retiring.wait(), timeout=2)
            assert await opened is None
            release.set()
            await server.einf_analysis_queue.wait_for_latest(uri=uri)
            assert server.einf_service.get_document_state(uri=uri) is None
            assert [(p.version, p.diagnostics[0].code) for p in published] == [
                (2, "analysis_unavailable")
            ]
        finally:
            release.set()
            await _shutdown(server)
            await opened

    asyncio.run(scenario())


def test_superseded_admission_does_not_cancel_newer_capture(
    monkeypatch: pytest.MonkeyPatch,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxDocumentBytes": 20})
        uri = "untitled:race"
        await _open(server, uri, "# first")
        blocked, release = asyncio.Event(), asyncio.Event()
        cancel = server.einf_checker_coordinator.cancel
        calls = 0

        async def suspend(*, uri: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                blocked.set()
                await release.wait()
            await cancel(uri=uri)

        monkeypatch.setattr(server.einf_checker_coordinator, "cancel", suspend)
        older = asyncio.create_task(_change(server, uri, "# second", 2))
        try:
            await asyncio.wait_for(blocked.wait(), timeout=2)
            await _change(server, uri, "# third", 3)
            release.set()
            await older
            state = await lsp_server._get_or_open_document_state(server, uri=uri)
            assert state is not None and state.version == 3
            assert [r.version for r in analyzed] == [1, 3]
            assert [p.version for p in published] == [1, 3]
        finally:
            release.set()
            await older
            await _shutdown(server)

    asyncio.run(scenario())


def test_save_replaces_pending_capture_and_shutdown_releases_it(
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxSourceBytes": 6})
        uri = "file:///outside/save.py"
        try:
            await _open(server, uri, "# old")
            await _change(server, uri, "# new!", 2)
            await server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_SAVE](
                lsp.DidSaveTextDocumentParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri)
                )
            )
            assert [request.version for request in analyzed] == [1, 2]
            assert published[-1].diagnostics == []
            assert not server.einf_change_debouncer.has_pending(uri=uri)
            await _change(server, uri, "# last", 3)
        finally:
            await _shutdown(server)
        server.einf_source_budget.reserve(uri="untitled:new", source="123456").release()

    asyncio.run(scenario())


def test_invalid_utf8_source_is_explicitly_unavailable(
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxDocumentBytes": 100})
        try:
            await _open(server, "untitled:encoding", "# \ud800")
            assert analyzed == []
            assert published[-1].diagnostics[0].code == "analysis_unavailable"
            assert "valid UTF-8" in published[-1].diagnostics[0].message
            await _change(server, "untitled:encoding", "# valid", 2)
            state = await lsp_server._get_or_open_document_state(
                server, uri="untitled:encoding"
            )
            assert state is not None and state.version == 2
        finally:
            await _shutdown(server)

    asyncio.run(scenario())


def test_checker_commit_rejects_state_obsoleted_during_admission(
    monkeypatch: pytest.MonkeyPatch,
    analyzed: list[DocumentAnalysisRequest],
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        server = _initialize({"analysisMaxDocuments": 1})
        uri = "file:///outside/checker.py"
        try:
            await _open(server, uri, "# old")
            state = server.einf_service.get_document_state(uri=uri)
            assert state is not None

            async def check(request, *, commit):
                # pygls updates text before the application handles didChange.
                server.workspace.update_text_document(
                    lsp.VersionedTextDocumentIdentifier(uri=uri, version=2),
                    lsp.TextDocumentContentChangeWholeDocument(text="# new"),
                )
                assert (
                    commit(request, CheckerResult(diagnostics=(), failures=())) is False
                )

            monkeypatch.setattr(server.einf_checker_coordinator, "check", check)
            await lsp_server._check_document_state(server, state=state)
            assert [p.version for p in published] == [1]
        finally:
            await _shutdown(server)

    asyncio.run(scenario())


def test_real_worker_analysis_recovers_after_resource_refusal(
    published: list[lsp.PublishDiagnosticsParams],
) -> None:
    async def scenario() -> None:
        source = (
            'from einf import ax, axes, reduce\nb, n, z = axes("b", "n", "z")\n'
            "reduce(ax[b, n], ax[b, z])\n"
        )
        server = _initialize({"analysisMaxDocumentBytes": len(source)})
        uri = "untitled:real-worker.py"
        try:
            await _open(server, uri, source + "# over budget")
            assert published[-1].diagnostics[0].code == "analysis_unavailable"
            await _change(server, uri, source, 2)
            state = await lsp_server._get_or_open_document_state(server, uri=uri)
            assert state is not None and state.version == 2
            assert state.semantic_report.diagnostics
            assert state.semantic_report.axis_tokens
            assert all(
                diagnostic.code != "analysis_unavailable"
                for diagnostic in published[-1].diagnostics
            )
        finally:
            await _shutdown(server)

    asyncio.run(scenario())
