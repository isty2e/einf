import asyncio
import sys
import threading
from pathlib import Path

import pytest

pygls = pytest.importorskip("pygls")
_ = pygls

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerDiagnostic,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.lsp.analysis_queue import DocumentAnalysisRequest
from einf.analysis.lsp.checker_coordinator import LspCheckerCoordinator
from einf.analysis.lsp.server import (
    EinfLanguageServer,
    _analyze_document_request,
    _check_document_state,
    build_server,
)
from einf.analysis.lsp.service import LspDocumentState, LspService
from einf.analysis.validator.model import ValidationFileReport


class _NoopAdapter(CheckerAdapter):
    name = "stub"
    executable = sys.executable

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        _ = request
        return (sys.executable, "-c", "")

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr, request
        return CheckerResult(diagnostics=(), failures=())


class _ResultExecutor(CheckerExecutor):
    def __init__(self, result: CheckerResult) -> None:
        super().__init__(CheckerExecutionPolicy())
        self._result = result

    async def run_all(
        self,
        adapters: tuple[CheckerAdapter, ...],
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = adapters, request
        return self._result


def _coordinator(result: CheckerResult) -> LspCheckerCoordinator:
    return LspCheckerCoordinator(
        adapters=(_NoopAdapter(),),
        executor=_ResultExecutor(result),
    )


def _state(request: DocumentAnalysisRequest) -> LspDocumentState:
    return LspDocumentState(
        uri=request.uri,
        path=Path(request.uri.removeprefix("file://")),
        version=request.version,
        source=request.source,
        semantic_report=ValidationFileReport(
            path=request.uri,
            diagnostics=(),
            checker_diagnostics=(),
            axis_tokens=(),
            failures=(),
        ),
    )


async def _wait_until_set(event: threading.Event) -> None:
    for _ in range(1_000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("worker did not reach the blocking analysis")


def test_build_server_returns_language_server() -> None:
    server = build_server()

    assert isinstance(server, EinfLanguageServer)
    assert server.einf_config.parser == "ast"
    assert server.einf_config.checkers == ()
    assert server.einf_service.parser == "ast"
    assert server.einf_checker_coordinator.enabled is False


def test_server_discards_stale_result_before_commit_and_publish(monkeypatch) -> None:
    async def scenario() -> None:
        stale_started = threading.Event()
        release_stale = threading.Event()
        published_versions: list[int | None] = []
        server = EinfLanguageServer()
        service = server.einf_service

        def analyze_document(
            *,
            uri: str,
            source: str,
            version: int | None,
        ) -> LspDocumentState:
            request = DocumentAnalysisRequest(
                uri=uri,
                source=source,
                version=version,
            )
            if version == 1:
                stale_started.set()
                assert release_stale.wait(timeout=2)
            return _state(request)

        monkeypatch.setattr(service, "analyze_document", analyze_document)
        monkeypatch.setattr(
            server,
            "text_document_publish_diagnostics",
            lambda params: published_versions.append(params.version),
        )

        stale_task = asyncio.create_task(
            _analyze_document_request(
                server,
                request=DocumentAnalysisRequest(
                    uri="file:///sample.py",
                    source="old",
                    version=1,
                ),
            )
        )
        latest_task: asyncio.Task[LspDocumentState | None] | None = None
        latest_state: LspDocumentState | None = None
        try:
            await _wait_until_set(stale_started)
            latest_task = asyncio.create_task(
                _analyze_document_request(
                    server,
                    request=DocumentAnalysisRequest(
                        uri="file:///sample.py",
                        source="new",
                        version=2,
                    ),
                )
            )
            await asyncio.sleep(0)

            assert service.get_document_state(uri="file:///sample.py") is None
            assert published_versions == []
        finally:
            release_stale.set()
            stale_state = await stale_task
            if latest_task is not None:
                latest_state = await latest_task
            await server.einf_analysis_queue.close()

        assert stale_state is None
        assert latest_state is not None
        assert service.get_document_state(uri="file:///sample.py") == latest_state
        assert published_versions == [2]

    asyncio.run(scenario())


def test_server_commits_and_publishes_checker_result_for_current_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "sample.py"
        source = (
            "from einf import ax, axes, rearrange\n"
            'b = axes("b")[0]\n'
            "rearrange(ax[b], ax[b])\n"
        )
        diagnostic = CheckerDiagnostic(
            tool="stub",
            path=path.resolve(),
            code="stub-rule",
            message="checker diagnostic",
            severity="warning",
            span=None,
        )
        result = CheckerResult(diagnostics=(diagnostic,), failures=())
        server = EinfLanguageServer()
        server.einf_service = LspService()
        server.einf_checker_coordinator = _coordinator(result)
        state = server.einf_service.open_document(
            uri=path.resolve().as_uri(),
            source=source,
            version=1,
        )
        published_diagnostic_counts: list[int] = []
        monkeypatch.setattr(
            server,
            "text_document_publish_diagnostics",
            lambda params: published_diagnostic_counts.append(len(params.diagnostics)),
        )

        checked = await _check_document_state(server, state=state)
        committed = server.einf_service.get_document_state(uri=state.uri)
        await server.einf_checker_coordinator.close()

        assert checked == result
        assert committed is not None
        assert committed.checker_result == result
        assert committed.report.checker_diagnostics == (diagnostic,)
        assert published_diagnostic_counts == [1]

    asyncio.run(scenario())


def test_server_rejects_checker_result_for_stale_document_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "sample.py"
        result = CheckerResult(diagnostics=(), failures=())
        server = EinfLanguageServer()
        server.einf_service = LspService()
        server.einf_checker_coordinator = _coordinator(result)
        stale_state = server.einf_service.open_document(
            uri=path.resolve().as_uri(),
            source="value = 1\n",
            version=1,
        )
        current_state = server.einf_service.change_document(
            uri=stale_state.uri,
            source="value = 2\n",
            version=2,
        )
        published_versions: list[int | None] = []
        monkeypatch.setattr(
            server,
            "text_document_publish_diagnostics",
            lambda params: published_versions.append(params.version),
        )

        checked = await _check_document_state(server, state=stale_state)
        committed = server.einf_service.get_document_state(uri=stale_state.uri)
        await server.einf_checker_coordinator.close()

        assert checked is None
        assert committed == current_state
        assert committed is not None
        assert committed.checker_result is None
        assert published_versions == []

    asyncio.run(scenario())
