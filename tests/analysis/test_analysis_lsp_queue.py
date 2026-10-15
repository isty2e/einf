import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Never

import pytest

from einf.analysis.lsp.analysis_process import (
    AnalysisProcessCleanupError,
    LspAnalysisProcess,
)
from einf.analysis.lsp.analysis_queue import (
    DocumentAnalysisRequest,
    LspAnalysisQueue,
)
from einf.analysis.lsp.service import LspDocumentState
from einf.analysis.report import AnalysisFileReport


def _state(request: DocumentAnalysisRequest) -> LspDocumentState:
    return LspDocumentState(
        uri=request.uri,
        path=Path(request.uri.removeprefix("file://")),
        version=request.version,
        source=request.source,
        semantic_report=AnalysisFileReport(
            path=request.uri,
            diagnostics=(),
            checker_diagnostics=(),
            axis_tokens=(),
            failures=(),
        ),
    )


async def _wait_until_set(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=2)


def _install_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    analyze: Callable[[DocumentAnalysisRequest], Awaitable[LspDocumentState]],
) -> None:
    async def execute(
        self: LspAnalysisProcess, request: DocumentAnalysisRequest, *, parser: str
    ) -> LspDocumentState:
        return await analyze(request)

    monkeypatch.setattr(LspAnalysisProcess, "analyze", execute)


def test_blocked_analysis_does_not_block_event_loop_or_other_document(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        blocked_started = asyncio.Event()
        release_blocked = asyncio.Event()
        committed_versions: list[int | None] = []
        queue = LspAnalysisQueue(worker_count=2, pending_limit=2)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri.endswith("blocked.py"):
                blocked_started.set()
                await asyncio.wait_for(release_blocked.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        blocked_request = DocumentAnalysisRequest(
            uri="file:///blocked.py",
            source="blocked",
            version=1,
        )
        ready_request = DocumentAnalysisRequest(
            uri="file:///ready.py",
            source="ready",
            version=1,
        )

        blocked_task = asyncio.create_task(
            queue.analyze(
                blocked_request,
                parser="ast",
                commit=lambda state: committed_versions.append(state.version),
            )
        )
        try:
            await _wait_until_set(blocked_started)
            await asyncio.sleep(0)

            ready_state = await asyncio.wait_for(
                queue.analyze(
                    ready_request,
                    parser="ast",
                    commit=lambda state: committed_versions.append(state.version),
                ),
                timeout=0.5,
            )

            assert ready_state is not None
            assert ready_state.uri == ready_request.uri
            assert blocked_task.done() is False
        finally:
            release_blocked.set()
            await blocked_task
            await queue.close()

        assert committed_versions == [1, 1]

    asyncio.run(scenario())


def test_replacement_waits_for_cleanup_without_blocking_other_document(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        first_started = asyncio.Event()
        replacement_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_first = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=2, pending_limit=2)
        uri = "file:///same.py"

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri == uri:
                if request.version == 1:
                    first_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleanup_started.set()
                        await release_first.wait()
                else:
                    replacement_started.set()
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        first_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri=uri,
                    source="first",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: None,
            )
        )
        replacement_task: asyncio.Task[LspDocumentState | None] | None = None
        try:
            await _wait_until_set(first_started)
            replacement_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri=uri,
                        source="replacement",
                        version=2,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await _wait_until_set(cleanup_started)

            other_state = await asyncio.wait_for(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///other.py",
                        source="other",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                ),
                timeout=0.5,
            )

            assert other_state is not None
            assert other_state.uri == "file:///other.py"
            assert replacement_started.is_set() is False
        finally:
            release_first.set()
            await first_task
            if replacement_task is not None:
                await replacement_task
            await queue.close()

    asyncio.run(scenario())


def test_stale_running_analysis_result_is_discarded(monkeypatch) -> None:
    async def scenario() -> None:
        stale_started = asyncio.Event()
        release_stale = asyncio.Event()
        committed_versions: list[int | None] = []
        queue = LspAnalysisQueue(worker_count=2, pending_limit=2)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.version == 1:
                stale_started.set()
                await asyncio.wait_for(release_stale.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        stale_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri="file:///sample.py",
                    source="old",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: committed_versions.append(state.version),
            )
        )
        latest_task: asyncio.Task[LspDocumentState | None] | None = None
        latest_state: LspDocumentState | None = None
        try:
            await _wait_until_set(stale_started)
            latest_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///sample.py",
                        source="new",
                        version=2,
                    ),
                    parser="ast",
                    commit=lambda state: committed_versions.append(state.version),
                )
            )
            await asyncio.sleep(0)

            assert latest_task.done() is False
            assert committed_versions == []
        finally:
            release_stale.set()
            stale_state = await stale_task
            if latest_task is not None:
                latest_state = await latest_task
            await queue.close()

        assert stale_state is None
        assert latest_state is not None
        assert latest_state.version == 2
        assert committed_versions == [2]

    asyncio.run(scenario())


def test_latest_waiter_follows_replacement_generation(monkeypatch) -> None:
    async def scenario() -> None:
        stale_started = asyncio.Event()
        release_stale = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=2, pending_limit=2)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.version == 1:
                stale_started.set()
                await asyncio.wait_for(release_stale.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        stale_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri="file:///sample.py",
                    source="old",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: None,
            )
        )
        try:
            await _wait_until_set(stale_started)
            latest_waiter = asyncio.create_task(
                queue.wait_for_latest(uri="file:///sample.py")
            )
            await asyncio.sleep(0)
            latest_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///sample.py",
                        source="new",
                        version=2,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)
            release_stale.set()
            latest_state = await latest_task
            waited_state = await latest_waiter

            assert latest_state is not None
            assert waited_state == latest_state
        finally:
            release_stale.set()
            await stale_task
            await queue.close()

    asyncio.run(scenario())


def test_pending_request_is_replaceable_while_queue_is_bounded(monkeypatch) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        analyzed_sources: list[str] = []
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            analyzed_sources.append(request.source)
            if request.uri.endswith("active.py"):
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        active_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri="file:///active.py",
                    source="active",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: None,
            )
        )
        try:
            await _wait_until_set(active_started)
            old_pending_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///pending.py",
                        source="old-pending",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)
            latest_pending_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///pending.py",
                        source="new-pending",
                        version=2,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)
            backpressured_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///waiting.py",
                        source="waiting",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)

            assert await old_pending_task is None
            assert backpressured_task.done() is False
        finally:
            release_active.set()
            await active_task

        latest_pending_state = await latest_pending_task
        waiting_state = await backpressured_task
        await queue.close()

        assert latest_pending_state is not None
        assert waiting_state is not None
        assert analyzed_sources == ["active", "new-pending", "waiting"]

    asyncio.run(scenario())


def test_backpressured_request_cancellation_clears_latest_waiter(monkeypatch) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri.endswith("active.py"):
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        active_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri="file:///active.py",
                    source="active",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: None,
            )
        )
        pending_task: asyncio.Task[LspDocumentState | None] | None = None
        try:
            await _wait_until_set(active_started)
            pending_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///pending.py",
                        source="pending",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)
            backpressured_task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri="file:///cancelled.py",
                        source="cancelled",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )
            await asyncio.sleep(0)

            backpressured_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await backpressured_task

            latest = await asyncio.wait_for(
                queue.wait_for_latest(uri="file:///cancelled.py"),
                timeout=0.1,
            )
            assert latest is None
        finally:
            release_active.set()
            await active_task
            if pending_task is not None:
                await pending_task
            await queue.close()

    asyncio.run(scenario())


def test_backpressured_request_is_not_published_before_admission(monkeypatch) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri.endswith("active.py"):
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        def submit(uri: str) -> asyncio.Task[LspDocumentState | None]:
            return asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri=uri,
                        source="source",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )

        active_task = submit("file:///active.py")
        pending_task: asyncio.Task[LspDocumentState | None] | None = None
        backpressured_task: asyncio.Task[LspDocumentState | None] | None = None
        try:
            await _wait_until_set(active_started)
            pending_task = submit("file:///pending.py")
            await asyncio.sleep(0)
            backpressured_task = submit("file:///backpressured.py")
            await asyncio.sleep(0)

            latest = await asyncio.wait_for(
                queue.wait_for_latest(uri="file:///backpressured.py"),
                timeout=0.1,
            )

            assert latest is None
            assert backpressured_task.done() is False
        finally:
            release_active.set()
            await active_task
            if pending_task is not None:
                await pending_task
            if backpressured_task is not None:
                await backpressured_task
            await queue.close()

    asyncio.run(scenario())


def test_backpressured_request_is_replaced_before_capacity_is_available(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri.endswith("active.py"):
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        def submit(
            uri: str, source: str, version: int
        ) -> asyncio.Task[LspDocumentState | None]:
            return asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri=uri,
                        source=source,
                        version=version,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )

        active_task = submit("file:///active.py", "active", 1)
        pending_task: asyncio.Task[LspDocumentState | None] | None = None
        latest_task: asyncio.Task[LspDocumentState | None] | None = None
        try:
            await _wait_until_set(active_started)
            pending_task = submit("file:///pending.py", "pending", 1)
            await asyncio.sleep(0)
            stale_backpressured_task = submit("file:///waiting.py", "old", 1)
            await asyncio.sleep(0)
            latest_task = submit("file:///waiting.py", "new", 2)

            stale_state = await asyncio.wait_for(
                stale_backpressured_task,
                timeout=0.1,
            )
            assert stale_state is None
            assert latest_task.done() is False
        finally:
            release_active.set()
            await active_task
            if pending_task is not None:
                await pending_task
            if latest_task is not None:
                await latest_task
            await queue.close()

    asyncio.run(scenario())


def test_cancelled_running_analysis_never_commits_and_queue_recovers(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        committed_versions: list[int | None] = []
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)
        uri = "file:///sample.py"

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.version == 1:
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        active_task = asyncio.create_task(
            queue.analyze(
                DocumentAnalysisRequest(
                    uri=uri,
                    source="cancelled",
                    version=1,
                ),
                parser="ast",
                commit=lambda state: committed_versions.append(state.version),
            )
        )
        try:
            await _wait_until_set(active_started)
            await queue.cancel(uri=uri)
            assert await active_task is None
        finally:
            release_active.set()

        recovered_state = await queue.analyze(
            DocumentAnalysisRequest(
                uri=uri,
                source="recovered",
                version=2,
            ),
            parser="ast",
            commit=lambda state: committed_versions.append(state.version),
        )
        await queue.close()

        assert recovered_state is not None
        assert recovered_state.version == 2
        assert committed_versions == [2]

    asyncio.run(scenario())


def test_queue_worker_survives_analysis_and_commit_failures(monkeypatch) -> None:
    async def scenario() -> None:
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def fail_analysis(request: DocumentAnalysisRequest) -> LspDocumentState:
            _ = request
            raise ValueError("analysis failed")

        def fail_commit(state: LspDocumentState) -> None:
            _ = state
            raise RuntimeError("commit failed")

        request = DocumentAnalysisRequest(
            uri="file:///sample.py",
            source="source",
            version=1,
        )
        _install_analyzer(monkeypatch, fail_analysis)
        with pytest.raises(ValueError, match="analysis failed"):
            await queue.analyze(
                request,
                parser="ast",
                commit=lambda state: None,
            )

        async def succeed(request: DocumentAnalysisRequest) -> LspDocumentState:
            return _state(request)

        _install_analyzer(monkeypatch, succeed)
        with pytest.raises(RuntimeError, match="commit failed"):
            await queue.analyze(
                request,
                parser="ast",
                commit=fail_commit,
            )

        recovered_state = await queue.analyze(
            request,
            parser="ast",
            commit=lambda state: None,
        )
        await queue.close()

        assert recovered_state is not None

    asyncio.run(scenario())


def test_close_resolves_backpressured_request_and_latest_waiter(monkeypatch) -> None:
    async def scenario() -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            if request.uri.endswith("active.py"):
                active_started.set()
                await asyncio.wait_for(release_active.wait(), timeout=2)
            return _state(request)

        _install_analyzer(monkeypatch, analyze)

        def submit(uri: str) -> asyncio.Task[LspDocumentState | None]:
            return asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(
                        uri=uri,
                        source="source",
                        version=1,
                    ),
                    parser="ast",
                    commit=lambda state: None,
                )
            )

        active_task = submit("file:///active.py")
        pending_task: asyncio.Task[LspDocumentState | None] | None = None
        backpressured_task: asyncio.Task[LspDocumentState | None] | None = None
        latest_waiter: asyncio.Task[LspDocumentState | None] | None = None
        try:
            await _wait_until_set(active_started)
            pending_task = submit("file:///pending.py")
            await asyncio.sleep(0)
            backpressured_task = submit("file:///backpressured.py")
            await asyncio.sleep(0)
            latest_waiter = asyncio.create_task(
                queue.wait_for_latest(uri="file:///backpressured.py")
            )

            await asyncio.wait_for(queue.close(), timeout=0.1)

            assert await active_task is None
            assert await pending_task is None
            assert await backpressured_task is None
            assert await latest_waiter is None
        finally:
            release_active.set()
            await queue.close()

    asyncio.run(scenario())


def test_concurrent_close_waits_for_cleanup_after_first_caller_is_cancelled(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleaning = asyncio.Event()
        release_cleanup = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def analyze(request: DocumentAnalysisRequest) -> LspDocumentState:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release_cleanup.wait()
            return _state(request)

        _install_analyzer(monkeypatch, analyze)
        request = DocumentAnalysisRequest(uri="untitled:close", source="", version=1)
        active = asyncio.create_task(
            queue.analyze(request, parser="ast", commit=lambda state: None)
        )
        first_close = None
        second_close = None
        try:
            await _wait_until_set(started)
            first_close = asyncio.create_task(queue.close())
            await _wait_until_set(cleaning)
            second_close = asyncio.create_task(queue.close())
            await asyncio.sleep(0)
            assert not second_close.done()

            first_close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_close
            assert not second_close.done()
            release_cleanup.set()
            await asyncio.wait_for(second_close, 2)
            assert await active is None
            await queue.close()
            with pytest.raises(RuntimeError, match="queue is closed"):
                await queue.analyze(request, parser="ast", commit=lambda state: None)
        finally:
            release_cleanup.set()
            await queue.close()
            await asyncio.gather(
                active,
                *([first_close] if first_close is not None else []),
                *([second_close] if second_close is not None else []),
                return_exceptions=True,
            )

    asyncio.run(scenario())


def test_unconfirmed_cleanup_fails_queued_and_backpressured_work(monkeypatch) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)
        executions: list[int | None] = []

        async def analyze(request: DocumentAnalysisRequest) -> Never:
            executions.append(request.version)
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise AnalysisProcessCleanupError(
                    "controlled cleanup failure"
                ) from None
            raise AnalysisProcessCleanupError("unexpected analyzer return")

        _install_analyzer(monkeypatch, analyze)

        def submit(version: int, *, uri: str = "untitled:failed"):
            return asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(uri=uri, source="", version=version),
                    parser="ast",
                    commit=lambda state: pytest.fail("failed cleanup cannot commit"),
                )
            )

        old = submit(1)
        latest = None
        pending = None
        waiting = None
        try:
            await _wait_until_set(started)
            pending = submit(1, uri="untitled:pending")
            await asyncio.sleep(0)
            waiting = submit(1, uri="untitled:backpressured")
            await asyncio.sleep(0)
            latest = submit(2)
            assert await asyncio.wait_for(old, 2) is None
            for task in (latest, pending, waiting):
                with pytest.raises(
                    AnalysisProcessCleanupError, match="controlled cleanup failure"
                ):
                    await asyncio.wait_for(task, 2)
            with pytest.raises(RuntimeError, match="queue is closed"):
                await submit(3)
            assert executions == [1]
        finally:
            with pytest.raises(
                AnalysisProcessCleanupError, match="controlled cleanup failure"
            ):
                await queue.close()
            await asyncio.gather(
                old,
                *([latest] if latest is not None else []),
                *([pending] if pending is not None else []),
                *([waiting] if waiting is not None else []),
                return_exceptions=True,
            )

    asyncio.run(scenario())
