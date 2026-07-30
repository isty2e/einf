import asyncio

import pytest

from einf.analysis.lsp.change_debounce import (
    LspChangeDebouncer,
    PendingDocumentChange,
)


def test_lsp_change_debouncer_coalesces_latest_change() -> None:
    events: list[tuple[str, int | None, str]] = []
    failures: list[tuple[PendingDocumentChange, Exception]] = []

    async def scenario() -> None:
        debouncer = LspChangeDebouncer(
            delay_seconds=0.001,
            report_failure=lambda change, error: failures.append((change, error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            events.append((change.uri, change.version, change.source))

        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="old", version=1),
            analyze=analyze,
        )
        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="new", version=2),
            analyze=analyze,
        )
        await asyncio.sleep(0.01)

    asyncio.run(scenario())

    assert events == [("file:///sample.py", 2, "new")]
    assert failures == []


def test_lsp_change_debouncer_can_flush_pending_change() -> None:
    failures: list[tuple[PendingDocumentChange, Exception]] = []

    async def scenario() -> PendingDocumentChange | None:
        debouncer = LspChangeDebouncer(
            delay_seconds=1.0,
            report_failure=lambda change, error: failures.append((change, error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            _ = change

        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="latest", version=3),
            analyze=analyze,
        )
        pending = debouncer.take_pending(uri="file:///sample.py")
        await asyncio.sleep(0)
        return pending

    pending = asyncio.run(scenario())

    assert pending == PendingDocumentChange(
        uri="file:///sample.py",
        source="latest",
        version=3,
    )
    assert failures == []


def test_lsp_change_debouncer_cancel_drops_pending_change() -> None:
    events: list[PendingDocumentChange] = []
    failures: list[tuple[PendingDocumentChange, Exception]] = []

    async def scenario() -> bool:
        debouncer = LspChangeDebouncer(
            delay_seconds=0.001,
            report_failure=lambda change, error: failures.append((change, error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            events.append(change)

        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="stale", version=1),
            analyze=analyze,
        )
        debouncer.cancel("file:///sample.py")
        await asyncio.sleep(0.01)
        return debouncer.has_pending(uri="file:///sample.py")

    has_pending = asyncio.run(scenario())

    assert has_pending is False
    assert events == []
    assert failures == []


def test_lsp_change_debouncer_replaces_running_analysis_without_failure() -> None:
    events: list[PendingDocumentChange] = []
    failures: list[tuple[PendingDocumentChange, Exception]] = []

    async def scenario() -> None:
        stale_started = asyncio.Event()
        stale_finished = asyncio.Event()
        release_stale = asyncio.Event()
        latest_completed = asyncio.Event()
        debouncer = LspChangeDebouncer(
            delay_seconds=0,
            report_failure=lambda change, error: failures.append((change, error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            if change.version == 1:
                stale_started.set()
                try:
                    await release_stale.wait()
                finally:
                    stale_finished.set()
                return
            events.append(change)
            latest_completed.set()

        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="old", version=1),
            analyze=analyze,
        )
        await asyncio.wait_for(stale_started.wait(), timeout=1)
        debouncer.schedule(
            PendingDocumentChange(uri="file:///sample.py", source="new", version=2),
            analyze=analyze,
        )
        await asyncio.wait_for(
            asyncio.gather(stale_finished.wait(), latest_completed.wait()),
            timeout=1,
        )
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert events == [
        PendingDocumentChange(uri="file:///sample.py", source="new", version=2)
    ]
    assert failures == []


def test_lsp_change_debouncer_reports_and_retrieves_analysis_failure() -> None:
    change = PendingDocumentChange(
        uri="file:///sample.py",
        source="broken",
        version=4,
    )
    failures: list[tuple[PendingDocumentChange, Exception]] = []
    loop_errors: list[dict[str, object]] = []

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: loop_errors.append(context))
        reported = asyncio.Event()

        def report_failure(
            failed_change: PendingDocumentChange,
            error: Exception,
        ) -> None:
            failures.append((failed_change, error))
            reported.set()

        debouncer = LspChangeDebouncer(
            delay_seconds=0,
            report_failure=report_failure,
        )

        async def analyze(pending_change: PendingDocumentChange) -> None:
            _ = pending_change
            raise RuntimeError("analysis failed")

        try:
            debouncer.schedule(change, analyze=analyze)
            await asyncio.wait_for(reported.wait(), timeout=1)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())

    assert len(failures) == 1
    failed_change, error = failures[0]
    assert failed_change == change
    assert isinstance(error, RuntimeError)
    assert str(error) == "analysis failed"
    assert loop_errors == []


def test_lsp_change_debouncer_close_cancels_and_awaits_running_analysis() -> None:
    failures: list[tuple[PendingDocumentChange, Exception]] = []

    async def scenario() -> tuple[bool, bool]:
        analysis_started = asyncio.Event()
        analysis_stopped = asyncio.Event()
        debouncer = LspChangeDebouncer(
            delay_seconds=0,
            report_failure=lambda change, error: failures.append((change, error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            _ = change
            analysis_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                analysis_stopped.set()

        change = PendingDocumentChange(
            uri="file:///sample.py",
            source="pending",
            version=5,
        )
        debouncer.schedule(change, analyze=analyze)
        await asyncio.wait_for(analysis_started.wait(), timeout=1)

        await debouncer.close()
        await debouncer.close()

        with pytest.raises(RuntimeError, match="change debouncer is closed"):
            debouncer.schedule(change, analyze=analyze)

        return (
            analysis_stopped.is_set(),
            debouncer.has_pending(uri=change.uri),
        )

    analysis_stopped, has_pending = asyncio.run(scenario())

    assert analysis_stopped is True
    assert has_pending is False
    assert failures == []
