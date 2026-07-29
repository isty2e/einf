import asyncio

from einf.analysis.lsp.change_debounce import (
    LspChangeDebouncer,
    PendingDocumentChange,
)


def test_lsp_change_debouncer_coalesces_latest_change() -> None:
    events: list[tuple[str, int | None, str]] = []

    async def scenario() -> None:
        debouncer = LspChangeDebouncer(delay_seconds=0.001)

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


def test_lsp_change_debouncer_can_flush_pending_change() -> None:
    async def scenario() -> PendingDocumentChange | None:
        debouncer = LspChangeDebouncer(delay_seconds=1.0)

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


def test_lsp_change_debouncer_cancel_drops_pending_change() -> None:
    events: list[PendingDocumentChange] = []

    async def scenario() -> bool:
        debouncer = LspChangeDebouncer(delay_seconds=0.001)

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
