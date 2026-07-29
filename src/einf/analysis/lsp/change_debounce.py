import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingDocumentChange:
    """Latest document content waiting for delayed semantic analysis."""

    uri: str
    source: str
    version: int | None


AnalyzePendingChange = Callable[[PendingDocumentChange], Awaitable[None]]


class LspChangeDebouncer:
    """Coalesce rapid document changes before running semantic analysis."""

    def __init__(self, *, delay_seconds: float) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        self._delay_seconds = delay_seconds
        self._pending: dict[str, PendingDocumentChange] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def schedule(
        self,
        change: PendingDocumentChange,
        *,
        analyze: AnalyzePendingChange,
    ) -> None:
        """Schedule semantic analysis for the latest change to one URI."""
        self.cancel(change.uri)
        self._pending[change.uri] = change
        self._tasks[change.uri] = asyncio.create_task(
            self._run_after_delay(uri=change.uri, analyze=analyze)
        )

    def take_pending(self, *, uri: str) -> PendingDocumentChange | None:
        """Cancel delayed analysis and return the pending change, if any."""
        task = self._tasks.pop(uri, None)
        if task is not None and not task.done():
            task.cancel()
        return self._pending.pop(uri, None)

    def cancel(self, uri: str) -> None:
        """Drop pending analysis for one URI."""
        _ = self.take_pending(uri=uri)

    def has_pending(self, *, uri: str) -> bool:
        """Return whether one URI has delayed analysis waiting."""
        return uri in self._pending

    async def _run_after_delay(
        self,
        *,
        uri: str,
        analyze: AnalyzePendingChange,
    ) -> None:
        try:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            change = self._pending.pop(uri, None)
            if change is not None:
                await analyze(change)
        finally:
            current_task = asyncio.current_task()
            if self._tasks.get(uri) is current_task:
                self._tasks.pop(uri, None)


__all__ = ["AnalyzePendingChange", "LspChangeDebouncer", "PendingDocumentChange"]
