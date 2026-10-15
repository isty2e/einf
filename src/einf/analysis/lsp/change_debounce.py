import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

from .analysis_limits import LspSourceReservation


@dataclass(frozen=True, slots=True)
class PendingDocumentChange:
    """Latest document content waiting for delayed semantic analysis.

    Parameters
    ----------
    uri : str
        Document identity.
    source : str
        Immutable source snapshot.
    version : int or None
        Client document version, if available.
    source_reservation : LspSourceReservation or None, optional
        Borrowed token for the snapshot; scheduling retains its own holder.
    """

    uri: str
    source: str
    version: int | None
    source_reservation: LspSourceReservation | None = None


AnalyzePendingChange = Callable[[PendingDocumentChange], Awaitable[None]]


class LspChangeDebouncer:
    """Coalesce rapid document changes before running semantic analysis."""

    def __init__(
        self,
        *,
        delay_seconds: float,
        report_failure: Callable[[PendingDocumentChange, Exception], None],
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        self._delay_seconds = delay_seconds
        self._report_failure = report_failure
        self._pending: dict[str, PendingDocumentChange] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._live_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def schedule(
        self,
        change: PendingDocumentChange,
        *,
        analyze: AnalyzePendingChange,
    ) -> None:
        """Schedule the latest change, retaining its reservation until task exit.

        Parameters
        ----------
        change : PendingDocumentChange
            Borrowed snapshot; the caller keeps its own reservation reference.
        analyze : AnalyzePendingChange
            Callback consuming the snapshot after the delay.

        Raises
        ------
        RuntimeError
            The debouncer is closed or no event loop is running.
        """
        if self._closed:
            raise RuntimeError("change debouncer is closed")
        asyncio.get_running_loop()
        self.cancel(change.uri)
        if change.source_reservation is not None:
            change.source_reservation.retain()
        self._pending[change.uri] = change
        execution = self._run_after_delay(uri=change.uri, analyze=analyze)
        try:
            task = asyncio.create_task(execution)
        except BaseException:
            execution.close()
            self._pending.pop(change.uri, None)
            if change.source_reservation is not None:
                change.source_reservation.release()
            raise
        self._tasks[change.uri] = task
        self._live_tasks.add(task)
        task.add_done_callback(partial(self._handle_task_completion, change=change))

    def take_pending(self, *, uri: str) -> PendingDocumentChange | None:
        """Transfer a pending change to its caller and cancel delayed execution.

        Parameters
        ----------
        uri : str
            Document to flush.

        Returns
        -------
        PendingDocumentChange or None
            Pending snapshot. Its caller owns one reservation reference, when
            present, and must release it after use.
        """
        task = self._tasks.pop(uri, None)
        if task is not None and not task.done():
            task.cancel()
        change = self._pending.pop(uri, None)
        if change is not None and change.source_reservation is not None:
            change.source_reservation.retain()
        return change

    def cancel(self, uri: str) -> None:
        """Drop pending analysis for one URI."""
        change = self.take_pending(uri=uri)
        if change is not None and change.source_reservation is not None:
            change.source_reservation.release()

    async def cancel_and_wait(self, *, uri: str) -> None:
        """Cancel one URI and wait for its task to relinquish source ownership.

        Parameters
        ----------
        uri : str
            Document whose delayed or running analysis is obsolete.
        """
        task = self._tasks.get(uri)
        self.cancel(uri)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def has_pending(self, *, uri: str) -> bool:
        """Return whether one URI has delayed analysis waiting."""
        return uri in self._pending

    async def close(self) -> None:
        """Drop pending changes and stop all owned analysis tasks."""
        if self._closed:
            return
        self._closed = True

        tasks = tuple(self._live_tasks)
        self._pending.clear()
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

    def _handle_task_completion(
        self,
        task: asyncio.Task[None],
        *,
        change: PendingDocumentChange,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:  # noqa: BLE001
            self._report_failure(change, error)
        finally:
            self._live_tasks.discard(task)
            if change.source_reservation is not None:
                change.source_reservation.release()


__all__ = [
    "AnalyzePendingChange",
    "LspChangeDebouncer",
    "PendingDocumentChange",
]
