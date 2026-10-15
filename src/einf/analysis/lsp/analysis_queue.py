import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from .analysis_process import AnalysisProcessCleanupError, LspAnalysisProcess
from .service import LspDocumentState


@dataclass(frozen=True, slots=True)
class DocumentAnalysisRequest:
    """Immutable document snapshot submitted for asynchronous analysis."""

    uri: str
    source: str
    version: int | None


@dataclass(frozen=True, slots=True)
class _PendingAnalysis:
    request: DocumentAnalysisRequest
    generation: int
    parser: str
    commit: Callable[[LspDocumentState], None]
    completion: asyncio.Future[LspDocumentState | None]
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)


class LspAnalysisQueue:
    """Bounded URI-keyed queue with reusable semantic-analysis processes.

    Parameters
    ----------
    worker_count : int
        Positive maximum number of owned worker processes.
    pending_limit : int
        Positive maximum number of admitted pending document snapshots.

    Raises
    ------
    ValueError
        Either limit is less than one.
    """

    def __init__(self, *, worker_count: int, pending_limit: int) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if pending_limit < 1:
            raise ValueError("pending_limit must be positive")

        self._worker_count = worker_count
        self._pending_limit = pending_limit
        self._pending: OrderedDict[str, _PendingAnalysis] = OrderedDict()
        self._running: dict[tuple[str, int], _PendingAnalysis] = {}
        self._generations: dict[str, int] = {}
        # Submissions own generations; completions expose only admitted work.
        self._latest_submissions: dict[
            str, asyncio.Future[LspDocumentState | None]
        ] = {}
        self._latest_completions: dict[
            str, asyncio.Future[LspDocumentState | None]
        ] = {}
        self._condition = asyncio.Condition()
        self._worker_tasks: tuple[asyncio.Task[None], ...] = ()
        self._closed = False
        self._cleanup_error: AnalysisProcessCleanupError | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def analyze(
        self,
        request: DocumentAnalysisRequest,
        *,
        parser: str,
        commit: Callable[[LspDocumentState], None],
    ) -> LspDocumentState | None:
        """Analyze one snapshot and commit it only while it remains current.

        Parameters
        ----------
        request : DocumentAnalysisRequest
            Document snapshot; a newer submission replaces work for this URI.
        parser : str
            Built-in parser name sent to the isolated worker.
        commit : Callable[[LspDocumentState], None]
            Parent-side callback invoked only for a still-current result.

        Returns
        -------
        LspDocumentState or None
            Committed snapshot, or ``None`` after invalidation or shutdown.

        Raises
        ------
        RuntimeError
            The queue is closed or worker execution failed.
        ValueError
            The selected parser is unsupported, or the commit callback raises it.
        Exception
            The commit callback failed; its original exception is preserved.
        asyncio.CancelledError
            The submission was cancelled; running work is signalled for retirement.
        """
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[LspDocumentState | None] = loop.create_future()
        generation: int | None = None

        try:
            async with self._condition:
                if self._closed:
                    if self._cleanup_error is not None:
                        raise RuntimeError(
                            "analysis queue is closed after worker cleanup failed; "
                            "restart the einf language server"
                        ) from self._cleanup_error
                    raise RuntimeError("analysis queue is closed")
                self._ensure_workers()

                generation = self._generations.get(request.uri, 0) + 1
                self._generations[request.uri] = generation
                self._supersede_latest_submission(request.uri)
                self._latest_submissions[request.uri] = completion
                self._condition.notify_all()

                previous_pending = self._pending.pop(request.uri, None)
                if previous_pending is not None:
                    _resolve_stale(previous_pending.completion)

                while len(self._pending) >= self._pending_limit:
                    await self._condition.wait()
                    if self._closed:
                        _resolve_stale(completion)
                        return completion.result()
                    if self._generations.get(request.uri) != generation:
                        _resolve_stale(completion)
                        return None

                item = _PendingAnalysis(
                    request=request,
                    generation=generation,
                    parser=parser,
                    commit=commit,
                    completion=completion,
                )
                self._pending[request.uri] = item
                self._latest_completions[request.uri] = completion
                self._condition.notify()

            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            if generation is not None:
                await asyncio.shield(
                    self._cancel_generation(uri=request.uri, generation=generation)
                )
            raise

    async def wait_for_latest(self, *, uri: str) -> LspDocumentState | None:
        """Wait for the latest queued or running analysis for one URI."""
        while True:
            async with self._condition:
                completion = self._latest_completions.get(uri)
            if completion is None:
                return None

            state = await asyncio.shield(completion)
            async with self._condition:
                replacement = self._latest_completions.get(uri)
            if replacement is None or replacement is completion:
                return state

    async def cancel(self, *, uri: str) -> None:
        """Invalidate pending and running analysis for one document URI."""
        async with self._condition:
            self._generations[uri] = self._generations.get(uri, 0) + 1
            pending = self._pending.pop(uri, None)
            if pending is not None:
                _resolve_stale(pending.completion)

            submission = self._latest_submissions.pop(uri, None)
            if submission is not None:
                _resolve_stale(submission)
            completion = self._latest_completions.pop(uri, None)
            if completion is not None:
                _resolve_stale(completion)
            for (running_uri, _), running in self._running.items():
                if running_uri == uri:
                    running.cancelled.set()
                    _resolve_stale(running.completion)

            self._condition.notify_all()
            self._discard_generation_if_idle(uri)

    async def close(self) -> None:
        """Invalidate work and wait for the shared process-cleanup operation.

        Raises
        ------
        AnalysisProcessCleanupError
            An owned worker could not finish cleanup within its deadline.
        """
        async with self._condition:
            if self._close_task is None:
                self._closed = True

                for item in self._pending.values():
                    _resolve_stale(item.completion)
                for item in self._running.values():
                    item.cancelled.set()
                    _resolve_stale(item.completion)
                for submission in self._latest_submissions.values():
                    _resolve_stale(submission)
                for completion in self._latest_completions.values():
                    _resolve_stale(completion)
                self._pending.clear()
                self._latest_submissions.clear()
                self._latest_completions.clear()

                self._close_task = asyncio.create_task(self._close_workers())
                self._condition.notify_all()
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_workers(self) -> None:
        results = await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        async with self._condition:
            self._worker_tasks = ()
            self._running.clear()
            self._generations.clear()
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _fail_cleanup(self, error: AnalysisProcessCleanupError) -> None:
        async with self._condition:
            if self._cleanup_error is None:
                self._cleanup_error = error
            self._closed = True
            for submission in self._latest_submissions.values():
                if not submission.done():
                    submission.set_exception(error)
            for item in self._running.values():
                item.cancelled.set()
            self._pending.clear()
            self._latest_submissions.clear()
            self._latest_completions.clear()
            self._condition.notify_all()

    def _ensure_workers(self) -> None:
        if self._worker_tasks:
            return
        self._worker_tasks = tuple(
            asyncio.create_task(
                self._run_worker(),
                name=f"einf-analysis-{worker_index}",
            )
            for worker_index in range(self._worker_count)
        )

    async def _run_worker(self) -> None:
        process = LspAnalysisProcess()
        try:
            await self._work_until_closed(process)
        finally:
            await process.close()

    async def _work_until_closed(self, process: LspAnalysisProcess) -> None:
        while True:
            async with self._condition:
                next_item = self._take_runnable_pending()
                while next_item is None and not self._closed:
                    await self._condition.wait()
                    next_item = self._take_runnable_pending()
                if self._closed:
                    return
                if next_item is None:
                    continue

                uri, item = next_item
                running_key = (uri, item.generation)
                self._running[running_key] = item
                self._condition.notify_all()

            try:
                state = await self._analyze_pending(process=process, item=item)
            except asyncio.CancelledError:
                _resolve_stale(item.completion)
                raise
            except AnalysisProcessCleanupError as error:
                await self._fail_cleanup(error)
                raise
            except Exception as error:  # noqa: BLE001
                await self._finish_failure(
                    running_key=running_key,
                    item=item,
                    error=error,
                )
            else:
                await self._finish_success(
                    running_key=running_key,
                    item=item,
                    state=state,
                )

    async def _analyze_pending(
        self, *, process: LspAnalysisProcess, item: _PendingAnalysis
    ) -> LspDocumentState | None:
        if item.cancelled.is_set():
            return None
        execution = asyncio.create_task(
            process.analyze(item.request, parser=item.parser)
        )
        interruption = asyncio.create_task(item.cancelled.wait())
        try:
            await asyncio.wait(
                (execution, interruption), return_when=asyncio.FIRST_COMPLETED
            )
            if item.cancelled.is_set():
                execution.cancel()
                try:
                    await execution
                except asyncio.CancelledError:
                    pass
                return None
            return await execution
        finally:
            interruption.cancel()
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, interruption, return_exceptions=True)

    def _take_runnable_pending(self) -> tuple[str, _PendingAnalysis] | None:
        running_uris = {running_uri for running_uri, _ in self._running}
        for uri in self._pending:
            if uri not in running_uris:
                return uri, self._pending.pop(uri)
        return None

    async def _finish_success(
        self,
        *,
        running_key: tuple[str, int],
        item: _PendingAnalysis,
        state: LspDocumentState | None,
    ) -> None:
        async with self._condition:
            self._running.pop(running_key, None)
            is_current = (
                not self._closed
                and self._generations.get(item.request.uri) == item.generation
                and not item.completion.done()
            )
            if is_current and state is not None:
                try:
                    item.commit(state)
                except Exception as error:  # noqa: BLE001
                    item.completion.set_exception(error)
                else:
                    item.completion.set_result(state)
            else:
                _resolve_stale(item.completion)

            self._complete_generation(item)
            self._condition.notify_all()

    async def _finish_failure(
        self,
        *,
        running_key: tuple[str, int],
        item: _PendingAnalysis,
        error: Exception,
    ) -> None:
        async with self._condition:
            self._running.pop(running_key, None)
            is_current = (
                not self._closed
                and self._generations.get(item.request.uri) == item.generation
                and not item.completion.done()
            )
            if is_current:
                item.completion.set_exception(error)
            else:
                _resolve_stale(item.completion)

            self._complete_generation(item)
            self._condition.notify_all()

    async def _cancel_generation(self, *, uri: str, generation: int) -> None:
        async with self._condition:
            if self._generations.get(uri) != generation:
                return
            self._generations[uri] = generation + 1
            for (running_uri, running_generation), running in self._running.items():
                if running_uri == uri and running_generation == generation:
                    running.cancelled.set()
            pending = self._pending.pop(uri, None)
            if pending is not None and pending.generation == generation:
                _resolve_stale(pending.completion)
            submission = self._latest_submissions.pop(uri, None)
            if submission is not None:
                _resolve_stale(submission)
            completion = self._latest_completions.pop(uri, None)
            if completion is not None:
                _resolve_stale(completion)
            self._condition.notify_all()
            self._discard_generation_if_idle(uri)

    def _supersede_latest_submission(self, uri: str) -> None:
        for (running_uri, _), running in self._running.items():
            if running_uri == uri:
                running.cancelled.set()
        submission = self._latest_submissions.get(uri)
        if submission is not None:
            _resolve_stale(submission)

    def _complete_generation(self, item: _PendingAnalysis) -> None:
        uri = item.request.uri
        if self._latest_submissions.get(uri) is item.completion:
            self._latest_submissions.pop(uri, None)
        if self._latest_completions.get(uri) is item.completion:
            self._latest_completions.pop(uri, None)
        self._discard_generation_if_idle(uri)

    def _discard_generation_if_idle(self, uri: str) -> None:
        has_running = any(running_uri == uri for running_uri, _ in self._running)
        if (
            uri not in self._pending
            and not has_running
            and uri not in self._latest_submissions
            and uri not in self._latest_completions
        ):
            self._generations.pop(uri, None)


def _resolve_stale(
    completion: asyncio.Future[LspDocumentState | None],
) -> None:
    if not completion.done():
        completion.set_result(None)


__all__ = ["DocumentAnalysisRequest", "LspAnalysisQueue"]
