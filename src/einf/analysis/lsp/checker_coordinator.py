import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerExecutor,
    CheckerRequest,
    CheckerResult,
)


@dataclass(frozen=True, slots=True)
class DocumentCheckerRequest:
    """Immutable saved-document snapshot submitted for external checking."""

    uri: str
    path: Path
    version: int | None


_CommitCheckerResult = Callable[[DocumentCheckerRequest, CheckerResult], bool]


@dataclass(frozen=True, slots=True)
class _CheckerJob:
    generation: int
    completion: asyncio.Future[CheckerResult | None]
    task: asyncio.Task[None]


class LspCheckerCoordinator:
    """Coordinate latest-only checker jobs for LSP document snapshots."""

    def __init__(
        self,
        *,
        adapters: tuple[CheckerAdapter, ...],
        executor: CheckerExecutor,
    ) -> None:
        self._adapters = adapters
        self._executor = executor
        self._jobs: dict[str, _CheckerJob] = {}
        self._generations: dict[str, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        """Return whether this coordinator has configured checker adapters."""
        return bool(self._adapters)

    async def check(
        self,
        request: DocumentCheckerRequest,
        *,
        commit: _CommitCheckerResult,
    ) -> CheckerResult | None:
        """Run configured checkers and commit only the latest URI generation."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("checker coordinator is closed")
            if not self._adapters:
                return None

            loop = asyncio.get_running_loop()
            completion: asyncio.Future[CheckerResult | None] = loop.create_future()

            generation = self._generations.get(request.uri, 0) + 1
            self._generations[request.uri] = generation
            previous = self._jobs.pop(request.uri, None)
            if previous is not None:
                _resolve_stale(previous.completion)
                previous.task.cancel()

            task = asyncio.create_task(
                self._run_job(
                    request=request,
                    generation=generation,
                    completion=completion,
                    commit=commit,
                ),
                name=f"einf-checker-{generation}",
            )
            self._jobs[request.uri] = _CheckerJob(
                generation=generation,
                completion=completion,
                task=task,
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._cancel_generation(uri=request.uri, generation=generation),
                name=f"einf-checker-cancel-{generation}",
            )
            await _await_cleanup(cleanup)
            raise

    async def cancel(self, *, uri: str) -> None:
        """Cancel and reap the current checker job for one document URI."""
        async with self._lock:
            self._generations[uri] = self._generations.get(uri, 0) + 1
            job = self._jobs.pop(uri, None)
            if job is not None:
                _resolve_stale(job.completion)
                job.task.cancel()

        if job is not None:
            await asyncio.gather(job.task, return_exceptions=True)

        async with self._lock:
            self._discard_generation_if_idle(uri)

    async def close(self) -> None:
        """Cancel and reap all active checker jobs."""
        async with self._lock:
            if self._close_task is None:
                self._closed = True
                for job in self._jobs.values():
                    _resolve_stale(job.completion)
                self._jobs.clear()

                tasks = tuple(self._tasks)
                for task in tasks:
                    task.cancel()
                self._close_task = asyncio.create_task(
                    self._finish_close(tasks),
                    name="einf-checker-close",
                )
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _finish_close(self, tasks: tuple[asyncio.Task[None], ...]) -> None:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        async with self._lock:
            self._tasks.clear()
            self._generations.clear()

    async def _run_job(
        self,
        *,
        request: DocumentCheckerRequest,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
        commit: _CommitCheckerResult,
    ) -> None:
        checker_request = CheckerRequest(
            targets=(request.path,),
            project_root=request.path.parent,
        )
        try:
            result = await self._executor.run_all(self._adapters, checker_request)
        except asyncio.CancelledError:
            await self._finish_cancelled(
                uri=request.uri,
                generation=generation,
                completion=completion,
            )
        except Exception as error:  # noqa: BLE001
            await self._finish_failure(
                uri=request.uri,
                generation=generation,
                completion=completion,
                error=error,
            )
        else:
            await self._finish_success(
                request=request,
                generation=generation,
                completion=completion,
                result=result,
                commit=commit,
            )

    async def _finish_success(
        self,
        *,
        request: DocumentCheckerRequest,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
        result: CheckerResult,
        commit: _CommitCheckerResult,
    ) -> None:
        async with self._lock:
            is_current = self._is_current(
                uri=request.uri,
                generation=generation,
                completion=completion,
            )
            if is_current:
                try:
                    committed = commit(request, result)
                except Exception as error:  # noqa: BLE001
                    completion.set_exception(error)
                else:
                    completion.set_result(result if committed else None)
            else:
                _resolve_stale(completion)
            self._complete_generation(
                uri=request.uri,
                generation=generation,
                completion=completion,
            )

    async def _finish_failure(
        self,
        *,
        uri: str,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
        error: Exception,
    ) -> None:
        async with self._lock:
            if self._is_current(
                uri=uri,
                generation=generation,
                completion=completion,
            ):
                completion.set_exception(error)
            else:
                _resolve_stale(completion)
            self._complete_generation(
                uri=uri,
                generation=generation,
                completion=completion,
            )

    async def _finish_cancelled(
        self,
        *,
        uri: str,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
    ) -> None:
        async with self._lock:
            _resolve_stale(completion)
            self._complete_generation(
                uri=uri,
                generation=generation,
                completion=completion,
            )

    async def _cancel_generation(self, *, uri: str, generation: int) -> None:
        async with self._lock:
            job = self._jobs.get(uri)
            if job is None or job.generation != generation:
                return
            self._generations[uri] = generation + 1
            self._jobs.pop(uri)
            _resolve_stale(job.completion)
            job.task.cancel()

        await asyncio.gather(job.task, return_exceptions=True)
        async with self._lock:
            self._discard_generation_if_idle(uri)

    def _is_current(
        self,
        *,
        uri: str,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
    ) -> bool:
        job = self._jobs.get(uri)
        return (
            not self._closed
            and job is not None
            and job.generation == generation
            and job.completion is completion
            and not completion.done()
        )

    def _complete_generation(
        self,
        *,
        uri: str,
        generation: int,
        completion: asyncio.Future[CheckerResult | None],
    ) -> None:
        job = self._jobs.get(uri)
        if (
            job is not None
            and job.generation == generation
            and job.completion is completion
        ):
            self._jobs.pop(uri)
        self._discard_generation_if_idle(uri)

    def _discard_generation_if_idle(self, uri: str) -> None:
        if uri not in self._jobs:
            self._generations.pop(uri, None)


def _resolve_stale(completion: asyncio.Future[CheckerResult | None]) -> None:
    if not completion.done():
        completion.set_result(None)


async def _await_cleanup(cleanup: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            cancellation = error
    await cleanup
    if cancellation is not None:
        raise cancellation


__all__ = ["DocumentCheckerRequest", "LspCheckerCoordinator"]
