import asyncio
import logging
import math
import os
import shutil
import signal
from dataclasses import dataclass

from einf.analysis.checkers.base import CheckerAdapter
from einf.analysis.checkers.model import CheckerFailure, CheckerRequest, CheckerResult

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckerExecutionPolicy:
    """Resource limits applied to external checker processes."""

    timeout_seconds: float = 30.0
    max_concurrency: int = 1
    cleanup_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("checker timeout must be a finite positive number")
        if (
            isinstance(self.cleanup_timeout_seconds, bool)
            or not isinstance(self.cleanup_timeout_seconds, (int, float))
            or not math.isfinite(self.cleanup_timeout_seconds)
            or self.cleanup_timeout_seconds <= 0
        ):
            raise ValueError("checker cleanup timeout must be a finite positive number")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("checker max_concurrency must be positive")


class CheckerExecutor:
    """Run external checker processes under one bounded execution policy."""

    def __init__(self, policy: CheckerExecutionPolicy) -> None:
        self._policy = policy
        self._capacity = asyncio.Semaphore(policy.max_concurrency)

    @property
    def policy(self) -> CheckerExecutionPolicy:
        """Return the immutable execution policy."""
        return self._policy

    async def run(
        self,
        adapter: CheckerAdapter,
        request: CheckerRequest,
    ) -> CheckerResult:
        """Execute one checker with bounded concurrency and process cleanup."""
        async with self._capacity:
            return await self._run_bounded(adapter=adapter, request=request)

    async def run_all(
        self,
        adapters: tuple[CheckerAdapter, ...],
        request: CheckerRequest,
    ) -> CheckerResult:
        """Execute adapters under the shared bound and merge their results."""
        tasks = tuple(
            asyncio.create_task(self.run(adapter, request)) for adapter in adapters
        )
        try:
            results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await _cancel_tasks(tasks)
            raise
        except Exception:
            await _cancel_tasks(tasks)
            raise
        return CheckerResult.merge(results)

    async def _run_bounded(
        self,
        *,
        adapter: CheckerAdapter,
        request: CheckerRequest,
    ) -> CheckerResult:
        if shutil.which(adapter.executable) is None:
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="unavailable",
                    message=f"checker executable not found: {adapter.executable}",
                )
            )

        command = adapter.build_command(request)
        try:
            if os.name == "posix":
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=request.project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                process_group_id = process.pid
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=request.project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                process_group_id = None
        except OSError as error:
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="spawn_error",
                    message=str(error),
                )
            )

        communication = asyncio.create_task(process.communicate())
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=self._policy.timeout_seconds,
            )
        except asyncio.TimeoutError:
            cleanup_completed = await _kill_and_reap(
                process,
                communication,
                process_group_id=process_group_id,
                timeout_seconds=self._policy.cleanup_timeout_seconds,
            )
            message = (
                f"{adapter.name} exceeded {self._policy.timeout_seconds:g} seconds"
            )
            if not cleanup_completed:
                message += (
                    "; process cleanup did not complete within "
                    f"{self._policy.cleanup_timeout_seconds:g} seconds"
                )
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="timeout",
                    message=message,
                )
            )
        except asyncio.CancelledError:
            cleanup_completed = await _kill_and_reap(
                process,
                communication,
                process_group_id=process_group_id,
                timeout_seconds=self._policy.cleanup_timeout_seconds,
            )
            if not cleanup_completed:
                _LOGGER.warning(
                    "%s process cleanup did not complete within %g seconds",
                    adapter.name,
                    self._policy.cleanup_timeout_seconds,
                )
            raise
        except OSError as error:
            cleanup_completed = await _kill_and_reap(
                process,
                communication,
                process_group_id=process_group_id,
                timeout_seconds=self._policy.cleanup_timeout_seconds,
            )
            message = str(error)
            if not cleanup_completed:
                message += (
                    "; process cleanup did not complete within "
                    f"{self._policy.cleanup_timeout_seconds:g} seconds"
                )
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="execution_error",
                    message=message,
                )
            )

        return adapter.normalize_output(
            returncode=process.returncode or 0,
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
            request=request,
        )


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]],
    *,
    process_group_id: int | None,
    timeout_seconds: float,
) -> bool:
    cleanup = asyncio.create_task(
        _kill_and_wait(
            process,
            communication,
            process_group_id=process_group_id,
            timeout_seconds=timeout_seconds,
        )
    )
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            cancellation = error
    cleanup_completed = await cleanup
    if cancellation is not None:
        raise cancellation
    return cleanup_completed


async def _kill_and_wait(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]],
    *,
    process_group_id: int | None,
    timeout_seconds: float,
) -> bool:
    termination_succeeded = _terminate_process_scope(
        process,
        process_group_id=process_group_id,
    )
    process_wait = asyncio.create_task(process.wait())
    done, pending = await asyncio.wait(
        {communication, process_wait},
        timeout=timeout_seconds,
    )
    if communication in done:
        _consume_task_exception(communication)
    if process_wait in done:
        _consume_task_exception(process_wait)
    if not pending:
        if not termination_succeeded:
            termination_succeeded = _terminate_process_scope(
                process,
                process_group_id=process_group_id,
            )
        return termination_succeeded

    _terminate_process_scope(process, process_group_id=process_group_id)
    if communication in pending:
        communication.add_done_callback(_consume_task_exception)
        communication.cancel()
    if process_wait in pending:
        process_wait.add_done_callback(_consume_task_exception)
        process_wait.cancel()
    return False


def _terminate_process_scope(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int | None,
) -> bool:
    scope_terminated = process_group_id is None
    if process_group_id is not None:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            scope_terminated = True
        except OSError:
            scope_terminated = False
        else:
            return True
    if process.returncode is not None:
        return scope_terminated
    try:
        process.kill()
    except ProcessLookupError:
        return scope_terminated
    except OSError:
        return False
    return scope_terminated


def _consume_task_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _decode_output(output: bytes | None) -> str:
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace")


def _failure_result(failure: CheckerFailure) -> CheckerResult:
    return CheckerResult(diagnostics=(), failures=(failure,))


async def _cancel_tasks(tasks: tuple[asyncio.Task[CheckerResult], ...]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["CheckerExecutionPolicy", "CheckerExecutor"]
