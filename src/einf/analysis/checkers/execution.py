import asyncio
import math
import shutil
from dataclasses import dataclass

from einf.analysis.checkers.base import CheckerAdapter
from einf.analysis.checkers.model import CheckerFailure, CheckerRequest, CheckerResult


@dataclass(frozen=True, slots=True)
class CheckerExecutionPolicy:
    """Resource limits applied to external checker processes."""

    timeout_seconds: float = 30.0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("checker timeout must be a finite positive number")
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
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=request.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
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
            await _kill_and_reap(process, communication)
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="timeout",
                    message=(
                        f"{adapter.name} exceeded "
                        f"{self._policy.timeout_seconds:g} seconds"
                    ),
                )
            )
        except asyncio.CancelledError:
            await _kill_and_reap(process, communication)
            raise
        except OSError as error:
            await _kill_and_reap(process, communication)
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="execution_error",
                    message=str(error),
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
) -> None:
    cleanup = asyncio.create_task(_kill_and_wait(process, communication))
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            cancellation = error
    await cleanup
    if cancellation is not None:
        raise cancellation


async def _kill_and_wait(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await asyncio.gather(communication, return_exceptions=True)
    await process.wait()


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
