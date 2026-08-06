import asyncio
import logging
import math
import os
import shutil
import signal
from dataclasses import dataclass

from einf.analysis.checkers.base import (
    CheckerAdapter,
    adapter_supports_limits,
    diagnostic_count_violation,
    diagnostic_field_violation,
)
from einf.analysis.checkers.model import (
    CheckerFailure,
    CheckerOutputLimits,
    CheckerRequest,
    CheckerResult,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckerExecutionPolicy:
    """Resource limits applied to external checker processes.

    ``max_output_bytes`` bounds the combined stdout+stderr output of one
    process; ``max_diagnostics`` bounds one checker result and the merged
    result; ``max_field_length`` bounds externally derived string fields.
    """

    timeout_seconds: float = 30.0
    max_concurrency: int = 1
    cleanup_timeout_seconds: float = 1.0
    max_output_bytes: int = 8 * 1024 * 1024
    max_diagnostics: int = 10_000
    max_field_length: int = 4096

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
        for field_name, value in (
            ("max_concurrency", self.max_concurrency),
            ("max_output_bytes", self.max_output_bytes),
            ("max_diagnostics", self.max_diagnostics),
            ("max_field_length", self.max_field_length),
        ):
            if isinstance(value, bool) or type(value) is not int or value < 1:
                raise ValueError(f"checker {field_name} must be a positive integer")

    @property
    def limits(self) -> CheckerOutputLimits:
        """Return the parse-time output bounds for one checker result."""
        return CheckerOutputLimits(
            max_diagnostics=self.max_diagnostics,
            max_field_length=self.max_field_length,
        )


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
            try:
                return await self._run_bounded(adapter=adapter, request=request)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                return _failure_result(
                    CheckerFailure(
                        tool=adapter.name,
                        kind="execution_error",
                        message=f"{adapter.name} checker failed: {error}",
                    )
                )

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
        merged = CheckerResult.merge(results)
        if len(merged.diagnostics) > self._policy.max_diagnostics:
            return CheckerResult(
                diagnostics=(),
                failures=merged.failures
                + (
                    CheckerFailure(
                        tool="einf-checkers",
                        kind="output_limit_exceeded",
                        message=(
                            "aggregate checker diagnostics exceeded "
                            f"{self._policy.max_diagnostics}"
                        ),
                    ),
                ),
            )
        return merged

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

        communication = asyncio.create_task(
            bounded_communicate(
                process,
                max_output_bytes=self._policy.max_output_bytes,
            )
        )
        try:
            stdout_bytes, stderr_bytes, output_exceeded = await asyncio.wait_for(
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
        except Exception as error:  # noqa: BLE001
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

        if output_exceeded:
            cleanup_completed = await _kill_and_reap(
                process,
                communication,
                process_group_id=process_group_id,
                timeout_seconds=self._policy.cleanup_timeout_seconds,
            )
            message = (
                f"{adapter.name} output exceeded {self._policy.max_output_bytes} bytes"
            )
            if not cleanup_completed:
                message += (
                    "; process cleanup did not complete within "
                    f"{self._policy.cleanup_timeout_seconds:g} seconds"
                )
            return _failure_result(
                CheckerFailure(
                    tool=adapter.name,
                    kind="output_limit_exceeded",
                    message=message,
                )
            )

        if adapter_supports_limits(adapter.normalize_output):
            result = adapter.normalize_output(
                returncode=process.returncode or 0,
                stdout=_decode_output(stdout_bytes),
                stderr=_decode_output(stderr_bytes),
                request=request,
                limits=self._policy.limits,
            )
        else:
            result = adapter.normalize_output(  # type: ignore[call-arg]
                returncode=process.returncode or 0,
                stdout=_decode_output(stdout_bytes),
                stderr=_decode_output(stderr_bytes),
                request=request,
            )
        return _enforce_result_limits(
            result=result,
            adapter_name=adapter.name,
            limits=self._policy.limits,
        )


class _ByteBudget:
    """Shared byte budget for both output streams of one checker process."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._total = 0

    def consume(self, byte_count: int) -> bool:
        """Reserve bytes; return True when the shared budget is exceeded."""
        self._total += byte_count
        return self._total > self._max_bytes


async def bounded_communicate(
    process: asyncio.subprocess.Process,
    *,
    max_output_bytes: int,
) -> tuple[bytes, bytes, bool]:
    """Read process output under one shared byte budget and wait for exit.

    Returns ``(stdout, stderr, exceeded)`` where ``exceeded`` marks output
    beyond the combined per-process bound; over-bound bytes are drained
    without accumulation so the process can reach EOF without a pipe
    deadlock. The first stream error cancels the sibling reader and
    propagates immediately instead of waiting for its EOF.
    """
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    if stdout_stream is None or stderr_stream is None:
        raise OSError("checker process output pipes unavailable")

    budget = _ByteBudget(max_output_bytes)

    async def read_stream(stream: asyncio.StreamReader) -> tuple[bytes, bool]:
        accumulated = bytearray()
        exceeded = False
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            if budget.consume(len(chunk)):
                exceeded = True
                continue
            accumulated.extend(chunk)
        return bytes(accumulated), exceeded

    stdout_task = asyncio.create_task(read_stream(stdout_stream))
    stderr_task = asyncio.create_task(read_stream(stderr_stream))
    reader_tasks = (stdout_task, stderr_task)
    pending: set[asyncio.Task[tuple[bytes, bool]]] = set(reader_tasks)
    results: dict[asyncio.Task[tuple[bytes, bool]], tuple[bytes, bool]] = {}
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                if task.cancelled():
                    continue
                error = task.exception()
                if error is not None:
                    others = (pending | done) - {task}
                    for sibling in others:
                        sibling.cancel()
                    if others:
                        await asyncio.gather(*others, return_exceptions=True)
                    raise error
                results[task] = task.result()
    except BaseException:
        for task in reader_tasks:
            task.cancel()
        await asyncio.gather(*reader_tasks, return_exceptions=True)
        raise
    await process.wait()
    stdout_bytes, stdout_exceeded = results[stdout_task]
    stderr_bytes, stderr_exceeded = results[stderr_task]
    return stdout_bytes, stderr_bytes, stdout_exceeded or stderr_exceeded


def _enforce_result_limits(
    *,
    result: CheckerResult,
    adapter_name: str,
    limits: CheckerOutputLimits,
) -> CheckerResult:
    bounded_failures = tuple(
        _truncate_failure_message(failure, limits=limits) for failure in result.failures
    )
    if len(result.diagnostics) > limits.max_diagnostics:
        return CheckerResult(
            diagnostics=(),
            failures=bounded_failures
            + (diagnostic_count_violation(tool=adapter_name, limits=limits),),
        )
    for diagnostic in result.diagnostics:
        field_violation = diagnostic_field_violation(
            tool=adapter_name,
            limits=limits,
            diagnostic=diagnostic,
        )
        if field_violation is not None:
            return CheckerResult(
                diagnostics=(),
                failures=bounded_failures + (field_violation,),
            )
    return CheckerResult(
        diagnostics=result.diagnostics,
        failures=bounded_failures,
    )


def _truncate_failure_message(
    failure: CheckerFailure,
    *,
    limits: CheckerOutputLimits,
) -> CheckerFailure:
    if len(failure.message) <= limits.max_field_length:
        return failure
    return CheckerFailure(
        tool=failure.tool,
        kind=failure.kind,
        message=failure.message[: limits.max_field_length],
    )


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[object],
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
    communication: asyncio.Task[object],
    *,
    process_group_id: int | None,
    timeout_seconds: float,
) -> bool:
    termination_succeeded = _terminate_process_scope(
        process,
        process_group_id=process_group_id,
    )
    transport_close_succeeded = _close_process_transport(process)
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
        return termination_succeeded and transport_close_succeeded

    _terminate_process_scope(process, process_group_id=process_group_id)
    if communication in pending:
        communication.add_done_callback(_consume_task_exception)
        communication.cancel()
    if process_wait in pending:
        process_wait.add_done_callback(_consume_task_exception)
        process_wait.cancel()
    return False


def _close_process_transport(process: asyncio.subprocess.Process) -> bool:
    # asyncio.Process has no public close method; its transport owns all pipe FDs.
    transport = getattr(process, "_transport", None)
    if not isinstance(transport, asyncio.SubprocessTransport):
        return False
    try:
        transport.close()
    except OSError:
        return False
    return True


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
