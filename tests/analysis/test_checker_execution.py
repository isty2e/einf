import asyncio
import os
import sys
from pathlib import Path

import pytest

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerRequest,
    CheckerResult,
)


class _CommandAdapter(CheckerAdapter):
    name = "stub"

    def __init__(self, *, executable: str, command: tuple[str, ...]) -> None:
        self.executable = executable
        self._command = command

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        _ = request
        return self._command

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr, request
        return CheckerResult(diagnostics=(), failures=())


class _ConcurrencyProbeExecutor(CheckerExecutor):
    def __init__(self, policy: CheckerExecutionPolicy) -> None:
        super().__init__(policy)
        self.active = 0
        self.max_active = 0

    async def _run_bounded(
        self,
        *,
        adapter: CheckerAdapter,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = adapter, request
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return CheckerResult(diagnostics=(), failures=())


class _FailingOutputAdapter(_CommandAdapter):
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr, request
        raise RuntimeError("adapter parse failed")


class _CommunicateErrorProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes | None, bytes | None]:
        raise OSError("pipe failed")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


class _SlowCleanupProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.communicate_started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.finish_cleanup = asyncio.Event()
        self.reaped = asyncio.Event()

    async def communicate(self) -> tuple[bytes | None, bytes | None]:
        self.communicate_started.set()
        await self.cleanup_started.wait()
        await self.finish_cleanup.wait()
        return b"", b""

    def kill(self) -> None:
        self.returncode = -9
        self.cleanup_started.set()

    async def wait(self) -> int:
        self.reaped.set()
        return self.returncode or 0


def _request(tmp_path: Path) -> CheckerRequest:
    target = tmp_path / "sample.py"
    target.write_text("value: int = 1\n", encoding="utf-8")
    return CheckerRequest(targets=(target,), project_root=tmp_path)


def _write_blocking_checker(path: Path) -> None:
    path.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "pid_path = Path(sys.argv[1])\n"
        "pending_path = pid_path.with_suffix('.tmp')\n"
        "pending_path.write_text(str(os.getpid()), encoding='utf-8')\n"
        "pending_path.replace(pid_path)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )


async def _wait_for_pid(path: Path) -> int:
    for _ in range(1_000):
        if path.exists():
            return int(path.read_text(encoding="utf-8"))
        await asyncio.sleep(0.001)
    raise AssertionError("checker process did not publish its pid")


def _assert_process_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    ("timeout_seconds", "max_concurrency"),
    [
        (0.0, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (True, 1),
        ("1", 1),
        (1.0, 0),
        (1.0, True),
        (1.0, 1.5),
    ],
)
def test_checker_execution_policy_rejects_unbounded_values(
    timeout_seconds: float,
    max_concurrency: int,
) -> None:
    with pytest.raises(ValueError):
        CheckerExecutionPolicy(
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
        )


def test_checker_executor_normalizes_spawn_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(str(tmp_path / "missing-command"),),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "spawn_error"

    asyncio.run(scenario())


def test_checker_executor_accepts_empty_adapter_set(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(CheckerExecutionPolicy())

        result = await executor.run_all((), _request(tmp_path))

        assert result == CheckerResult(diagnostics=(), failures=())

    asyncio.run(scenario())


def test_checker_executor_normalizes_nonzero_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('checker failed'); raise SystemExit(2)",
            ),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "execution_error"
        assert result.failures[0].message == "checker failed"

    asyncio.run(scenario())


def test_checker_executor_decodes_non_utf8_output_lossily(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(
                sys.executable,
                "-c",
                "import os; os.write(1, b'\\xff')",
            ),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result == CheckerResult(diagnostics=(), failures=())

    asyncio.run(scenario())


def test_checker_executor_reaps_process_after_communication_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()

        async def create_process(
            *command: str,
            **options: object,
        ) -> _CommunicateErrorProcess:
            _ = command, options
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "execution_error"
        assert result.failures[0].message == "pipe failed"
        assert process.killed is True
        assert process.waited is True

    asyncio.run(scenario())


def test_repeated_cancellation_cannot_complete_before_process_reap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        process = _SlowCleanupProcess()

        async def create_process(
            *command: str,
            **options: object,
        ) -> _SlowCleanupProcess:
            _ = command, options
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        await process.communicate_started.wait()

        task.cancel()
        await process.cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        completion_waited_for_reap = not task.done()

        process.finish_cleanup.set()
        await asyncio.wait_for(process.reaped.wait(), timeout=1.0)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completion_waited_for_reap

    asyncio.run(scenario())


def test_checker_executor_kills_and_reaps_timed_out_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=1.0, max_concurrency=1)
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(checker_script), str(pid_path)),
        )

        result = await executor.run(adapter, _request(tmp_path))
        pid = await _wait_for_pid(pid_path)

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "timeout"
        _assert_process_reaped(pid)

    asyncio.run(scenario())


def test_checker_executor_cancellation_kills_and_reaps_process(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=1)
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(checker_script), str(pid_path)),
        )
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        pid = await _wait_for_pid(pid_path)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        _assert_process_reaped(pid)

    asyncio.run(scenario())


def test_cancelling_capacity_waiter_does_not_cancel_running_checker(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=1)
        )
        blocking_adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(checker_script), str(pid_path)),
        )
        waiting_adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )
        request = _request(tmp_path)
        running = asyncio.create_task(executor.run(blocking_adapter, request))
        pid = await _wait_for_pid(pid_path)
        waiting = asyncio.create_task(executor.run(waiting_adapter, request))
        await asyncio.sleep(0)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        os.kill(pid, 0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        _assert_process_reaped(pid)

    asyncio.run(scenario())


def test_checker_executor_enforces_shared_concurrency_bound(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = _ConcurrencyProbeExecutor(
            CheckerExecutionPolicy(timeout_seconds=1.0, max_concurrency=1)
        )
        adapters = tuple(
            _CommandAdapter(executable=sys.executable, command=(sys.executable,))
            for _ in range(3)
        )

        result = await executor.run_all(adapters, _request(tmp_path))

        assert result == CheckerResult(diagnostics=(), failures=())
        assert executor.max_active == 1

    asyncio.run(scenario())


def test_checker_executor_reaps_siblings_when_one_adapter_fails(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
        wait_for_pid = (
            "import pathlib,time;"
            f"p=pathlib.Path({str(pid_path)!r});"
            "deadline=time.monotonic()+5;"
            "\nwhile not p.exists() and time.monotonic() < deadline: time.sleep(0.001)"
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=2)
        )
        adapters = (
            _CommandAdapter(
                executable=sys.executable,
                command=(sys.executable, str(checker_script), str(pid_path)),
            ),
            _FailingOutputAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", wait_for_pid),
            ),
        )

        with pytest.raises(RuntimeError, match="adapter parse failed"):
            await executor.run_all(adapters, _request(tmp_path))

        pid = await _wait_for_pid(pid_path)
        _assert_process_reaped(pid)

    asyncio.run(scenario())
