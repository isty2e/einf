import asyncio
import ctypes
import gc
import json
import os
import subprocess
import sys
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Literal, cast

import pytest

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerOutputLimits,
    CheckerRequest,
    CheckerResult,
    execution,
    windows_job,
)
from einf.analysis.checkers.windows_job import WindowsJob


class _KernelProbe:
    def __init__(self) -> None:
        self.job = 0x123456789
        self.event = 0x23456789A
        self.closed: list[int] = []
        self.fail: str | int | None = None

    def CreateJobObjectW(self, attributes: None, name: str) -> int:
        assert attributes is None
        assert name.startswith("Local\\einf-checker-")
        return 0 if self.fail == "create" else self.job

    def SetInformationJobObject(
        self, handle: int, kind: int, value: object, size: int
    ) -> int:
        assert handle == self.job and kind == 9
        assert size == ctypes.sizeof(windows_job._ExtendedLimits)
        limits = getattr(value, "_obj", None)
        assert isinstance(limits, windows_job._ExtendedLimits)
        assert limits.BasicLimitInformation.LimitFlags == 0x2000
        return int(self.fail != "configure")

    def CreateEventW(
        self, attributes: None, manual: bool, initial: bool, name: str
    ) -> int:
        assert attributes is None and manual and not initial
        assert name.endswith("-started")
        return 0 if self.fail == "event" else self.event

    def CloseHandle(self, handle: int) -> int:
        self.closed.append(handle)
        return int(self.fail != handle)


@pytest.mark.parametrize("failure", ["create", "configure", "event"])
def test_windows_job_closes_only_acquired_handles_on_setup_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    kernel = _KernelProbe()
    kernel.fail = failure
    monkeypatch.setattr(
        windows_job, "_load_kernel32", lambda: cast(ctypes.CDLL, kernel)
    )

    with pytest.raises(OSError):
        WindowsJob()

    assert kernel.closed == ([] if failure == "create" else [kernel.job])


def test_windows_job_attempts_both_closes_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _KernelProbe()
    monkeypatch.setattr(
        windows_job, "_load_kernel32", lambda: cast(ctypes.CDLL, kernel)
    )
    job = WindowsJob()
    kernel.fail = kernel.event

    with pytest.raises(OSError):
        job.close()
    assert kernel.closed == [kernel.event, kernel.job]

    kernel.fail = None
    job.close()
    job.close()
    assert kernel.closed == [kernel.event, kernel.job, kernel.event]


def test_windows_job_close_failure_does_not_replace_cancellation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    kernel = _KernelProbe()
    monkeypatch.setattr(
        windows_job, "_load_kernel32", lambda: cast(ctypes.CDLL, kernel)
    )
    job = WindowsJob()
    kernel.fail = kernel.event
    original = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as caught, job:
        raise original
    assert caught.value is original
    assert kernel.closed == [kernel.event, kernel.job]
    assert "handle cleanup failed" in caplog.text
    kernel.fail = None
    job.close()


def test_windows_launcher_rejects_failed_membership_before_invocation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject_join(name: str) -> None:
        raise OSError("job assignment denied")

    def unexpected_invoke(*args: object, **kwargs: object) -> None:
        pytest.fail("checker executed without Job membership")

    monkeypatch.setattr(windows_job, "_join_job", reject_join)
    monkeypatch.setattr(subprocess, "Popen", unexpected_invoke)
    monkeypatch.setattr(sys, "argv", ["launcher", "unavailable-job", "checker"])
    assert windows_job._main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "job assignment denied" in captured.err


class _ScopeProbe(WindowsJob):
    def __init__(self) -> None:
        self.terminations = 0
        self.queries = 0
        self.empty = True
        self.started = True
        self.closed = False
        self.query_error = False
        self.queried = asyncio.Event()

    def launch_command(self, command: tuple[str, ...]) -> tuple[str, ...]:
        return command

    def terminate(self) -> None:
        self.terminations += 1

    def is_empty(self) -> bool:
        self.queries += 1
        self.queried.set()
        if self.query_error:
            raise OSError("Job query failed")
        return self.empty

    def checker_started(self) -> bool:
        return self.started

    def close(self) -> None:
        self.closed = True


def test_windows_cleanup_proof_waits_for_launcher_exit_and_scope_emptiness() -> None:
    async def scenario() -> None:
        exited = asyncio.Event()
        scope = _ScopeProbe()
        scope.empty = False

        async def wait_for_exit() -> int:
            await exited.wait()
            return 0

        process_wait = asyncio.create_task(wait_for_exit())
        cleanup = asyncio.create_task(
            execution._wait_for_windows_job(process_wait, scope)
        )
        await asyncio.sleep(0)
        assert scope.queries == 0 and scope.terminations == 0
        assert not cleanup.done()

        exited.set()
        await asyncio.wait_for(scope.queried.wait(), timeout=5)
        assert scope.terminations == 1
        assert not cleanup.done()
        scope.empty = True
        await asyncio.wait_for(cleanup, timeout=5)

    asyncio.run(scenario())


class _RecordingAdapter(CheckerAdapter):
    name = "probe"
    executable = sys.executable

    def __init__(self, command: tuple[str, ...]) -> None:
        self.command = command
        self.outputs: list[tuple[str, str]] = []

    def build_command(self, request: CheckerRequest, /) -> tuple[str, ...]:
        return self.command

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        self.outputs.append((stdout, stderr))
        return CheckerResult(diagnostics=(), failures=())


def _request(root: Path) -> CheckerRequest:
    return CheckerRequest(project_root=root, targets=(root / "sample.py",))


@pytest.mark.parametrize("started", [False, True])
def test_windows_executor_closes_scope_and_preserves_launch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, started: bool
) -> None:
    scope = _ScopeProbe()
    scope.started = started
    monkeypatch.setattr(windows_job, "WindowsJob", lambda: scope)
    monkeypatch.setattr(execution, "os", SimpleNamespace(name="nt"))
    adapter = _RecordingAdapter((sys.executable, "-c", "print('payload')"))
    result = asyncio.run(
        CheckerExecutor(CheckerExecutionPolicy()).run(adapter, _request(tmp_path))
    )

    assert scope.closed and scope.terminations >= 2 and scope.queries >= 1
    if started:
        assert result.failures == ()
        assert adapter.outputs == [("payload" + os.linesep, "")]
    else:
        assert len(result.failures) == 1 and result.failures[0].kind == "spawn_error"
        assert adapter.outputs == []


@pytest.mark.parametrize("query_error", [False, True])
def test_windows_executor_rejects_unproven_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, query_error: bool
) -> None:
    scope = _ScopeProbe()
    scope.empty = False
    scope.query_error = query_error
    monkeypatch.setattr(windows_job, "WindowsJob", lambda: scope)
    monkeypatch.setattr(execution, "os", SimpleNamespace(name="nt"))
    adapter = _RecordingAdapter((sys.executable, "-c", ""))
    executor = CheckerExecutor(CheckerExecutionPolicy(cleanup_timeout_seconds=0.01))
    result = asyncio.run(executor.run(adapter, _request(tmp_path)))

    assert scope.closed
    assert adapter.outputs == []
    assert len(result.failures) == 1 and result.failures[0].kind == "execution_error"
    assert "cleanup did not complete" in result.failures[0].message


def test_windows_spawn_cancellation_closes_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        scope = _ScopeProbe()
        entered = asyncio.Event()
        monkeypatch.setattr(windows_job, "WindowsJob", lambda: scope)
        monkeypatch.setattr(execution, "os", SimpleNamespace(name="nt"))

        async def stalled_spawn(*command: str, **options: object) -> None:
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_spawn)
        adapter = _RecordingAdapter((sys.executable, "-c", ""))
        task = asyncio.create_task(
            CheckerExecutor(CheckerExecutionPolicy()).run(adapter, _request(tmp_path))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scope.closed
        assert adapter.outputs == []

    asyncio.run(scenario())


def test_repeated_windows_cancellation_waits_for_scope_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        scope = _ScopeProbe()
        scope.empty = False
        monkeypatch.setattr(windows_job, "WindowsJob", lambda: scope)
        monkeypatch.setattr(execution, "os", SimpleNamespace(name="nt"))
        pid_path = tmp_path / "direct.pid"
        source = (
            "import os, time; from pathlib import Path; "
            f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
        )
        adapter = _RecordingAdapter((sys.executable, "-c", source))
        executor = CheckerExecutor(CheckerExecutionPolicy(cleanup_timeout_seconds=5))
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        try:
            await _wait_pid(pid_path)
            task.cancel()
            await asyncio.wait_for(scope.queried.wait(), timeout=5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not scope.closed
            scope.empty = True
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            assert scope.closed
        finally:
            scope.empty = True
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
@pytest.mark.parametrize("exit_code", [0, 1, 7, 0xC0000005])
def test_windows_launcher_preserves_argv_environment_and_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    monkeypatch.setenv("EINF_CHECKER_PROBE", "inherited")
    script = (
        "import ctypes, json, os, sys; "
        "print(json.dumps([sys.argv[1:], os.getcwd(), os.environ['EINF_CHECKER_PROBE']]), flush=True); "
        "print('checker stderr', file=sys.stderr, flush=True); "
        f"ctypes.WinDLL('kernel32').ExitProcess(ctypes.c_uint32({exit_code}))"
    )
    with WindowsJob() as job:
        completed = subprocess.run(
            job.launch_command(
                (sys.executable, "-c", script, "argument with spaces", "한글")
            ),
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert job.checker_started()
        assert completed.returncode == exit_code
        assert json.loads(completed.stdout) == [
            ["argument with spaces", "한글"],
            str(tmp_path),
            "inherited",
        ]
        assert completed.stderr == "checker stderr\n"
        assert job.is_empty()


@pytest.mark.skipif(os.name != "nt", reason="native Windows error state")
def test_windows_api_failure_uses_ctypes_error_snapshot() -> None:
    if sys.platform == "win32":
        native = ctypes.WinDLL("kernel32", use_last_error=False)
        native.SetLastError.argtypes = [ctypes.c_uint32]
        native.SetLastError.restype = None
        previous = ctypes.get_last_error()
        ctypes.set_last_error(5)
        native.SetLastError(87)
        try:
            assert getattr(windows_job._win_error(), "winerror", None) == 5
        finally:
            ctypes.set_last_error(previous)


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
def test_windows_launcher_cannot_run_without_parent_job(tmp_path: Path) -> None:
    marker = tmp_path / "unexpected-body"
    with WindowsJob() as job:
        command = job.launch_command(
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            )
        )
    completed = subprocess.run(
        command, cwd=tmp_path, check=False, capture_output=True, timeout=15
    )
    assert completed.returncode != 0
    assert b"Windows checker launcher failed" in completed.stderr
    assert not marker.exists()


async def _wait_pid(path: Path) -> int:
    async def read() -> int:
        while True:
            try:
                return int(path.read_text())
            except (OSError, ValueError):
                await asyncio.sleep(0.01)

    return await asyncio.wait_for(read(), timeout=10)


class _WindowsProcessProbe(AbstractContextManager["_WindowsProcessProbe"]):
    def __init__(self, pid: int) -> None:
        self.kernel = windows_job._load_kernel32()
        self.kernel.OpenProcess.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
        ]
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel.TerminateProcess.restype = ctypes.c_int32
        self.handle = self.kernel.OpenProcess(0x00100001, False, pid)
        if not self.handle:
            raise windows_job._win_error()

    def has_exited(self) -> bool:
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status == 0:
            return True
        if status == 258:
            return False
        raise windows_job._win_error()

    async def wait_exited(self) -> None:
        async def observe() -> None:
            while not self.has_exited():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(observe(), timeout=5)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if not self.has_exited():
                terminated = self.kernel.TerminateProcess(self.handle, 1)
                error = None if terminated else windows_job._win_error()
                exited = self.kernel.WaitForSingleObject(self.handle, 5000) == 0
                if not exited and error is not None:
                    raise error
                assert exited, "probe process did not exit within five seconds"
        finally:
            assert self.kernel.CloseHandle(self.handle)
        return False


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
@pytest.mark.parametrize(
    ("finish", "parent_exits"),
    [
        ("timeout", False),
        ("timeout", True),
        ("cancel", False),
        ("cancel", True),
        ("normal", True),
        ("overflow", True),
    ],
)
def test_windows_executor_reaps_owned_descendants(
    tmp_path: Path,
    finish: Literal["timeout", "cancel", "normal", "overflow"],
    *,
    parent_exits: bool,
) -> None:
    async def scenario() -> None:
        parent_pid_path = tmp_path / "parent.pid"
        child_pid_path = tmp_path / "child.pid"
        grandchild_pid_path = tmp_path / "grandchild.pid"
        release = tmp_path / "release-checker"
        grandchild = (
            "import os, time; from pathlib import Path; "
            f"Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        child = (
            "import os, subprocess, sys, time; from pathlib import Path; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        redirection = (
            ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
            if finish in {"normal", "overflow"}
            else ""
        )
        parent = (
            "import os, subprocess, sys, time\nfrom pathlib import Path\n"
            f"Path({str(parent_pid_path)!r}).write_text(str(os.getpid()))\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}]{redirection})\n"
            f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
            + ("print('x' * 8192, flush=True)\n" if finish == "overflow" else "")
            + ("time.sleep(60)\n" if not parent_exits else "")
        )
        adapter = _RecordingAdapter((sys.executable, "-c", parent))
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=15 if finish == "timeout" else 30,
                cleanup_timeout_seconds=2,
                max_output_bytes=1024 if finish == "overflow" else 8192,
            )
        )
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        with ExitStack() as resources:
            try:
                parent_pid = await _wait_pid(parent_pid_path)
                checker = resources.enter_context(_WindowsProcessProbe(parent_pid))
                child_pid = await _wait_pid(child_pid_path)
                child_probe = resources.enter_context(_WindowsProcessProbe(child_pid))
                grandchild_pid = await _wait_pid(grandchild_pid_path)
                grandchild_probe = resources.enter_context(
                    _WindowsProcessProbe(grandchild_pid)
                )
                release.touch()
                if parent_exits:
                    await checker.wait_exited()
                if finish == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=5)
                else:
                    result = await asyncio.wait_for(task, timeout=25)
                    if finish == "normal":
                        assert result.failures == ()
                    else:
                        expected = (
                            "timeout"
                            if finish == "timeout"
                            else "output_limit_exceeded"
                        )
                        assert (
                            len(result.failures) == 1
                            and result.failures[0].kind == expected
                        )
                        assert (
                            "cleanup did not complete" not in result.failures[0].message
                        )
                await child_probe.wait_exited()
                await grandchild_probe.wait_exited()
                assert checker.has_exited()
                assert child_probe.has_exited()
                assert grandchild_probe.has_exited()
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
def test_windows_launcher_is_independent_of_package_site_and_workspace_imports(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "unexpected-import"
    for name in ("ctypes.py", "logging.py", "sitecustomize.py"):
        (tmp_path / name).write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
            encoding="utf-8",
        )
    guard = (
        "import sys\n"
        "class PackageGuard:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'einf' or name.startswith('einf.'):\n"
        "            raise RuntimeError('launcher imported einf')\n"
        "sys.meta_path.insert(0, PackageGuard())\n"
    )
    standalone = tmp_path / "standalone launcher.py"
    standalone.write_text(
        guard + Path(windows_job.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    with WindowsJob() as job:
        command = list(
            job.launch_command((sys.executable, "-I", "-S", "-c", "print('isolated')"))
        )
        command[3] = str(standalone)
        result = subprocess.run(
            command, cwd=tmp_path, check=True, capture_output=True, timeout=15
        )
        assert result.stdout.strip() == b"isolated"
        assert job.checker_started() and job.is_empty()
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
def test_windows_checker_runs_do_not_accumulate_native_handles(tmp_path: Path) -> None:
    async def scenario() -> None:
        kernel = windows_job._load_kernel32()
        kernel.GetProcessHandleCount.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        kernel.GetProcessHandleCount.restype = ctypes.c_int32

        def handle_count() -> int:
            count = ctypes.c_uint32()
            assert kernel.GetProcessHandleCount(
                kernel.GetCurrentProcess(), ctypes.byref(count)
            )
            return count.value

        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _RecordingAdapter((sys.executable, "-c", ""))
        assert not (await executor.run(adapter, _request(tmp_path))).failures
        await asyncio.sleep(0)
        gc.collect()
        baseline = handle_count()
        for _ in range(5):
            assert not (await executor.run(adapter, _request(tmp_path))).failures
            await asyncio.sleep(0)
            gc.collect()
        assert handle_count() <= baseline + 2

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job integration")
def test_windows_cancellation_before_job_join_reaps_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        launcher_pid = tmp_path / "launcher.pid"
        checker_marker = tmp_path / "checker-started"
        delayed = tmp_path / "delayed launcher.py"
        delayed.write_text(
            "import os, time\nfrom pathlib import Path\n"
            f"Path({str(launcher_pid)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n" + Path(windows_job.__file__).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        original = WindowsJob.launch_command

        def launch(self: WindowsJob, command: tuple[str, ...]) -> tuple[str, ...]:
            arguments = list(original(self, command))
            arguments[3] = str(delayed)
            return tuple(arguments)

        monkeypatch.setattr(WindowsJob, "launch_command", launch)
        adapter = _RecordingAdapter(
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(checker_marker)!r}).touch()",
            )
        )
        executor = CheckerExecutor(CheckerExecutionPolicy(cleanup_timeout_seconds=2))
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        with ExitStack() as resources:
            try:
                pid = await _wait_pid(launcher_pid)
                probe = resources.enter_context(_WindowsProcessProbe(pid))
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=5)
                assert probe.has_exited()
                assert not checker_marker.exists()
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
