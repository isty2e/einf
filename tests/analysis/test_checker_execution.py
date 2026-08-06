import asyncio
import gc
import json
import os
import signal
import sys
import tracemalloc
from pathlib import Path
from typing import Protocol

import pytest

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerDiagnostic,
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerFailure,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.checkers.execution import bounded_communicate
from einf.analysis.checkers.json_limited import load_list_field_limited
from einf.analysis.checkers.model import CheckerOutputLimits
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.checkers.registry import build_checker_adapters
from einf.analysis.checkers.ty import TyAdapter
from einf.analysis.model import TextPosition, TextSpan

_CHECKER_STARTUP_BOUND_SECONDS = 10.0
_CHECKER_TIMEOUT_MARGIN_SECONDS = 5.0


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
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, request, limits
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
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, request, limits
        raise RuntimeError("adapter parse failed")


class _DiagnosticOutputAdapter(_CommandAdapter):
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, limits
        target = request.targets[0]
        return CheckerResult(
            diagnostics=(
                CheckerDiagnostic(
                    tool=self.name,
                    path=target,
                    code="sibling-rule",
                    message="sibling diagnostic",
                    severity="warning",
                    span=TextSpan(
                        start=TextPosition(line=1, column=0),
                        end=TextPosition(line=1, column=1),
                    ),
                ),
            ),
            failures=(),
        )


class _FailingBuildCommandAdapter(_CommandAdapter):
    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        _ = request
        raise RuntimeError("adapter build failed")


class _ManyDiagnosticsAdapter(_CommandAdapter):
    def __init__(
        self,
        *,
        executable: str,
        command: tuple[str, ...],
        count: int,
    ) -> None:
        super().__init__(executable=executable, command=command)
        self._count = count

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, limits
        target = request.targets[0]
        return CheckerResult(
            diagnostics=tuple(
                CheckerDiagnostic(
                    tool=self.name,
                    path=target,
                    code=f"rule-{index}",
                    message=f"diagnostic {index}",
                    severity="warning",
                    span=TextSpan(
                        start=TextPosition(line=index + 1, column=0),
                        end=TextPosition(line=index + 1, column=1),
                    ),
                )
                for index in range(self._count)
            ),
            failures=(),
        )


class _FieldLengthAdapter(_CommandAdapter):
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, limits
        target = request.targets[0]
        return CheckerResult(
            diagnostics=(
                CheckerDiagnostic(
                    tool=self.name,
                    path=target,
                    code="long-message",
                    message="a" * 64,
                    severity="warning",
                    span=TextSpan(
                        start=TextPosition(line=1, column=0),
                        end=TextPosition(line=1, column=1),
                    ),
                ),
            ),
            failures=(),
        )


class _RuntimeErrorStreamReader:
    """StreamReader stand-in that raises an unexpected error on first read."""

    async def read(self, size: int) -> bytes:
        _ = size
        raise RuntimeError("reader failed unexpectedly")


class _FragmentedStreamReader:
    """StreamReader stand-in that yields many tiny chunks."""

    def __init__(self, *, chunk_count: int) -> None:
        self._remaining = chunk_count

    async def read(self, size: int) -> bytes:
        _ = size
        if self._remaining <= 0:
            return b""
        self._remaining -= 1
        return b"x"


class _LongFailureMessageAdapter(_CommandAdapter):
    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
        limits: CheckerOutputLimits | None = None,
    ) -> CheckerResult:
        _ = stdout, stderr, request, limits
        return CheckerResult(
            diagnostics=(),
            failures=(
                CheckerFailure(
                    tool=self.name,
                    kind="output_parse_error",
                    message="unrecognized " + "x" * 4096,
                ),
            ),
        )


class _StringStreamReader:
    """StreamReader stand-in that yields one fixed chunk then EOF."""

    def __init__(self, content: str) -> None:
        self._content = content
        self._served = False

    async def read(self, size: int) -> bytes:
        _ = size
        if self._served:
            return b""
        self._served = True
        return self._content.encode()


class _ReaderProtocol(Protocol):
    async def read(self, size: int) -> bytes: ...


class _LegacySignatureAdapter(_CommandAdapter):
    def parse_output(  # type: ignore[override]
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr, request
        return CheckerResult(diagnostics=(), failures=())


class _RecordingSubprocessTransport(asyncio.SubprocessTransport):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _StalledStreamReader:
    """StreamReader stand-in that blocks until released or fails on first read."""

    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        error: OSError | None = None,
    ) -> None:
        self._started = started
        self._release = release
        self._error = error
        self._eof = False

    async def read(self, size: int) -> bytes:
        _ = size
        if self._error is not None:
            raise self._error
        if self._eof:
            return b""
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        self._eof = True
        return b""


class _CommunicateErrorProcess:
    def __init__(self) -> None:
        self._transport = _RecordingSubprocessTransport()
        self.pid = 100_001
        self.returncode: int | None = None
        self.killed = False
        self.waited = False
        error = OSError("pipe failed")
        self.stdout: _ReaderProtocol = _StalledStreamReader(error=error)
        self.stderr: _ReaderProtocol = _StalledStreamReader(error=error)

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


class _SlowCleanupProcess:
    def __init__(self) -> None:
        self._transport = _RecordingSubprocessTransport()
        self.pid = 100_002
        self.returncode: int | None = None
        self.communicate_started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.finish_cleanup = asyncio.Event()
        self.reaped = asyncio.Event()
        self.stdout = _StalledStreamReader(
            started=self.communicate_started,
            release=self.finish_cleanup,
        )
        self.stderr = _StalledStreamReader(release=self.finish_cleanup)

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


def _write_checker_with_descendant(
    path: Path,
    *,
    detach_child: bool = False,
    parent_sleep_seconds: float,
    child_sleep_seconds: float = 5.0,
) -> None:
    child_session_option = "    start_new_session=True,\n" if detach_child else ""
    path.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "pid_path = Path(sys.argv[1])\n"
        "pending_path = pid_path.with_suffix('.tmp')\n"
        "pending_path.write_text(str(os.getpid()), encoding='utf-8')\n"
        "child = subprocess.Popen(\n"
        f"    [sys.executable, '-c', 'import time; time.sleep({child_sleep_seconds!r})'],\n"
        f"{child_session_option}"
        ")\n"
        "pending_path.write_text(str(child.pid), encoding='utf-8')\n"
        "pending_path.replace(pid_path)\n"
        f"time.sleep({parent_sleep_seconds!r})\n",
        encoding="utf-8",
    )


async def _wait_for_pid(
    path: Path,
    *,
    timeout_seconds: float = _CHECKER_STARTUP_BOUND_SECONDS,
) -> int:
    """Wait for the checker to publish its pid under a bounded readiness deadline.

    The pid file is the readiness signal and is written atomically
    (pending_path.replace) after the checker interpreter starts and spawns
    its descendant. The deadline is an explicit bound on that startup
    sequence, not a guess at nominal latency: ~20x a cold interpreter spawn
    on a loaded CI worker.
    """
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        if path.exists():
            return int(path.read_text(encoding="utf-8"))
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                "checker process did not publish its pid within "
                f"{timeout_seconds:g} seconds"
            )
        await asyncio.sleep(0.005)


def _assert_process_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def _assert_process_exited(pid: int) -> None:
    for _ in range(1_000):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"process {pid} was not reaped")


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


@pytest.mark.parametrize(
    "cleanup_timeout_seconds",
    [0.0, float("nan"), float("inf"), True, "1"],
)
def test_checker_execution_policy_rejects_invalid_cleanup_timeout(
    cleanup_timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        CheckerExecutionPolicy(cleanup_timeout_seconds=cleanup_timeout_seconds)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_concurrency", 0),
        ("max_output_bytes", -1),
        ("max_output_bytes", True),
        ("max_diagnostics", 0),
        ("max_field_length", True),
    ],
)
def test_checker_execution_policy_rejects_invalid_resource_bounds(
    field: str,
    bad_value: object,
) -> None:
    kwargs: dict[str, object] = {field: bad_value}
    with pytest.raises(ValueError):
        CheckerExecutionPolicy(**kwargs)  # type: ignore[arg-type]


def test_checker_executor_prefers_timeout_when_runaway_output_never_ends(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runaway_script = tmp_path / "runaway_checker.py"
        runaway_script.write_text(
            "import sys, time\n"
            "sys.stdout.write('x' * 4096)\n"
            "sys.stdout.flush()\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=0.2,
                max_output_bytes=1024,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(runaway_script)),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "timeout"

    asyncio.run(scenario())


def test_checker_executor_reports_stream_error_not_timeout_when_sibling_stalls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _RuntimeErrorStreamReader()
        process.stderr = _StalledStreamReader()

        async def create_process(
            *command: str,
            **options: object,
        ) -> _CommunicateErrorProcess:
            _ = command, options
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=0.01,
                cleanup_timeout_seconds=0.01,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert len(result.failures) == 1
        assert result.failures[0].kind == "execution_error"
        assert "reader failed unexpectedly" in result.failures[0].message
        assert process.killed is True
        assert process.waited is True
        assert process._transport.closed is True

    asyncio.run(scenario())


def test_bounded_communicate_uses_combined_stream_budget() -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _FragmentedStreamReader(chunk_count=3)
        process.stderr = _FragmentedStreamReader(chunk_count=3)

        stdout_bytes, stderr_bytes, exceeded = await bounded_communicate(
            process,  # type: ignore[arg-type]
            max_output_bytes=4,
        )

        assert exceeded is True
        assert len(stdout_bytes) + len(stderr_bytes) <= 4

    asyncio.run(scenario())


def test_bounded_communicate_cancellation_reaps_reader_tasks(
    caplog,
) -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _StalledStreamReader()
        process.stderr = _StalledStreamReader()

        communication = asyncio.create_task(
            bounded_communicate(process, max_output_bytes=1024)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        communication.cancel()
        with pytest.raises(asyncio.CancelledError):
            await communication

    asyncio.run(scenario())
    gc.collect()
    assert "Task was destroyed but it is pending" not in caplog.text


def test_bounded_communicate_retrieves_simultaneous_stream_errors(
    caplog,
) -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _RuntimeErrorStreamReader()
        process.stderr = _RuntimeErrorStreamReader()

        with pytest.raises(RuntimeError, match="reader failed unexpectedly"):
            await bounded_communicate(process, max_output_bytes=1024)  # type: ignore[arg-type]

    asyncio.run(scenario())
    gc.collect()
    assert "exception was never retrieved" not in caplog.text


def test_bounded_communicate_skips_large_metadata_without_materializing() -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _StringStreamReader("x" * 1_000_000)
        process.stderr = _StringStreamReader("")

        tracemalloc.start()
        try:
            stdout_bytes, _, exceeded = await bounded_communicate(
                process,  # type: ignore[arg-type]
                max_output_bytes=8 * 1024 * 1024,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert not exceeded
        assert len(stdout_bytes) == 1_000_000
        assert peak < 3_000_000

    asyncio.run(scenario())


def test_pyright_adapter_rejects_oversized_raw_path_before_normalization() -> None:
    long_path = "../" * 40 + "sample.py"
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": long_path,
                    "severity": "warning",
                    "message": "m",
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 1},
                    },
                }
            ]
        }
    )

    result = PyrightAdapter(name="pyright", executable="pyright").parse_output(
        stdout=payload,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=64),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"


def test_checker_executor_reaps_process_on_unexpected_communication_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _RuntimeErrorStreamReader()
        process.stderr = _RuntimeErrorStreamReader()

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
        assert "reader failed unexpectedly" in result.failures[0].message
        assert process.killed is True
        assert process.waited is True
        assert process._transport.closed is True

    asyncio.run(scenario())


def test_bounded_communicate_keeps_peak_memory_proportional_to_limit() -> None:
    async def scenario() -> None:
        process = _CommunicateErrorProcess()
        process.stdout = _FragmentedStreamReader(chunk_count=20_000)
        process.stderr = _FragmentedStreamReader(chunk_count=20_000)

        tracemalloc.start()
        try:
            stdout_bytes, stderr_bytes, exceeded = await bounded_communicate(
                process,  # type: ignore[arg-type]
                max_output_bytes=8 * 1024 * 1024,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert not exceeded
        assert len(stdout_bytes) == 20_000
        assert len(stderr_bytes) == 20_000
        assert peak < 200_000

    asyncio.run(scenario())


def test_ty_adapter_stops_parsing_at_diagnostic_limit() -> None:
    many_lines = "".join(
        f"/tmp/sample.py:1:{index}: error[rule-{index}] message {index}\n"
        for index in range(1, 5_000)
    )
    result = TyAdapter().parse_output(
        stdout=many_lines,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=2, max_field_length=4096),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"
    assert "2 diagnostics" in result.failures[0].message


def test_checker_executor_fails_closed_on_diagnostic_code_length(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_field_length=8,
            )
        )
        adapter = _FieldLengthAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_limit_exceeded"

    asyncio.run(scenario())


def test_checker_executor_truncates_oversized_failure_messages(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_field_length=64,
            )
        )
        adapter = _LongFailureMessageAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_parse_error"
        assert len(result.failures[0].message) <= 64

    asyncio.run(scenario())


def test_run_all_caps_aggregate_diagnostics(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_concurrency=2,
                max_diagnostics=2,
            )
        )
        adapters = (
            _ManyDiagnosticsAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", "pass"),
                count=2,
            ),
            _ManyDiagnosticsAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", "pass"),
                count=2,
            ),
        )

        result = await executor.run_all(adapters, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_limit_exceeded"
        assert "aggregate" in result.failures[0].message

    asyncio.run(scenario())


def test_build_checker_adapters_collapses_duplicate_names() -> None:
    adapters = build_checker_adapters(("ty", "pyright", "ty", "basedpyright"))

    assert tuple(adapter.name for adapter in adapters) == (
        "ty",
        "pyright",
        "basedpyright",
    )


def test_checker_executor_keeps_legacy_adapter_contract_working(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_diagnostics=1,
            )
        )
        adapter = _LegacySignatureAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert result.failures == ()

    asyncio.run(scenario())


def test_load_list_field_limited_stops_at_entry_cap() -> None:
    payload = '{"generalDiagnostics": [{"a": 1}, {"b": 2}, {"c": 3}]}'

    entries, truncated, parse_error = load_list_field_limited(
        payload,
        field="generalDiagnostics",
        max_entries=2,
    )

    assert parse_error is None
    assert truncated is True
    assert len(entries) == 2


def test_load_list_field_limited_accepts_exactly_cap_entries() -> None:
    payload = '{"generalDiagnostics": [{"a": 1}, {"b": 2}]}'

    entries, truncated, parse_error = load_list_field_limited(
        payload,
        field="generalDiagnostics",
        max_entries=2,
    )

    assert parse_error is None
    assert truncated is False
    assert len(entries) == 2


def test_load_list_field_limited_handles_braces_inside_strings() -> None:
    payload = '{"summary": "a } b { c", "errors": [{"x": 1}, {"y": 2}]}'

    entries, truncated, parse_error = load_list_field_limited(
        payload,
        field="errors",
        max_entries=10,
    )

    assert parse_error is None
    assert truncated is False
    assert len(entries) == 2
    assert entries[0] == {"x": 1}


def test_load_list_field_limited_reports_missing_field() -> None:
    entries, truncated, parse_error = load_list_field_limited(
        '{"summary": "ok"}',
        field="errors",
        max_entries=10,
    )

    assert entries == []
    assert truncated is False
    assert parse_error is not None
    assert "missing" in parse_error


def test_load_list_field_limited_reports_malformed_json() -> None:
    for payload in ("not json", "{", '{"errors": [1, 2}'):
        entries, truncated, parse_error = load_list_field_limited(
            payload,
            field="errors",
            max_entries=10,
        )
        assert entries == []
        assert truncated is False
        assert parse_error is not None


def test_load_list_field_limited_rejects_trailing_content() -> None:
    entries, truncated, parse_error = load_list_field_limited(
        '{"generalDiagnostics": []} trailing',
        field="generalDiagnostics",
        max_entries=10,
    )

    assert entries == []
    assert truncated is False
    assert parse_error is not None
    assert "trailing" in parse_error


def test_load_list_field_limited_rejects_trailing_comma() -> None:
    for payload in ('{"generalDiagnostics": [1, 2,]}', '{"errors": [1, 2, 3]}'):
        entries, truncated, parse_error = load_list_field_limited(
            payload,
            field="generalDiagnostics" if "generalDiagnostics" in payload else "errors",
            max_entries=10,
        )
        assert truncated is False
        if "generalDiagnostics" in payload:
            assert parse_error is not None
        else:
            assert parse_error is None
            assert len(entries) == 3


def test_load_list_field_limited_rejects_duplicate_target_field() -> None:
    payload = '{"errors": [1], "errors": [2]}'

    _, truncated, parse_error = load_list_field_limited(
        payload,
        field="errors",
        max_entries=10,
    )

    assert truncated is False
    assert parse_error is not None
    assert "repeats" in parse_error


def test_load_list_field_limited_rejects_strictness_violations() -> None:
    cases = (
        '{"errors": [],}',
        '{"meta": [1,], "errors": []}',
        '{"meta": {"x": 1,}, "errors": []}',
        '{"meta": "bad\\q", "errors": []}',
        '{"meta": "line' + chr(10) + 'break", "errors": []}',
        '{"meta": "\\u12xz", "errors": []}',
    )
    for payload in cases:
        entries, truncated, parse_error = load_list_field_limited(
            payload,
            field="errors",
            max_entries=10,
        )
        assert entries == [], payload
        assert truncated is False, payload
        assert parse_error is not None, payload


def test_load_list_field_limited_normalizes_deep_and_huge_inputs() -> None:
    deep_payload = '{"errors": ' + "[" * 2_000 + "]" * 2_000 + "}"
    huge_int_payload = '{"meta": ' + "9" * 5_000 + ', "errors": []}'

    for payload in (deep_payload, huge_int_payload):
        entries, truncated, parse_error = load_list_field_limited(
            payload,
            field="errors",
            max_entries=10,
        )
        assert entries == []
        assert truncated is False
        assert parse_error is not None
        assert "not valid JSON" in parse_error


def test_line_adapter_drops_partial_diagnostics_on_field_limit() -> None:
    long_path = "x" * 101
    payload = (
        f"/tmp/sample.py:1:1: error[rule-ok] fine\n"
        f"{long_path}:1:1: error[rule-long] oversized\n"
    )
    result = TyAdapter().parse_output(
        stdout=payload,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=100),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"


def test_json_adapter_drops_partial_diagnostics_on_field_limit() -> None:
    long_path = "../" * 40 + "sample.py"
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": "/tmp/sample.py",
                    "severity": "warning",
                    "message": "fine",
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 1},
                    },
                },
                {
                    "file": long_path,
                    "severity": "warning",
                    "message": "oversized",
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 1},
                    },
                },
            ]
        }
    )
    result = PyrightAdapter(name="pyright", executable="pyright").parse_output(
        stdout=payload,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=64),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"


def test_load_list_field_limited_rejects_non_standard_constants() -> None:
    for payload in (
        '{"meta": NaN, "errors": []}',
        '{"meta": Infinity, "errors": []}',
        '{"meta": -Infinity, "errors": []}',
        '{"errors": [1, NaN, 3]}',
    ):
        _, truncated, parse_error = load_list_field_limited(
            payload,
            field="errors",
            max_entries=2,
        )
        assert truncated is False, payload
        assert parse_error is not None, payload


def test_load_list_field_limited_rejects_constants_in_capped_remainder() -> None:
    payload = '{"errors": [1, 2, NaN]}'

    entries, truncated, parse_error = load_list_field_limited(
        payload,
        field="errors",
        max_entries=2,
    )

    assert entries == []
    assert truncated is False
    assert parse_error is not None


def test_line_adapter_field_limit_wins_over_prior_parse_failure() -> None:
    long_path = "x" * 101
    payload = (
        f"/tmp/sample.py:1:1: error[rule-ok] fine\n"
        f"not a diagnostic line\n"
        f"{long_path}:1:1: error[rule-long] oversized\n"
    )
    result = TyAdapter().parse_output(
        stdout=payload,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=100),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"


def test_json_adapter_field_limit_wins_over_prior_parse_failure() -> None:
    long_path = "../" * 40 + "sample.py"
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": "/tmp/sample.py",
                    "severity": "warning",
                    "message": "fine",
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 1},
                    },
                },
                {"file": "sample.py"},
                {
                    "file": long_path,
                    "severity": "warning",
                    "message": "oversized",
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 1},
                    },
                },
            ]
        }
    )
    result = PyrightAdapter(name="pyright", executable="pyright").parse_output(
        stdout=payload,
        stderr="",
        request=_request(Path("/tmp")),
        limits=CheckerOutputLimits(max_diagnostics=10, max_field_length=64),
    )

    assert result.diagnostics == ()
    assert len(result.failures) == 1
    assert result.failures[0].kind == "output_limit_exceeded"


def test_checker_executor_truncates_failures_preserved_by_limit_path(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_field_length=8,
            )
        )
        adapter = _LongFailureMessageAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert len(result.failures) == 1
        assert len(result.failures[0].message) <= 8

    asyncio.run(scenario())


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


def test_checker_executor_fails_closed_when_output_exceeds_byte_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        noisy_script = tmp_path / "noisy_checker.py"
        noisy_script.write_text(
            "import sys\nsys.stdout.write('x' * 8192)\nsys.stdout.flush()\n",
            encoding="utf-8",
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_output_bytes=1024,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(noisy_script)),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_limit_exceeded"
        assert "1024 bytes" in result.failures[0].message

    asyncio.run(scenario())


def test_checker_executor_accepts_output_at_byte_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        exact_script = tmp_path / "exact_checker.py"
        exact_script.write_text(
            "import sys\nsys.stdout.write('x' * 512)\nsys.stdout.flush()\n",
            encoding="utf-8",
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_output_bytes=512,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(exact_script)),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert result.failures == ()

    asyncio.run(scenario())


def test_checker_executor_fails_closed_on_diagnostic_count_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_diagnostics=2,
            )
        )
        adapter = _ManyDiagnosticsAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
            count=3,
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_limit_exceeded"
        assert "2 diagnostics" in result.failures[0].message

    asyncio.run(scenario())


def test_checker_executor_fails_closed_on_diagnostic_field_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                max_field_length=8,
            )
        )
        adapter = _FieldLengthAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", "pass"),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "output_limit_exceeded"
        assert "8 characters" in result.failures[0].message

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
        if os.name == "posix":
            monkeypatch.setattr(
                os,
                "killpg",
                lambda process_group_id, signal_number: process.kill(),
            )
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
        assert process._transport.closed is True

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_checker_executor_reports_process_group_cleanup_failure(
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

        def reject_group_kill(
            process_group_id: int,
            signal_number: int,
        ) -> None:
            _ = process_group_id, signal_number
            raise PermissionError("group kill denied")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        monkeypatch.setattr(os, "killpg", reject_group_kill)
        executor = CheckerExecutor(CheckerExecutionPolicy())
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )

        result = await executor.run(adapter, _request(tmp_path))

        assert len(result.failures) == 1
        assert result.failures[0].kind == "execution_error"
        assert result.failures[0].message == (
            "pipe failed; process cleanup did not complete within 1 seconds"
        )
        assert process.killed is True
        assert process.waited is True
        assert process._transport.closed is True

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
        if os.name == "posix":
            monkeypatch.setattr(
                os,
                "killpg",
                lambda process_group_id, signal_number: process.kill(),
            )
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


def test_checker_executor_bounds_stalled_pipe_cleanup(
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
        if os.name == "posix":
            monkeypatch.setattr(
                os,
                "killpg",
                lambda process_group_id, signal_number: process.kill(),
            )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=0.01,
                cleanup_timeout_seconds=0.01,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )
        started = asyncio.get_running_loop().time()

        result = await executor.run(adapter, _request(tmp_path))

        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.5
        assert len(result.failures) == 1
        assert result.failures[0].kind == "timeout"
        assert (
            "process cleanup did not complete within 0.01 seconds"
            in result.failures[0].message
        )
        assert process.reaped.is_set()
        assert process._transport.closed is True

    asyncio.run(scenario())


def test_checker_executor_logs_incomplete_cleanup_during_cancellation(
    caplog,
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
        if os.name == "posix":
            monkeypatch.setattr(
                os,
                "killpg",
                lambda process_group_id, signal_number: process.kill(),
            )
        executor = CheckerExecutor(CheckerExecutionPolicy(cleanup_timeout_seconds=0.01))
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, "-c", ""),
        )
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        await process.communicate_started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "stub process cleanup did not complete within 0.01 seconds" in (
            caplog.text
        )
        assert process._transport.closed is True

    asyncio.run(scenario())


def test_checker_executor_kills_and_reaps_timed_out_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=_CHECKER_STARTUP_BOUND_SECONDS
                + _CHECKER_TIMEOUT_MARGIN_SECONDS,
                max_concurrency=1,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(sys.executable, str(checker_script), str(pid_path)),
        )

        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        pid = await _wait_for_pid(pid_path)
        result = await task

        assert result.diagnostics == ()
        assert len(result.failures) == 1
        assert result.failures[0].kind == "timeout"
        _assert_process_reaped(pid)

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
@pytest.mark.parametrize(
    "parent_sleep_seconds",
    [0.0, 5.0],
    ids=["parent-exited", "parent-running"],
)
def test_checker_executor_timeout_terminates_descendants_inheriting_pipes(
    tmp_path: Path,
    parent_sleep_seconds: float,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "checker_with_descendant.py"
        child_pid_path = tmp_path / "checker_child.pid"
        _write_checker_with_descendant(
            checker_script,
            parent_sleep_seconds=parent_sleep_seconds,
            child_sleep_seconds=60.0,
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=_CHECKER_STARTUP_BOUND_SECONDS
                + _CHECKER_TIMEOUT_MARGIN_SECONDS,
                cleanup_timeout_seconds=0.5,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(
                sys.executable,
                str(checker_script),
                str(child_pid_path),
            ),
        )

        task = asyncio.create_task(
            asyncio.wait_for(
                executor.run(adapter, _request(tmp_path)),
                timeout=_CHECKER_STARTUP_BOUND_SECONDS
                + _CHECKER_TIMEOUT_MARGIN_SECONDS
                + 5,
            )
        )
        child_pid = await _wait_for_pid(child_pid_path)
        result = await task

        assert len(result.failures) == 1
        assert result.failures[0].kind == "timeout"
        await _assert_process_exited(child_pid)

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.name != "posix" or not Path("/dev/fd").is_dir(),
    reason="POSIX process sessions and file descriptor inspection are required",
)
def test_checker_executor_closes_pipes_held_by_detached_descendants(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        original_create_subprocess_exec = asyncio.create_subprocess_exec
        spawned_processes: list[asyncio.subprocess.Process] = []
        detached_child_pids: list[int] = []

        async def capture_process(
            *command: str,
            cwd: Path,
            stdout: int,
            stderr: int,
            start_new_session: bool,
        ) -> asyncio.subprocess.Process:
            process = await original_create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=start_new_session,
            )
            spawned_processes.append(process)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)
        checker_script = tmp_path / "checker_with_detached_descendant.py"
        _write_checker_with_descendant(
            checker_script,
            detach_child=True,
            parent_sleep_seconds=5.0,
            child_sleep_seconds=60.0,
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=_CHECKER_STARTUP_BOUND_SECONDS
                + _CHECKER_TIMEOUT_MARGIN_SECONDS,
                cleanup_timeout_seconds=0.1,
            )
        )
        initial_fd_count = len(os.listdir("/dev/fd"))
        fd_counts: list[int] = []

        try:
            for index in range(3):
                child_pid_path = tmp_path / f"checker_child_{index}.pid"
                adapter = _CommandAdapter(
                    executable=sys.executable,
                    command=(
                        sys.executable,
                        str(checker_script),
                        str(child_pid_path),
                    ),
                )

                task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
                child_pid = await _wait_for_pid(child_pid_path)
                detached_child_pids.append(child_pid)
                result = await task
                process = spawned_processes[index]
                transport = getattr(process, "_transport", None)

                assert len(result.failures) == 1
                assert result.failures[0].kind == "timeout"
                os.kill(child_pid, 0)
                assert isinstance(transport, asyncio.SubprocessTransport)
                stdout_transport = transport.get_pipe_transport(1)
                stderr_transport = transport.get_pipe_transport(2)
                assert stdout_transport is not None
                assert stderr_transport is not None
                assert stdout_transport.is_closing()
                assert stderr_transport.is_closing()

                os.kill(child_pid, signal.SIGKILL)
                await _assert_process_exited(child_pid)
                await asyncio.sleep(0)
                fd_counts.append(len(os.listdir("/dev/fd")))
        finally:
            candidate_paths = [
                *tmp_path.glob("checker_child_*.pid"),
                *tmp_path.glob("checker_child_*.tmp"),
            ]
            for pid_path in candidate_paths:
                try:
                    candidate = int(pid_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if candidate in detached_child_pids:
                    continue
                detached_child_pids.append(candidate)
            for child_pid in detached_child_pids:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        assert len(set(fd_counts)) == 1
        assert fd_counts[-1] <= initial_fd_count + 1

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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_checker_executor_cancellation_terminates_descendants_inheriting_pipes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "checker_with_descendant.py"
        child_pid_path = tmp_path / "checker_child.pid"
        _write_checker_with_descendant(
            checker_script,
            parent_sleep_seconds=5.0,
        )
        executor = CheckerExecutor(
            CheckerExecutionPolicy(
                timeout_seconds=30.0,
                cleanup_timeout_seconds=0.5,
            )
        )
        adapter = _CommandAdapter(
            executable=sys.executable,
            command=(
                sys.executable,
                str(checker_script),
                str(child_pid_path),
            ),
        )
        task = asyncio.create_task(executor.run(adapter, _request(tmp_path)))
        child_pid = await _wait_for_pid(child_pid_path)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

        await _assert_process_exited(child_pid)

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


def test_checker_executor_preserves_sibling_results_when_one_adapter_crashes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        sibling_script = tmp_path / "sibling_checker.py"
        sibling_script.write_text("print('ok')\n", encoding="utf-8")
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=2)
        )
        adapters = (
            _DiagnosticOutputAdapter(
                executable=sys.executable,
                command=(sys.executable, str(sibling_script)),
            ),
            _FailingOutputAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", "pass"),
            ),
        )

        result = await executor.run_all(adapters, _request(tmp_path))

        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].message == "sibling diagnostic"
        assert len(result.failures) == 1
        assert result.failures[0].tool == "stub"
        assert result.failures[0].kind == "execution_error"
        assert "adapter parse failed" in result.failures[0].message

    asyncio.run(scenario())


def test_checker_executor_normalizes_build_command_crash_per_tool(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=2)
        )
        adapters = (
            _DiagnosticOutputAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", "pass"),
            ),
            _FailingBuildCommandAdapter(
                executable=sys.executable,
                command=(),
            ),
        )

        result = await executor.run_all(adapters, _request(tmp_path))

        assert len(result.diagnostics) == 1
        assert len(result.failures) == 1
        assert result.failures[0].kind == "execution_error"
        assert "adapter build failed" in result.failures[0].message

    asyncio.run(scenario())


def test_checker_executor_preserves_all_tool_failures_when_every_adapter_crashes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = CheckerExecutor(
            CheckerExecutionPolicy(timeout_seconds=30.0, max_concurrency=2)
        )
        adapters = (
            _FailingOutputAdapter(
                executable=sys.executable,
                command=(sys.executable, "-c", "pass"),
            ),
            _FailingBuildCommandAdapter(
                executable=sys.executable,
                command=(),
            ),
        )

        result = await executor.run_all(adapters, _request(tmp_path))

        assert result.diagnostics == ()
        assert len(result.failures) == 2
        assert tuple(failure.kind for failure in result.failures) == (
            "execution_error",
            "execution_error",
        )
        assert tuple(failure.tool for failure in result.failures) == ("stub", "stub")

    asyncio.run(scenario())


def test_checker_executor_fails_fast_on_whole_request_cancellation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        checker_script = tmp_path / "blocking_checker.py"
        pid_path = tmp_path / "checker.pid"
        _write_blocking_checker(checker_script)
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
                command=(sys.executable, "-c", "pass"),
            ),
        )

        task = asyncio.create_task(executor.run_all(adapters, _request(tmp_path)))
        await _wait_for_pid(pid_path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pid = await _wait_for_pid(pid_path)
        _assert_process_reaped(pid)

    asyncio.run(scenario())
