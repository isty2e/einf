import asyncio
import importlib.util
import subprocess
import sys
import venv
from collections.abc import Callable
from pathlib import Path

import pytest

from einf.analysis.lsp.analysis_process import (
    AnalysisProcessCleanupError,
    LspAnalysisProcess,
)
from einf.analysis.lsp.analysis_queue import DocumentAnalysisRequest, LspAnalysisQueue
from einf.analysis.lsp.service import LspService

_SOURCE = (
    'from einf import ax, axes, reduce\nb, n, z = axes("b", "n", "z")\n'
    "reduce(ax[b, n], ax[b, z])\n"
)


def _request(version: int = 1) -> DocumentAnalysisRequest:
    return DocumentAnalysisRequest(
        uri="untitled:process-test.py", source=_SOURCE, version=version
    )


async def _wait_for_file(path: Path) -> None:
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.001)


def _record_spawns(
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_command: tuple[str, ...] | None = None,
    before_return: Callable[[], asyncio.Future[None]] | None = None,
) -> list[asyncio.SubprocessTransport]:
    loop = asyncio.get_running_loop()
    original = loop.subprocess_exec
    transports: list[asyncio.SubprocessTransport] = []

    async def spawn(protocol_factory, *command, **kwargs):
        first = not transports
        if first and first_command is not None:
            command = first_command
        transport, protocol = await original(protocol_factory, *command, **kwargs)
        transports.append(transport)
        if first and before_return is not None:
            await before_return()
        return transport, protocol

    monkeypatch.setattr(loop, "subprocess_exec", spawn)
    return transports


def _assert_reaped(transport: asyncio.SubprocessTransport) -> None:
    assert transport.get_returncode() is not None
    assert transport.is_closing()
    for descriptor in (0, 1):
        pipe = transport.get_pipe_transport(descriptor)
        assert pipe is not None
        assert pipe.is_closing()


@pytest.mark.parametrize("parser", ["ast", "libcst"])
def test_process_matches_service_and_reuses_child(monkeypatch, parser: str) -> None:
    if parser == "libcst":
        pytest.importorskip("libcst")

    async def scenario() -> None:
        transports = _record_spawns(monkeypatch)
        process = LspAnalysisProcess()
        service = LspService(parser)
        try:
            for version in (1, 2):
                request = _request(version)
                state = await asyncio.wait_for(
                    process.analyze(request, parser=parser), timeout=10
                )
                expected = service.analyze_document(
                    uri=request.uri, source=request.source, version=request.version
                )
                assert state == expected
                assert state.semantic_report.diagnostics
                assert state.semantic_report.axis_tokens
            assert len(transports) == 1
            assert transports[0].get_returncode() is None
        finally:
            await process.close()
        _assert_reaped(transports[0])

    asyncio.run(scenario())


@pytest.mark.parametrize("module_name", ["pickle", "logging", "typing", "einf"])
def test_worker_does_not_import_implicit_workspace_modules(
    monkeypatch, tmp_path: Path, module_name: str
) -> None:
    marker = tmp_path / "workspace-module-executed"
    (tmp_path / f"{module_name}.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "raise RuntimeError('workspace module must not run')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)

    async def scenario() -> None:
        process = LspAnalysisProcess()
        try:
            state = await asyncio.wait_for(
                process.analyze(_request(), parser="ast"), 10
            )
            assert state.semantic_report.diagnostics
            assert not marker.exists()
        finally:
            await process.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "startup_output",
    [
        "",
        "print('startup Python output', flush=True)\n",
        "import os; os.write(1, b'startup descriptor output\\n')\n",
    ],
)
def test_worker_preserves_explicit_pythonpath_configuration(
    monkeypatch, tmp_path: Path, capfd, startup_output: str
):
    configured_path = tmp_path / "configured-modules"
    configured_path.mkdir()
    marker = tmp_path / "configured-path-loaded"
    (configured_path / "sitecustomize.py").write_text(
        f"import sys\nopen({str(marker)!r}, 'w').write(sys.prefix)\n" + startup_output,
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(configured_path))

    async def scenario() -> None:
        process = LspAnalysisProcess()
        try:
            state = await asyncio.wait_for(
                process.analyze(_request(), parser="ast"), 10
            )
            assert state.semantic_report.diagnostics
            assert marker.read_text(encoding="utf-8") == sys.prefix
        finally:
            await process.close()

    asyncio.run(scenario())
    captured = capfd.readouterr()
    assert captured.out == ""
    if startup_output:
        assert "startup " in captured.err


def test_worker_separates_pth_startup_output(monkeypatch, tmp_path: Path, capfd):
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    executable = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    site_packages = Path(
        subprocess.run(
            [
                str(executable),
                "-P",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    )
    marker = tmp_path / "pth-loaded"
    import_paths = [path for path in sys.path if Path(path).is_absolute()]
    (site_packages / "analysis_startup.pth").write_text(
        "\n".join(import_paths)
        + "\nimport os; print('pth Python output', flush=True); "
        "os.write(1, b'pth descriptor output\\n'); "
        f"open({str(marker)!r}, 'w').write('loaded')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "executable", str(executable))

    async def scenario() -> None:
        transports = _record_spawns(monkeypatch)
        process = LspAnalysisProcess()
        try:
            for version in (1, 2):
                state = await asyncio.wait_for(
                    process.analyze(_request(version), parser="ast"), 10
                )
                assert state.semantic_report.diagnostics
            assert marker.read_text(encoding="utf-8") == "loaded"
            assert len(transports) == 1
        finally:
            await process.close()
        _assert_reaped(transports[0])

    asyncio.run(scenario())
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "pth Python output" in captured.err
    assert "pth descriptor output" in captured.err


def test_worker_cancellation_reaps_blocked_startup_hook(monkeypatch, tmp_path: Path):
    ready = tmp_path / "startup-ready"
    (tmp_path / "sitecustomize.py").write_text(
        f"import time\nopen({str(ready)!r}, 'w').close()\ntime.sleep(600)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    async def scenario() -> None:
        transports = _record_spawns(monkeypatch)
        process = LspAnalysisProcess()
        execution = asyncio.create_task(process.analyze(_request(), parser="ast"))
        try:
            await _wait_for_file(ready)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(execution, 10)
            _assert_reaped(transports[0])

            monkeypatch.delenv("PYTHONPATH")
            state = await asyncio.wait_for(
                process.analyze(_request(2), parser="ast"), 10
            )
            assert state.semantic_report.diagnostics
            assert len(transports) == 2
        finally:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            await process.close()
        _assert_reaped(transports[1])

    asyncio.run(scenario())


def test_queue_reuses_healthy_process_without_default_executor(monkeypatch) -> None:
    async def scenario() -> None:
        transports = _record_spawns(monkeypatch)

        def reject_executor(*args, **kwargs):
            pytest.fail("semantic analysis must not use the default thread executor")

        monkeypatch.setattr(
            asyncio.get_running_loop(), "run_in_executor", reject_executor
        )
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)
        committed: list[int | None] = []
        try:
            for version in (1, 2, 3):
                state = await asyncio.wait_for(
                    queue.analyze(
                        _request(version),
                        parser="ast",
                        commit=lambda state: committed.append(state.version),
                    ),
                    10,
                )
                assert state is not None and state.version == version
            assert len(transports) == 1
            assert committed == [1, 2, 3]
        finally:
            await queue.close()
        _assert_reaped(transports[0])

    asyncio.run(scenario())


def test_event_loop_shutdown_reaps_active_analysis(monkeypatch, tmp_path: Path) -> None:
    transports: list[asyncio.SubprocessTransport] = []

    async def scenario() -> None:
        ready = tmp_path / "ready"
        recorded = _record_spawns(
            monkeypatch,
            first_command=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import time; "
                    f"Path({str(ready)!r}).touch(); time.sleep(5)"
                ),
            ),
        )
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)
        asyncio.create_task(
            queue.analyze(_request(), parser="ast", commit=lambda state: None)
        )
        await _wait_for_file(ready)
        transports.extend(recorded)

    asyncio.run(scenario())
    assert len(transports) == 1
    _assert_reaped(transports[0])


@pytest.mark.parametrize("action", ["replace", "cancel", "close"])
def test_queue_terminates_blocked_child_and_closes_pipes(
    monkeypatch, tmp_path: Path, action: str
) -> None:
    async def scenario() -> None:
        ready = tmp_path / "ready"
        command = (
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import time; "
                f"Path({str(ready)!r}).touch(); time.sleep(600)"
            ),
        )
        transports = _record_spawns(monkeypatch, first_command=command)
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)
        committed: list[int | None] = []

        def submit(version: int):
            return asyncio.create_task(
                queue.analyze(
                    _request(version),
                    parser="ast",
                    commit=lambda state: committed.append(state.version),
                )
            )

        old = submit(1)
        latest = None
        try:
            await _wait_for_file(ready)
            if action == "replace":
                latest = submit(2)
            elif action == "cancel":
                await queue.cancel(uri=_request().uri)
                latest = submit(2)
            else:
                await asyncio.wait_for(queue.close(), timeout=10)

            assert await asyncio.wait_for(old, timeout=10) is None
            if latest is not None:
                state = await asyncio.wait_for(latest, timeout=10)
                assert state is not None and state.version == 2
                assert state.semantic_report.diagnostics
                assert committed == [2]
                assert len(transports) == 2
            else:
                assert committed == []
                assert len(transports) == 1
            _assert_reaped(transports[0])
        finally:
            await queue.close()
            await asyncio.gather(
                old, *([latest] if latest is not None else []), return_exceptions=True
            )
        for transport in transports:
            _assert_reaped(transport)

    asyncio.run(scenario())


def test_cancel_during_spawn_retains_process_ownership(monkeypatch) -> None:
    async def scenario() -> None:
        release_handoff: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        transports = _record_spawns(monkeypatch, before_return=lambda: release_handoff)
        process = LspAnalysisProcess()
        execution = asyncio.create_task(process.analyze(_request(), parser="ast"))
        try:
            async with asyncio.timeout(10):
                while not transports:
                    await asyncio.sleep(0)
            execution.cancel()
            await asyncio.sleep(0)
            assert not release_handoff.cancelled()
            release_handoff.set_result(None)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(execution, timeout=10)
        finally:
            if not release_handoff.done():
                release_handoff.set_result(None)
            await process.close()
            await asyncio.gather(execution, return_exceptions=True)
        assert len(transports) == 1
        _assert_reaped(transports[0])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "script",
    ["raise SystemExit(7)", "import sys; print('invalid worker output', flush=True)"],
)
def test_worker_crash_or_invalid_response_fails_then_recovers(monkeypatch, script: str):
    async def scenario() -> None:
        transports = _record_spawns(
            monkeypatch, first_command=(sys.executable, "-c", script)
        )
        process = LspAnalysisProcess()
        try:
            with pytest.raises(RuntimeError, match="semantic worker"):
                await asyncio.wait_for(process.analyze(_request(), parser="ast"), 10)
            _assert_reaped(transports[0])
            state = await asyncio.wait_for(
                process.analyze(_request(2), parser="ast"), 10
            )
            assert state.version == 2
            assert state.semantic_report.diagnostics
            assert len(transports) == 2
        finally:
            await process.close()
        for transport in transports:
            _assert_reaped(transport)

    asyncio.run(scenario())


def test_process_restarts_for_parser_change(monkeypatch) -> None:
    pytest.importorskip("libcst")

    async def scenario() -> None:
        transports = _record_spawns(monkeypatch)
        process = LspAnalysisProcess()
        try:
            for parser in ("ast", "libcst"):
                state = await asyncio.wait_for(
                    process.analyze(_request(), parser=parser), 10
                )
                assert state.semantic_report.diagnostics
            assert len(transports) == 2
            _assert_reaped(transports[0])
        finally:
            await process.close()
        _assert_reaped(transports[1])

    asyncio.run(scenario())


def test_analyzer_exception_is_a_failure_not_an_empty_report(monkeypatch) -> None:
    script = """
import sys
from einf.analysis.lsp.service import LspService
from einf.analysis.lsp.analysis_process import main

def fail_analysis(self, **kwargs):
    raise ValueError("controlled analyzer failure")

LspService.analyze_document = fail_analysis
sys.argv = ["analysis-worker", "ast"]
main(sys.stdout.buffer)
"""

    async def scenario() -> None:
        transports = _record_spawns(
            monkeypatch, first_command=(sys.executable, "-c", script)
        )
        process = LspAnalysisProcess()
        try:
            with pytest.raises(
                RuntimeError, match="ValueError: controlled analyzer failure"
            ):
                await asyncio.wait_for(process.analyze(_request(), parser="ast"), 10)
            _assert_reaped(transports[0])
            state = await asyncio.wait_for(
                process.analyze(_request(2), parser="ast"), 10
            )
            assert state.semantic_report.diagnostics
            assert len(transports) == 2
        finally:
            await process.close()
        _assert_reaped(transports[1])

    asyncio.run(scenario())


def test_worker_routes_incidental_prints_to_stderr(monkeypatch, capfd) -> None:
    script = """
import sys
from einf.analysis.lsp.service import LspService
from einf.analysis.lsp.analysis_process import main

original = LspService.analyze_document
original_init = LspService.__init__

def noisy_init(self, *args, **kwargs):
    print("incidental parser initialization", flush=True)
    original_init(self, *args, **kwargs)

def noisy_analysis(self, **kwargs):
    print("incidental semantic message", flush=True)
    return original(self, **kwargs)

LspService.analyze_document = noisy_analysis
LspService.__init__ = noisy_init
sys.argv = ["analysis-worker", "ast"]
main(sys.stdout.buffer)
"""

    async def scenario() -> None:
        _record_spawns(monkeypatch, first_command=(sys.executable, "-c", script))
        process = LspAnalysisProcess()
        try:
            state = await asyncio.wait_for(
                process.analyze(_request(), parser="ast"), 10
            )
            assert state.semantic_report.diagnostics
        finally:
            await process.close()

    asyncio.run(scenario())
    captured = capfd.readouterr()
    assert "incidental parser initialization" in captured.err
    assert "incidental semantic message" in captured.err
    assert captured.out == ""


@pytest.mark.skipif(
    importlib.util.find_spec("libcst") is not None,
    reason="requires an environment without the optional parser",
)
def test_missing_optional_parser_returns_failure_report() -> None:
    async def scenario() -> None:
        process = LspAnalysisProcess()
        try:
            state = await asyncio.wait_for(
                process.analyze(_request(), parser="libcst"), 10
            )
            assert state.semantic_report.diagnostics == ()
            assert len(state.semantic_report.failures) == 1
            assert state.semantic_report.failures[0].kind == "parser_unavailable"
        finally:
            await process.close()

    asyncio.run(scenario())


def test_cleanup_timeout_retains_inflight_spawn_and_prevents_replacement(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            "einf.analysis.lsp.analysis_process._CLEANUP_TIMEOUT_SECONDS", 0.01
        )
        handoff: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        transports = _record_spawns(monkeypatch, before_return=lambda: handoff)
        process = LspAnalysisProcess()
        execution = asyncio.create_task(process.analyze(_request(), parser="ast"))
        try:
            async with asyncio.timeout(10):
                while not transports:
                    await asyncio.sleep(0)
            execution.cancel()
            with pytest.raises(
                AnalysisProcessCleanupError, match="cleanup did not complete"
            ):
                await asyncio.wait_for(execution, 10)
            assert not handoff.cancelled()
            with pytest.raises(
                AnalysisProcessCleanupError, match="cleanup did not complete"
            ):
                await process.analyze(_request(2), parser="ast")
            assert len(transports) == 1
        finally:
            handoff.set_result(None)
            retirement = process._retirement
            assert retirement is not None
            await asyncio.wait_for(asyncio.shield(retirement), 10)
            await process.close()
            await asyncio.gather(execution, return_exceptions=True)
        _assert_reaped(transports[0])

    asyncio.run(scenario())
