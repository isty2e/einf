import asyncio
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
from einf.analysis.lsp.checker_coordinator import (
    DocumentCheckerRequest,
    LspCheckerCoordinator,
)


class _NoopAdapter(CheckerAdapter):
    name = "stub"
    executable = sys.executable

    def build_command(
        self,
        request: CheckerRequest,
        /,
    ) -> tuple[str, ...]:
        _ = request
        return (sys.executable, "-c", "")

    def parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = stdout, stderr, request
        return CheckerResult(diagnostics=(), failures=())


class _ControlledExecutor(CheckerExecutor):
    def __init__(self, result: CheckerResult) -> None:
        super().__init__(CheckerExecutionPolicy())
        self.result = result
        self.blocked_started = asyncio.Event()
        self.release_blocked = asyncio.Event()
        self.blocked_cancelled = asyncio.Event()
        self.fail_next = False

    async def run_all(
        self,
        adapters: tuple[CheckerAdapter, ...],
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = adapters
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("checker executor failed")
        if request.targets[0].name == "blocked.py":
            self.blocked_started.set()
            try:
                await self.release_blocked.wait()
            except asyncio.CancelledError:
                self.blocked_cancelled.set()
                raise
        return self.result


class _SlowCancellingExecutor(_ControlledExecutor):
    def __init__(self, result: CheckerResult) -> None:
        super().__init__(result)
        self.finish_cancellation = asyncio.Event()

    async def run_all(
        self,
        adapters: tuple[CheckerAdapter, ...],
        request: CheckerRequest,
    ) -> CheckerResult:
        _ = adapters, request
        self.blocked_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("blocking checker unexpectedly completed")
        except asyncio.CancelledError:
            self.blocked_cancelled.set()
            await self.finish_cancellation.wait()
            raise


def _request(path: Path, *, version: int) -> DocumentCheckerRequest:
    return DocumentCheckerRequest(
        uri=path.resolve().as_uri(),
        path=path.resolve(),
        version=version,
    )


def _coordinator(executor: CheckerExecutor) -> LspCheckerCoordinator:
    return LspCheckerCoordinator(
        adapters=(_NoopAdapter(),),
        executor=executor,
    )


def test_checker_coordinator_replaces_same_uri_and_commits_latest(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "blocked.py"
        result = CheckerResult(diagnostics=(), failures=())
        executor = _ControlledExecutor(result)
        coordinator = _coordinator(executor)
        committed_versions: list[int | None] = []

        def commit(request: DocumentCheckerRequest, result: CheckerResult) -> bool:
            _ = result
            committed_versions.append(request.version)
            return True

        stale_task = asyncio.create_task(
            coordinator.check(_request(path, version=1), commit=commit)
        )
        await executor.blocked_started.wait()
        latest_path = tmp_path / "latest.py"
        latest_request = DocumentCheckerRequest(
            uri=path.resolve().as_uri(),
            path=latest_path.resolve(),
            version=2,
        )

        latest_result = await coordinator.check(latest_request, commit=commit)
        stale_result = await stale_task
        await coordinator.close()

        assert stale_result is None
        assert latest_result == result
        assert executor.blocked_cancelled.is_set()
        assert committed_versions == [2]

    asyncio.run(scenario())


def test_checker_coordinator_cancel_prevents_commit_and_waits_for_cleanup(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "blocked.py"
        executor = _ControlledExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        committed: list[CheckerResult] = []
        task = asyncio.create_task(
            coordinator.check(
                _request(path, version=1),
                commit=lambda request, result: (
                    committed.append(result) or request.version == 1
                ),
            )
        )
        await executor.blocked_started.wait()

        await coordinator.cancel(uri=path.resolve().as_uri())
        stale_result = await task
        await coordinator.close()

        assert stale_result is None
        assert executor.blocked_cancelled.is_set()
        assert committed == []

    asyncio.run(scenario())


def test_checker_coordinator_does_not_block_other_uri(tmp_path: Path) -> None:
    async def scenario() -> None:
        blocked_path = tmp_path / "blocked.py"
        ready_path = tmp_path / "ready.py"
        result = CheckerResult(diagnostics=(), failures=())
        executor = _ControlledExecutor(result)
        coordinator = _coordinator(executor)
        blocked = asyncio.create_task(
            coordinator.check(
                _request(blocked_path, version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        ready = await asyncio.wait_for(
            coordinator.check(
                _request(ready_path, version=1),
                commit=lambda request, result: True,
            ),
            timeout=0.1,
        )

        assert ready == result
        assert blocked.done() is False
        await coordinator.cancel(uri=blocked_path.resolve().as_uri())
        assert await blocked is None
        await coordinator.close()

    asyncio.run(scenario())


def test_checker_coordinator_generation_replaces_unknown_versions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        blocked_path = tmp_path / "blocked.py"
        latest_path = tmp_path / "latest.py"
        uri = blocked_path.resolve().as_uri()
        result = CheckerResult(diagnostics=(), failures=())
        executor = _ControlledExecutor(result)
        coordinator = _coordinator(executor)
        committed_paths: list[Path] = []
        stale = asyncio.create_task(
            coordinator.check(
                DocumentCheckerRequest(uri=uri, path=blocked_path, version=None),
                commit=lambda request, result: (
                    committed_paths.append(request.path) or True
                ),
            )
        )
        await executor.blocked_started.wait()

        latest = await coordinator.check(
            DocumentCheckerRequest(uri=uri, path=latest_path, version=None),
            commit=lambda request, result: committed_paths.append(request.path) or True,
        )

        assert await stale is None
        assert latest == result
        assert committed_paths == [latest_path]
        await coordinator.close()

    asyncio.run(scenario())


def test_checker_coordinator_close_resolves_and_cleans_active_job(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "blocked.py"
        executor = _ControlledExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        task = asyncio.create_task(
            coordinator.check(
                _request(path, version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        await coordinator.close()

        assert await task is None
        assert executor.blocked_cancelled.is_set()

    asyncio.run(scenario())


def test_checker_coordinator_recovers_after_failure_and_commit_rejection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "sample.py"
        result = CheckerResult(diagnostics=(), failures=())
        executor = _ControlledExecutor(result)
        executor.fail_next = True
        coordinator = _coordinator(executor)

        try:
            await coordinator.check(
                _request(path, version=1),
                commit=lambda request, result: True,
            )
        except RuntimeError as error:
            assert str(error) == "checker executor failed"
        else:
            raise AssertionError("checker failure was not propagated")

        rejected = await coordinator.check(
            _request(path, version=2),
            commit=lambda request, result: False,
        )
        accepted = await coordinator.check(
            _request(path, version=3),
            commit=lambda request, result: True,
        )
        await coordinator.close()

        assert rejected is None
        assert accepted == result

    asyncio.run(scenario())


def test_checker_coordinator_recovers_after_commit_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "sample.py"
        result = CheckerResult(diagnostics=(), failures=())
        coordinator = _coordinator(_ControlledExecutor(result))

        def fail_commit(
            request: DocumentCheckerRequest,
            result: CheckerResult,
        ) -> bool:
            _ = request, result
            raise RuntimeError("commit failed")

        with pytest.raises(RuntimeError, match="commit failed"):
            await coordinator.check(_request(path, version=1), commit=fail_commit)

        recovered = await coordinator.check(
            _request(path, version=2),
            commit=lambda request, result: True,
        )
        await coordinator.close()

        assert recovered == result

    asyncio.run(scenario())


def test_closed_disabled_checker_coordinator_rejects_new_work(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        coordinator = LspCheckerCoordinator(
            adapters=(),
            executor=CheckerExecutor(CheckerExecutionPolicy()),
        )
        await coordinator.close()

        with pytest.raises(RuntimeError, match="checker coordinator is closed"):
            await coordinator.check(
                _request(tmp_path / "sample.py", version=1),
                commit=lambda request, result: True,
            )

    asyncio.run(scenario())


def test_checker_coordinator_repeated_cancel_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        uri = (tmp_path / "sample.py").resolve().as_uri()
        coordinator = _coordinator(
            _ControlledExecutor(CheckerResult(diagnostics=(), failures=()))
        )

        await coordinator.cancel(uri=uri)
        await coordinator.cancel(uri=uri)
        await coordinator.close()

    asyncio.run(scenario())


def test_concurrent_close_callers_wait_for_active_job_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = _SlowCancellingExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        check_task = asyncio.create_task(
            coordinator.check(
                _request(tmp_path / "sample.py", version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        first_close = asyncio.create_task(coordinator.close())
        await executor.blocked_cancelled.wait()
        second_close = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0)

        assert second_close.done() is False
        executor.finish_cancellation.set()
        await asyncio.gather(first_close, second_close)
        assert await check_task is None

    asyncio.run(scenario())


def test_cancelling_check_caller_waits_for_active_job_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = _SlowCancellingExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        check_task = asyncio.create_task(
            coordinator.check(
                _request(tmp_path / "sample.py", version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        check_task.cancel()
        await executor.blocked_cancelled.wait()
        check_task.cancel()
        await asyncio.sleep(0)

        assert check_task.done() is False
        executor.finish_cancellation.set()
        with pytest.raises(asyncio.CancelledError):
            await check_task
        await coordinator.close()

    asyncio.run(scenario())


def test_cancelling_cancel_caller_waits_for_active_job_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = _SlowCancellingExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        check_task = asyncio.create_task(
            coordinator.check(
                _request(tmp_path / "sample.py", version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        cancel_task = asyncio.create_task(
            coordinator.cancel(uri=(tmp_path / "sample.py").resolve().as_uri())
        )
        await executor.blocked_cancelled.wait()
        cancel_task.cancel()
        await asyncio.sleep(0)

        assert cancel_task.done() is False
        executor.finish_cancellation.set()
        with pytest.raises(asyncio.CancelledError):
            await cancel_task
        assert await check_task is None
        await coordinator.close()

    asyncio.run(scenario())


def test_cancelling_close_caller_does_not_cancel_shared_cleanup(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executor = _SlowCancellingExecutor(CheckerResult(diagnostics=(), failures=()))
        coordinator = _coordinator(executor)
        check_task = asyncio.create_task(
            coordinator.check(
                _request(tmp_path / "sample.py", version=1),
                commit=lambda request, result: True,
            )
        )
        await executor.blocked_started.wait()

        cancelled_close = asyncio.create_task(coordinator.close())
        await executor.blocked_cancelled.wait()
        cancelled_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_close

        replacement_close = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0)
        assert replacement_close.done() is False

        executor.finish_cancellation.set()
        await replacement_close
        assert await check_task is None

    asyncio.run(scenario())
