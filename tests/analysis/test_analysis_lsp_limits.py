import asyncio
import tracemalloc
from dataclasses import replace
from typing import cast
from weakref import ref

import pytest

from einf.analysis.lsp.analysis_limits import (
    LspAnalysisLimitExceeded,
    LspAnalysisLimits,
    LspSourceBudget,
)
from einf.analysis.lsp.analysis_process import (
    AnalysisProcessCleanupError,
    LspAnalysisProcess,
)
from einf.analysis.lsp.analysis_queue import DocumentAnalysisRequest, LspAnalysisQueue
from einf.analysis.lsp.change_debounce import LspChangeDebouncer, PendingDocumentChange
from einf.analysis.lsp.config import LspConfig
from einf.analysis.lsp.service import LspDocumentState
from einf.analysis.report import AnalysisFileReport


@pytest.mark.parametrize(
    "name", ["max_documents", "max_document_bytes", "max_source_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_limits_reject_invalid_explicit_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        replace(LspAnalysisLimits(), **{name: cast(int, value)})


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("analysisMaxDocuments", LspAnalysisLimits(max_documents=3)),
        ("analysisMaxDocumentBytes", LspAnalysisLimits(max_document_bytes=3)),
        ("analysisMaxSourceBytes", LspAnalysisLimits(max_source_bytes=3)),
    ],
)
def test_initialize_analysis_limits_are_independent(
    option: str, expected: LspAnalysisLimits
) -> None:
    assert LspConfig.from_initialize_options({option: 3}).analysis_limits == expected
    assert (
        LspConfig.from_initialize_options({option: None}).analysis_limits
        == LspAnalysisLimits()
    )
    with pytest.raises(ValueError, match=option):
        LspConfig.from_initialize_options({option: "invalid"})


def test_unlimited_and_count_only_budgets_do_not_scan_source() -> None:
    class NoByteAccess(str):
        def isascii(self) -> bool:
            pytest.fail("count-only admission must not inspect source bytes")

    assert not LspAnalysisLimits().enabled
    for limits in (LspAnalysisLimits(), LspAnalysisLimits(max_documents=1)):
        reservation = LspSourceBudget(limits).reserve(
            uri="untitled:one", source=NoByteAccess("x")
        )
        reservation.release()


def test_zero_byte_documents_still_consume_document_capacity() -> None:
    budget = LspSourceBudget(LspAnalysisLimits(max_documents=1))
    first = budget.reserve(uri="untitled:one", source="")
    shared_uri = budget.reserve(uri="untitled:one", source="")
    with pytest.raises(LspAnalysisLimitExceeded) as caught:
        budget.reserve(uri="untitled:two", source="")
    assert (caught.value.resource, caught.value.limit, caught.value.observed) == (
        "documents",
        1,
        2,
    )
    first.release()
    with pytest.raises(LspAnalysisLimitExceeded, match="documents"):
        budget.reserve(uri="untitled:two", source="")
    shared_uri.release()
    budget.reserve(uri="untitled:two", source="").release()


def test_utf8_document_boundary_and_rejection_are_atomic() -> None:
    budget = LspSourceBudget(
        LspAnalysisLimits(max_document_bytes=6, max_source_bytes=6)
    )
    with pytest.raises(LspAnalysisLimitExceeded) as caught:
        budget.reserve(uri="untitled:one", source="한🙂")
    assert (caught.value.resource, caught.value.limit, caught.value.observed) == (
        "document_bytes",
        6,
        7,
    )
    at_limit = budget.reserve(uri="untitled:one", source="é🙂")
    at_limit.release()
    with pytest.raises(UnicodeEncodeError):
        budget.reserve(uri="untitled:one", source="\ud800")
    budget.reserve(uri="untitled:one", source="é🙂").release()


def test_shared_holders_charge_once_but_overlapping_captures_charge_separately() -> (
    None
):
    budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=6))
    first = budget.reserve(uri="untitled:one", source="abc")
    first.retain()
    second = budget.reserve(uri="untitled:one", source="def")
    with pytest.raises(LspAnalysisLimitExceeded) as caught:
        budget.reserve(uri="untitled:two", source="g")
    assert (caught.value.resource, caught.value.limit, caught.value.observed) == (
        "source_bytes",
        6,
        7,
    )
    first.release()
    with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
        budget.reserve(uri="untitled:two", source="g")
    first.release()
    with pytest.raises(RuntimeError, match="already been released"):
        first.retain()
    with pytest.raises(RuntimeError, match="already been released"):
        first.release()
    budget.reserve(uri="untitled:two", source="ghi").release()
    second.release()
    budget.reserve(uri="untitled:two", source="abcdef").release()


def test_reservation_does_not_retain_source_text() -> None:
    class Source(str):
        pass

    text = Source("not retained")
    reference = ref(text)
    reservation = LspSourceBudget(LspAnalysisLimits(max_source_bytes=20)).reserve(
        uri="untitled:one", source=text
    )
    del text
    assert reference() is None
    reservation.release()


def test_utf8_counting_does_not_allocate_a_full_encoded_source() -> None:
    text = "é" * 1_000_000
    budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=2_000_000))
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        reservation = budget.reserve(uri="untitled:one", source=text)
        _, peak = tracemalloc.get_traced_memory()
        assert peak - before < 512 * 1024
        reservation.release()
    finally:
        if not already_tracing:
            tracemalloc.stop()


def _state(request: DocumentAnalysisRequest) -> LspDocumentState:
    return LspDocumentState(
        uri=request.uri,
        path=None,
        version=request.version,
        source=request.source,
        semantic_report=AnalysisFileReport(
            path="<in-memory>",
            diagnostics=(),
            checker_diagnostics=(),
            axis_tokens=(),
            failures=(),
        ),
    )


def test_debounce_cancel_and_flush_keep_ownership_until_retirement() -> None:
    async def scenario() -> None:
        budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=3))
        debouncer = LspChangeDebouncer(
            delay_seconds=60,
            report_failure=lambda change, error: pytest.fail(str(error)),
        )

        async def analyze(change: PendingDocumentChange) -> None:
            pytest.fail("cancelled delayed work must not analyze")

        for flush in (False, True):
            reservation = budget.reserve(uri="untitled:one", source="abc")
            debouncer.schedule(
                PendingDocumentChange("untitled:one", "abc", 1, reservation),
                analyze=analyze,
            )
            reservation.release()
            if flush:
                pending = debouncer.take_pending(uri="untitled:one")
                assert pending is not None and pending.source_reservation is not None
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
                    budget.reserve(uri="untitled:two", source="x")
                pending.source_reservation.release()
            else:
                await debouncer.cancel_and_wait(uri="untitled:one")
            budget.reserve(uri="untitled:two", source="abc").release()
        await debouncer.close()

    asyncio.run(scenario())


def test_debounce_failed_task_creation_releases_its_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=3))
        debouncer = LspChangeDebouncer(
            delay_seconds=60, report_failure=lambda change, error: None
        )
        reservation = budget.reserve(uri="untitled:one", source="abc")

        async def analyze(change: PendingDocumentChange) -> None:
            pytest.fail("failed task must not execute")

        def fail_creation(*args: object, **kwargs: object) -> None:
            raise RuntimeError("task factory failed")

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "create_task", fail_creation)
            with pytest.raises(RuntimeError, match="task factory failed"):
                debouncer.schedule(
                    PendingDocumentChange("untitled:one", "abc", 1, reservation),
                    analyze=analyze,
                )
        reservation.release()
        assert not debouncer.has_pending(uri="untitled:one")
        budget.reserve(uri="untitled:two", source="abc").release()
        await debouncer.close()

    asyncio.run(scenario())


def test_debounce_close_waits_for_previously_cancelled_task() -> None:
    async def scenario() -> None:
        budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=3))
        debouncer = LspChangeDebouncer(
            delay_seconds=0,
            report_failure=lambda change, error: pytest.fail(str(error)),
        )
        started, retiring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def analyze(change: PendingDocumentChange) -> None:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                retiring.set()
                # Cleanup must survive a second cancellation from close().
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass

        reservation = budget.reserve(uri="untitled:one", source="abc")
        debouncer.schedule(
            PendingDocumentChange("untitled:one", "abc", 1, reservation),
            analyze=analyze,
        )
        reservation.release()
        await asyncio.wait_for(started.wait(), timeout=2)
        debouncer.cancel("untitled:one")
        await asyncio.wait_for(retiring.wait(), timeout=2)
        closing = asyncio.create_task(debouncer.close())
        try:
            await asyncio.sleep(0)
            assert not closing.done()
            with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
                budget.reserve(uri="untitled:two", source="x")
        finally:
            release.set()
            await closing
        budget.reserve(uri="untitled:two", source="abc").release()

    asyncio.run(scenario())


def test_queue_waiters_and_retiring_execution_remain_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        retiring = asyncio.Event()
        release = asyncio.Event()
        budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=9))
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def execute(
            self: LspAnalysisProcess, request: DocumentAnalysisRequest, *, parser: str
        ) -> LspDocumentState:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                retiring.set()
                await release.wait()
                raise
            return _state(request)

        async def close(self: LspAnalysisProcess) -> None:
            pass

        monkeypatch.setattr(LspAnalysisProcess, "analyze", execute)
        monkeypatch.setattr(LspAnalysisProcess, "close", close)
        tasks = []
        for index in range(3):
            uri = f"untitled:{index}"
            reservation = budget.reserve(uri=uri, source="abc")
            task = asyncio.create_task(
                queue.analyze(
                    DocumentAnalysisRequest(uri, "abc", 1),
                    parser="ast",
                    commit=lambda state: None,
                    source_reservation=reservation,
                )
            )
            tasks.append(task)
            await asyncio.sleep(0)
            reservation.release()
            if index == 0:
                await asyncio.wait_for(started.wait(), timeout=2)
        try:
            with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
                budget.reserve(uri="untitled:extra", source="x")
            tasks[2].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[2]
            await queue.cancel(uri="untitled:1")
            assert await tasks[1] is None
            await queue.cancel(uri="untitled:0")
            assert await tasks[0] is None
            await asyncio.wait_for(retiring.wait(), timeout=2)
            blocker = budget.reserve(uri="untitled:extra", source="abcdef")
            with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
                budget.reserve(uri="untitled:more", source="x")
            release.set()
            await queue.close()
            budget.reserve(uri="untitled:more", source="abc").release()
            blocker.release()
        finally:
            release.set()
            await queue.close()

    asyncio.run(scenario())


def test_cleanup_failure_does_not_return_running_capacity_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        budget = LspSourceBudget(LspAnalysisLimits(max_source_bytes=3))
        queue = LspAnalysisQueue(worker_count=1, pending_limit=1)

        async def execute(
            self: LspAnalysisProcess, request: DocumentAnalysisRequest, *, parser: str
        ) -> LspDocumentState:
            raise AnalysisProcessCleanupError("controlled cleanup failure")

        async def close(self: LspAnalysisProcess) -> None:
            pass

        monkeypatch.setattr(LspAnalysisProcess, "analyze", execute)
        monkeypatch.setattr(LspAnalysisProcess, "close", close)
        reservation = budget.reserve(uri="untitled:one", source="abc")
        try:
            with pytest.raises(
                AnalysisProcessCleanupError, match="controlled cleanup failure"
            ):
                await queue.analyze(
                    DocumentAnalysisRequest("untitled:one", "abc", 1),
                    parser="ast",
                    commit=lambda state: None,
                    source_reservation=reservation,
                )
        finally:
            reservation.release()
        with pytest.raises(LspAnalysisLimitExceeded, match="source_bytes"):
            budget.reserve(uri="untitled:two", source="x")
        with pytest.raises(
            AnalysisProcessCleanupError, match="controlled cleanup failure"
        ):
            await queue.close()
        budget.reserve(uri="untitled:two", source="abc").release()

    asyncio.run(scenario())
