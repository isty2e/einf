import gc
import inspect
import traceback
from weakref import ReferenceType, ref

import pytest

from benchmarks.shared.failures import clear_failed_case_frames


class Payload:
    pass


def test_case_failure_releases_positional_and_keyword_arguments() -> None:
    payloads: list[ReferenceType[Payload]] = []
    original = ValueError("original failure")
    original.add_note("keep this note")

    @clear_failed_case_frames
    def fail(value: Payload, *, option: Payload) -> None:
        payloads.extend((ref(value), ref(option)))
        raise original

    with pytest.raises(ValueError, match="original failure") as caught:
        fail(Payload(), option=Payload())

    gc.collect()
    assert all(payload() is None for payload in payloads)
    assert caught.value is original
    assert caught.value.__notes__ == ["keep this note"]
    source = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert source.name == "fail"
    assert source.line == "raise original"
    assert source.lineno is not None and source.lineno > 0


@pytest.mark.parametrize(
    "chain", ["explicit", "implicit", "suppressed", "cycle", "group"]
)
def test_case_failure_clears_related_exception_frames_without_replacing_errors(
    chain: str,
) -> None:
    payloads: list[ReferenceType[Payload]] = []
    originals: list[Exception] = []

    def inner() -> None:
        payload = Payload()
        payloads.append(ref(payload))
        raise ValueError("inner failure")

    @clear_failed_case_frames
    def fail() -> None:
        if chain in {"explicit", "group"}:
            try:
                inner()
            except ValueError as error:
                originals.append(error)
            # Leave the handler so only cause/member edges reach the inner error.
            if chain == "group":
                raise ExceptionGroup("group failure", [originals[0]])
            outer = RuntimeError("outer failure")
            originals.append(outer)
            raise outer from originals[0]

        try:
            inner()
        except ValueError as error:
            originals.append(error)
            outer = RuntimeError("outer failure")
            originals.append(outer)
            if chain == "suppressed":
                raise outer from None
            if chain == "cycle":
                error.__cause__ = outer
            raise outer

    message = "group failure" if chain == "group" else "outer failure"
    with pytest.raises(Exception, match=message) as caught:
        fail()

    gc.collect()
    assert len(payloads) == 1 and payloads[0]() is None
    if chain == "group":
        assert isinstance(caught.value, ExceptionGroup)
        assert caught.value.exceptions == (originals[0],)
    else:
        assert caught.value is originals[1]
        assert caught.value.__context__ is (
            None if chain == "explicit" else originals[0]
        )
    if chain == "explicit":
        assert caught.value.__cause__ is originals[0]
    if chain == "suppressed":
        assert caught.value.__suppress_context__
    if chain == "cycle":
        assert originals[0].__cause__ is caught.value
    assert traceback.extract_tb(originals[0].__traceback__)[-1].name == "inner"


def test_case_cleanup_preserves_caller_owned_references() -> None:
    payload = Payload()
    reference = ref(payload)

    @clear_failed_case_frames
    def fail(value: Payload) -> None:
        assert value is payload
        raise ValueError("failure")

    with pytest.raises(ValueError, match="failure"):
        fail(payload)

    assert reference() is payload


def test_case_success_preserves_signature_arguments_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_cleanup(error: BaseException) -> None:
        pytest.fail(f"success triggered failure cleanup: {error}")

    monkeypatch.setattr(
        "benchmarks.shared.failures._clear_exception_frames", unexpected_cleanup
    )

    def case(value: Payload, *, option: int) -> tuple[Payload, int]:
        return value, option

    run = clear_failed_case_frames(case)
    payload = Payload()
    result = run(payload, option=7)

    assert result[0] is payload
    assert result[1] == 7
    assert inspect.signature(run) == inspect.signature(case)
