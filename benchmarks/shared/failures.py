import sys
from collections.abc import Callable, Iterator
from functools import wraps
from traceback import walk_tb
from types import FrameType, TracebackType
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
Result = TypeVar("Result")


def _exception_tracebacks(error: BaseException | None) -> Iterator[TracebackType]:
    pending = [] if error is None else [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            yield current.__traceback__

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _clear_exception_frames(
    error: BaseException, protected_frames: frozenset[FrameType]
) -> None:
    for tb in _exception_tracebacks(error):
        for frame, _ in walk_tb(tb):
            if frame in protected_frames:
                continue
            try:
                frame.clear()
            except RuntimeError:
                # Like traceback.clear_frames, leave executing frames alone.
                pass


def clear_failed_case_frames(case_call: Callable[P, Result]) -> Callable[P, Result]:
    """Release completed execution-frame locals when a benchmark case fails.

    Parameters
    ----------
    case_call : Callable[P, Result]
        Case entrypoint owning the prepared inputs and validation temporaries.

    Returns
    -------
    Callable[P, Result]
        Entrypoint with unchanged results and per-coordinate execution. On an
        ordinary exception, clears completed traceback frames in its chains and
        groups, except frames already reachable from the active exception on
        entry. Re-raising that exception still clears newly added case frames.
        The original exception and traceback locations remain available;
        cleared frame locals do not.

    Raises
    ------
    Exception
        The wrapped case's original failure, with completed case frame locals cleared.

    Notes
    -----
    Does not clear callback closures, caller-owned references, or payloads
    explicitly stored in user exception attributes. Caller traceback frames are
    snapshotted once per case. Apply at case boundaries, not inside timed loops.
    """

    @wraps(case_call)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result:
        protected_frames = frozenset(
            frame
            for tb in _exception_tracebacks(sys.exception())
            for frame, _ in walk_tb(tb)
        )
        try:
            return case_call(*args, **kwargs)
        except Exception as error:
            # This wrapper is still executing, so clear_frames cannot clear it.
            del args, kwargs
            _clear_exception_frames(error, protected_frames)
            del protected_frames
            raise

    return run
