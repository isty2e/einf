import sys
from collections.abc import Callable
from functools import wraps
from traceback import clear_frames
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
Result = TypeVar("Result")


def _clear_exception_frames(
    error: BaseException, inherited: BaseException | None
) -> None:
    pending = [error]
    seen = set() if inherited is None else {id(inherited)}
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        clear_frames(current.__traceback__)

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


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
        ordinary exception, clears completed traceback frames for exceptions
        raised within the case, including chains and groups, then re-raises the
        original exception. Stops at the exception active on entry. Traceback
        locations remain available; completed case frame locals do not.

    Raises
    ------
    Exception
        The wrapped case's original failure, with completed case frame locals cleared.

    Notes
    -----
    Does not clear callback closures, caller-owned references, or payloads
    explicitly stored in user exception attributes. Apply at case boundaries,
    not inside timed loops.
    """

    @wraps(case_call)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result:
        inherited = sys.exception()
        try:
            return case_call(*args, **kwargs)
        except Exception as error:
            # This wrapper is still executing, so clear_frames cannot clear it.
            del args, kwargs
            _clear_exception_frames(error, inherited)
            del inherited
            raise

    return run
