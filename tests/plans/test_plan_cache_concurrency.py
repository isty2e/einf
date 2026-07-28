from collections.abc import Callable
from queue import Empty, Queue
from threading import Event, Thread
from typing import TypeVar, cast

import numpy as np

from einf.backend import BACKEND_RESOLVER
from einf.plans.cache import (
    RouteOutputIndexCache,
    RunnerCache,
    RunnerCacheKey,
    RuntimeStepSpecializationCache,
    SelectionCache,
    SelectionCacheKey,
)
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram

ResultT = TypeVar("ResultT")

_INTERLEAVING_TIMEOUT_SECONDS = 5.0


class _AlternateArray(np.ndarray):
    pass


class _LastHitReadBarrier:
    _last_hit_read_attributes: frozenset[str]
    _last_hit_read_reached: Event
    _last_hit_read_can_continue: Event

    def arm_last_hit_read(self, *candidate_attributes: str) -> None:
        available_attributes = frozenset(
            attribute for attribute in candidate_attributes if hasattr(self, attribute)
        )
        if len(available_attributes) != 1:
            raise AssertionError(
                "expected exactly one available last-hit state attribute"
            )
        self._last_hit_read_attributes = available_attributes
        self._last_hit_read_reached = Event()
        self._last_hit_read_can_continue = Event()

    def wait_for_last_hit_read(self) -> bool:
        return self._last_hit_read_reached.wait(_INTERLEAVING_TIMEOUT_SECONDS)

    def allow_last_hit_read_to_continue(self) -> None:
        self._last_hit_read_can_continue.set()

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        try:
            read_attributes = super().__getattribute__("_last_hit_read_attributes")
        except AttributeError:
            return value
        if name not in read_attributes:
            return value

        read_reached = super().__getattribute__("_last_hit_read_reached")
        if read_reached.is_set():
            return value
        read_reached.set()
        read_can_continue = super().__getattribute__("_last_hit_read_can_continue")
        if not read_can_continue.wait(_INTERLEAVING_TIMEOUT_SECONDS):
            raise TimeoutError("cache last-hit read interleaving timed out")
        return value


class _InterleavingSelectionCache(_LastHitReadBarrier, SelectionCache):
    pass


class _InterleavingRouteOutputIndexCache(_LastHitReadBarrier, RouteOutputIndexCache):
    pass


class _InterleavingRuntimeStepCache(
    _LastHitReadBarrier,
    RuntimeStepSpecializationCache,
):
    pass


class _InterleavingRunnerCache(_LastHitReadBarrier, RunnerCache[RunnerCacheKey, str]):
    pass


def _replace_last_hit_after_state_read(
    *,
    cache: _LastHitReadBarrier,
    state_attributes: tuple[str, ...],
    getter: Callable[[], ResultT | None],
    replace_last_hit: Callable[[], None],
) -> ResultT | None:
    """Replace a cache's last hit after its current state has been read."""
    cache.arm_last_hit_read(*state_attributes)
    results: Queue[ResultT | None] = Queue(maxsize=1)
    getter_thread = Thread(
        target=lambda: results.put(getter()),
        daemon=True,
    )
    getter_thread.start()
    if not cache.wait_for_last_hit_read():
        cache.allow_last_hit_read_to_continue()
        getter_thread.join(_INTERLEAVING_TIMEOUT_SECONDS)
        raise AssertionError("getter did not reach its last-hit state read")

    try:
        replace_last_hit()
    finally:
        cache.allow_last_hit_read_to_continue()

    getter_thread.join(_INTERLEAVING_TIMEOUT_SECONDS)
    if getter_thread.is_alive():
        raise AssertionError("cache getter did not finish during forced interleaving")
    try:
        return results.get_nowait()
    except Empty as error:
        raise AssertionError(
            "cache getter failed during forced interleaving"
        ) from error


def test_selection_cache_publishes_last_hit_atomically() -> None:
    cache = _InterleavingSelectionCache()
    key_a = SelectionCacheKey(input_shapes=((2, 3),), explicit_sizes=())
    key_b = SelectionCacheKey(input_shapes=((3, 2),), explicit_sizes=())
    cache.set_index(key_a, 11)

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_key"),
        getter=lambda: cache.get_index(key_a),
        replace_last_hit=lambda: cache.set_index(key_b, 22),
    )

    assert result == 11


def test_route_output_index_cache_publishes_last_hit_atomically() -> None:
    cache = _InterleavingRouteOutputIndexCache(static_output_indices=None)
    input_shapes_a = ((2, 3), (3, 4))
    input_shapes_b = ((4, 3), (3, 2))
    output_indices_a = (0, 1)
    output_indices_b = (1, 0)
    cache.set(input_shapes=input_shapes_a, output_indices=output_indices_a)

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_input_shapes"),
        getter=lambda: cache.get(input_shapes_a),
        replace_last_hit=lambda: cache.set(
            input_shapes=input_shapes_b,
            output_indices=output_indices_b,
        ),
    )

    assert result == output_indices_a


def test_runtime_step_cache_publishes_last_hit_atomically() -> None:
    cache = _InterleavingRuntimeStepCache(depends_on_input_shapes=True)
    context_a = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )
    context_b = RuntimeSpecializationContext(
        input_shapes=((3, 2),),
        backend_profile=None,
    )
    steps_a = cast(tuple[RuntimeStep[StepProgram], ...], (object(),))
    steps_b = cast(tuple[RuntimeStep[StepProgram], ...], (object(),))
    cache.set(context=context_a, steps=steps_a)

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_key"),
        getter=lambda: cache.get(context_a),
        replace_last_hit=lambda: cache.set(context=context_b, steps=steps_b),
    )

    assert result is steps_a


def test_runner_cache_publishes_last_hit_atomically() -> None:
    cache = _InterleavingRunnerCache()
    backend_identity = BACKEND_RESOLVER.resolve(
        np.zeros((1,)),
        op_name="rearrange",
    ).execution_identity
    key_a: RunnerCacheKey = ((np.ndarray,), backend_identity, None)
    key_b: RunnerCacheKey = ((_AlternateArray,), backend_identity, None)
    cache.set(key_a, "runner-a")

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_key"),
        getter=lambda: cache.get(key_a),
        replace_last_hit=lambda: cache.set(key_b, "runner-b"),
    )

    assert result == "runner-a"
