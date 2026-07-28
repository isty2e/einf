from dataclasses import replace
from threading import Event, Thread
from typing import Callable, TypeVar, cast

import numpy as np

from einf.backend import BACKEND_RESOLVER
from einf.plans.cache import (
    BackendProfileCache,
    RouteOutputIndexCache,
    RunnerCache,
    RunnerCacheKey,
    RuntimeStepSpecializationCache,
    SelectionCache,
    SelectionCacheKey,
)
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram
from einf.tensor_types import TensorLike

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


class _InterleavingBackendProfileCache(_LastHitReadBarrier, BackendProfileCache):
    pass


class _InterleavingRouteOutputIndexCache(_LastHitReadBarrier, RouteOutputIndexCache):
    pass


class _InterleavingRuntimeStepCache(
    _LastHitReadBarrier,
    RuntimeStepSpecializationCache,
):
    pass


class _InterleavingRunnerCache(_LastHitReadBarrier, RunnerCache[str]):
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
    results: list[ResultT | None] = []
    failures: list[Exception] = []

    def run_getter() -> None:
        try:
            results.append(getter())
        except Exception as error:
            failures.append(error)

    getter_thread = Thread(target=run_getter, daemon=True)
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

    assert not getter_thread.is_alive()
    if failures:
        raise AssertionError(
            "cache getter failed during forced interleaving"
        ) from failures[0]
    assert len(results) == 1
    return results[0]


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


def test_backend_profile_cache_publishes_unary_last_hit_atomically() -> None:
    cache = _InterleavingBackendProfileCache()
    tensor_a = np.zeros((2, 3))
    tensor_b = cast(TensorLike, np.zeros((3, 2)).view(_AlternateArray))
    profile_a = BACKEND_RESOLVER.resolve(tensor_a, op_name="rearrange")
    profile_b = replace(profile_a, namespace_id="alternate")
    cache.set(tensors=(tensor_a,), profile=profile_a)

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_unary_hit", "_last_unary_type"),
        getter=lambda: cache.get((tensor_a,)),
        replace_last_hit=lambda: cache.set(tensors=(tensor_b,), profile=profile_b),
    )

    assert result is profile_a


def test_backend_profile_cache_publishes_general_last_hit_atomically() -> None:
    cache = _InterleavingBackendProfileCache()
    tensor_a = np.zeros((2, 3))
    tensor_b = cast(TensorLike, np.zeros((3, 2)).view(_AlternateArray))
    tensors_a: tuple[TensorLike, ...] = (tensor_a, tensor_a)
    tensors_b: tuple[TensorLike, ...] = (tensor_b, tensor_a)
    profile_a = BACKEND_RESOLVER.resolve(*tensors_a, op_name="contract")
    profile_b = replace(profile_a, namespace_id="alternate")
    cache.set(tensors=tensors_a, profile=profile_a)

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_key"),
        getter=lambda: cache.get(tensors_a),
        replace_last_hit=lambda: cache.set(tensors=tensors_b, profile=profile_b),
    )

    assert result is profile_a


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
    key_a: RunnerCacheKey = ((np.ndarray,), None)
    key_b: RunnerCacheKey = ((_AlternateArray,), None)
    cache.set(key_a, "runner-a")

    result = _replace_last_hit_after_state_read(
        cache=cache,
        state_attributes=("_last_hit", "_last_key"),
        getter=lambda: cache.get(key_a),
        replace_last_hit=lambda: cache.set(key_b, "runner-b"),
    )

    assert result == "runner-a"
