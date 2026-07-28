from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeAlias, TypeVar

from einf.backend import BackendExecutionIdentity
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram

_SELECTION_CACHE_MAX_ENTRIES = 256
_ROUTE_OUTPUT_INDEX_CACHE_MAX_ENTRIES = 256
_RUNTIME_STEP_CACHE_MAX_ENTRIES = 256

RuntimeStepCacheKey: TypeAlias = (
    tuple[
        tuple[tuple[int, ...], ...],
        BackendExecutionIdentity | None,
    ]
    | BackendExecutionIdentity
    | None
)
RunnerCacheKey: TypeAlias = tuple[
    tuple[type[object], ...],
    BackendExecutionIdentity,
    tuple[tuple[int, ...], ...] | None,
]
CacheKeyT = TypeVar("CacheKeyT")
CacheValueT = TypeVar("CacheValueT")
RunnerKeyT = TypeVar("RunnerKeyT", bound=Hashable)
RunnerT = TypeVar("RunnerT")


@dataclass(frozen=True, slots=True)
class _LastCacheHit(Generic[CacheKeyT, CacheValueT]):
    """One immutable key/value snapshot for a cache fast path."""

    key: CacheKeyT
    value: CacheValueT


@dataclass(frozen=True, slots=True)
class SelectionCacheKey:
    """Deterministic cache key for symbolic-candidate selection."""

    input_shapes: tuple[tuple[int, ...], ...]
    explicit_sizes: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class SelectionCache:
    """LRU cache with fast last-hit path for symbolic candidate selection."""

    max_entries: int = _SELECTION_CACHE_MAX_ENTRIES
    _entries: OrderedDict[SelectionCacheKey, int] = field(default_factory=OrderedDict)
    _lock: RLock = field(default_factory=RLock)
    _last_hit: _LastCacheHit[SelectionCacheKey, int] | None = None

    def get_index(self, key: SelectionCacheKey, /) -> int | None:
        """Return cached candidate index for one key, if present."""
        last_hit = self._last_hit
        if last_hit is not None and key == last_hit.key:
            return last_hit.value
        with self._lock:
            cached_index = self._entries.get(key)
            if cached_index is None:
                return None
            self._entries.move_to_end(key)
            self._last_hit = _LastCacheHit(key=key, value=cached_index)
        return cached_index

    def set_index(self, key: SelectionCacheKey, index: int, /) -> None:
        """Store one selected candidate index under key."""
        with self._lock:
            self._entries[key] = index
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._last_hit = _LastCacheHit(key=key, value=index)


@dataclass(slots=True)
class RouteOutputIndexCache:
    """Route-output index cache with static-route short-circuit."""

    static_output_indices: tuple[int, ...] | None
    max_entries: int = _ROUTE_OUTPUT_INDEX_CACHE_MAX_ENTRIES
    _entries: OrderedDict[tuple[tuple[int, ...], ...], tuple[int, ...]] = field(
        default_factory=OrderedDict
    )
    _lock: RLock = field(default_factory=RLock)
    _last_hit: _LastCacheHit[tuple[tuple[int, ...], ...], tuple[int, ...]] | None = None

    def get(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        /,
    ) -> tuple[int, ...] | None:
        """Return cached output indices for one route input-shape tuple."""
        if self.static_output_indices is not None:
            return self.static_output_indices
        last_hit = self._last_hit
        if last_hit is not None and input_shapes == last_hit.key:
            return last_hit.value
        with self._lock:
            cached_output_indices = self._entries.get(input_shapes)
            if cached_output_indices is None:
                return None
            self._entries.move_to_end(input_shapes)
            self._last_hit = _LastCacheHit(
                key=input_shapes,
                value=cached_output_indices,
            )
        return cached_output_indices

    def set(
        self,
        *,
        input_shapes: tuple[tuple[int, ...], ...],
        output_indices: tuple[int, ...],
    ) -> None:
        """Store one route-output index mapping."""
        with self._lock:
            self._entries[input_shapes] = output_indices
            self._entries.move_to_end(input_shapes)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._last_hit = _LastCacheHit(
                key=input_shapes,
                value=output_indices,
            )


@dataclass(slots=True)
class RuntimeStepSpecializationCache:
    """LRU cache for symbolic-plan runtime-step specialization."""

    depends_on_input_shapes: bool
    max_entries: int = _RUNTIME_STEP_CACHE_MAX_ENTRIES
    _entries: OrderedDict[
        RuntimeStepCacheKey,
        tuple[RuntimeStep[StepProgram], ...],
    ] = field(default_factory=OrderedDict)
    _lock: RLock = field(default_factory=RLock)
    _last_hit: (
        _LastCacheHit[RuntimeStepCacheKey, tuple[RuntimeStep[StepProgram], ...]] | None
    ) = None

    def get(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> tuple[RuntimeStep[StepProgram], ...] | None:
        """Return cached specialized runtime steps, if present."""
        cache_key = self.key(context)
        last_hit = self._last_hit
        if last_hit is not None and cache_key == last_hit.key:
            return last_hit.value
        with self._lock:
            cached_steps = self._entries.get(cache_key)
            if cached_steps is None:
                return None
            self._entries.move_to_end(cache_key)
            self._last_hit = _LastCacheHit(key=cache_key, value=cached_steps)
        return cached_steps

    def set(
        self,
        *,
        context: RuntimeSpecializationContext,
        steps: tuple[RuntimeStep[StepProgram], ...],
    ) -> None:
        """Store specialized runtime steps for one specialization context."""
        cache_key = self.key(context)
        with self._lock:
            self._entries[cache_key] = steps
            self._entries.move_to_end(cache_key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._last_hit = _LastCacheHit(key=cache_key, value=steps)

    def key(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStepCacheKey:
        """Build deterministic specialization cache key."""
        backend_profile = context.backend_profile
        backend_identity = (
            None if backend_profile is None else backend_profile.execution_identity
        )
        if self.depends_on_input_shapes:
            return (context.input_shapes, backend_identity)
        return backend_identity


@dataclass(slots=True)
class RunnerCache(Generic[RunnerKeyT, RunnerT]):
    """LRU cache for compiled runtime runners."""

    max_entries: int = _RUNTIME_STEP_CACHE_MAX_ENTRIES
    _entries: OrderedDict[RunnerKeyT, RunnerT] = field(default_factory=OrderedDict)
    _lock: RLock = field(default_factory=RLock)
    _last_hit: _LastCacheHit[RunnerKeyT, RunnerT] | None = None

    def get(self, key: RunnerKeyT, /) -> RunnerT | None:
        """Return cached runner for one runtime runner cache key."""
        last_hit = self._last_hit
        if last_hit is not None and key == last_hit.key:
            return last_hit.value
        with self._lock:
            runner = self._entries.get(key)
            if runner is None:
                return None
            self._entries.move_to_end(key)
            self._last_hit = _LastCacheHit(key=key, value=runner)
        return runner

    def set(self, key: RunnerKeyT, runner: RunnerT, /) -> None:
        """Store one compiled runner under one runtime runner cache key."""
        with self._lock:
            self._entries[key] = runner
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._last_hit = _LastCacheHit(key=key, value=runner)


__all__ = [
    "RouteOutputIndexCache",
    "RunnerCache",
    "RunnerCacheKey",
    "RuntimeStepCacheKey",
    "RuntimeStepSpecializationCache",
    "SelectionCache",
    "SelectionCacheKey",
]
