from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeAlias, TypeVar

from einf.backend import BackendProfile
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, StepProgram
from einf.tensor_types import TensorLike

_SELECTION_CACHE_MAX_ENTRIES = 256
_BACKEND_PROFILE_CACHE_MAX_ENTRIES = 64
_ROUTE_OUTPUT_INDEX_CACHE_MAX_ENTRIES = 256
_RUNTIME_STEP_CACHE_MAX_ENTRIES = 256

RuntimeStepCacheKey: TypeAlias = (
    tuple[tuple[tuple[int, ...], ...], str | None] | str | None
)
RunnerCacheKey: TypeAlias = tuple[
    tuple[type[object], ...], tuple[tuple[int, ...], ...] | None
]
RunnerT = TypeVar("RunnerT")


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
    _last_key: SelectionCacheKey | None = None
    _last_index: int | None = None

    def get_index(self, key: SelectionCacheKey, /) -> int | None:
        """Return cached candidate index for one key, if present."""
        if key == self._last_key:
            return self._last_index
        with self._lock:
            cached_index = self._entries.get(key)
            if cached_index is None:
                return None
            self._entries.move_to_end(key)
        self._last_key = key
        self._last_index = cached_index
        return cached_index

    def set_index(self, key: SelectionCacheKey, index: int, /) -> None:
        """Store one selected candidate index under key."""
        with self._lock:
            self._entries[key] = index
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        self._last_key = key
        self._last_index = index


@dataclass(slots=True)
class BackendProfileCache:
    """LRU cache for backend-profile resolution by input tensor types."""

    max_entries: int = _BACKEND_PROFILE_CACHE_MAX_ENTRIES
    _entries: OrderedDict[tuple[type[object], ...], BackendProfile] = field(
        default_factory=OrderedDict
    )
    _lock: RLock = field(default_factory=RLock)
    _last_key: tuple[type[object], ...] | None = None
    _last_profile: BackendProfile | None = None
    _last_unary_type: type[object] | None = None
    _last_unary_profile: BackendProfile | None = None

    def get(self, tensors: tuple[TensorLike, ...], /) -> BackendProfile | None:
        """Return cached backend profile for runtime tensors, if present."""
        if len(tensors) == 1:
            unary_type = type(tensors[0])
            if unary_type is self._last_unary_type:
                last_unary_profile = self._last_unary_profile
                if last_unary_profile is not None:
                    return last_unary_profile

        cache_key = tuple(type(tensor) for tensor in tensors)
        if cache_key == self._last_key:
            if self._last_profile is not None:
                return self._last_profile
        with self._lock:
            cached_profile = self._entries.get(cache_key)
            if cached_profile is None:
                return None
            self._entries.move_to_end(cache_key)
        self._set_last(cache_key=cache_key, profile=cached_profile, tensors=tensors)
        return cached_profile

    def set(
        self,
        *,
        tensors: tuple[TensorLike, ...],
        profile: BackendProfile,
    ) -> None:
        """Store one resolved backend profile."""
        cache_key = tuple(type(tensor) for tensor in tensors)
        with self._lock:
            self._entries[cache_key] = profile
            self._entries.move_to_end(cache_key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        self._set_last(cache_key=cache_key, profile=profile, tensors=tensors)

    def _set_last(
        self,
        *,
        cache_key: tuple[type[object], ...],
        profile: BackendProfile,
        tensors: tuple[TensorLike, ...],
    ) -> None:
        """Update fast last-hit backend cache state."""
        self._last_key = cache_key
        self._last_profile = profile
        if len(tensors) == 1:
            self._last_unary_type = type(tensors[0])
            self._last_unary_profile = profile


@dataclass(slots=True)
class RouteOutputIndexCache:
    """Route-output index cache with static-route short-circuit."""

    static_output_indices: tuple[int, ...] | None
    max_entries: int = _ROUTE_OUTPUT_INDEX_CACHE_MAX_ENTRIES
    _entries: OrderedDict[tuple[tuple[int, ...], ...], tuple[int, ...]] = field(
        default_factory=OrderedDict
    )
    _lock: RLock = field(default_factory=RLock)
    _last_input_shapes: tuple[tuple[int, ...], ...] | None = None
    _last_output_indices: tuple[int, ...] | None = None

    def get(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        /,
    ) -> tuple[int, ...] | None:
        """Return cached output indices for one route input-shape tuple."""
        if self.static_output_indices is not None:
            return self.static_output_indices
        if input_shapes == self._last_input_shapes:
            last_output_indices = self._last_output_indices
            if last_output_indices is not None:
                return last_output_indices
        with self._lock:
            cached_output_indices = self._entries.get(input_shapes)
            if cached_output_indices is None:
                return None
            self._entries.move_to_end(input_shapes)
        self._last_input_shapes = input_shapes
        self._last_output_indices = cached_output_indices
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
        self._last_input_shapes = input_shapes
        self._last_output_indices = output_indices


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
    _last_key: RuntimeStepCacheKey | None = None
    _last_steps: tuple[RuntimeStep[StepProgram], ...] | None = None

    def get(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> tuple[RuntimeStep[StepProgram], ...] | None:
        """Return cached specialized runtime steps, if present."""
        cache_key = self.key(context)
        if cache_key == self._last_key:
            if self._last_steps is not None:
                return self._last_steps
        with self._lock:
            cached_steps = self._entries.get(cache_key)
            if cached_steps is None:
                return None
            self._entries.move_to_end(cache_key)
        self._last_key = cache_key
        self._last_steps = cached_steps
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
        self._last_key = cache_key
        self._last_steps = steps

    def key(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStepCacheKey:
        """Build deterministic specialization cache key."""
        backend_profile = context.backend_profile
        backend_namespace_id = (
            None if backend_profile is None else backend_profile.namespace_id
        )
        if self.depends_on_input_shapes:
            return (context.input_shapes, backend_namespace_id)
        return backend_namespace_id


@dataclass(slots=True)
class RunnerCache(Generic[RunnerT]):
    """LRU cache for compiled runtime runners."""

    max_entries: int = _RUNTIME_STEP_CACHE_MAX_ENTRIES
    _entries: OrderedDict[RunnerCacheKey, RunnerT] = field(default_factory=OrderedDict)
    _lock: RLock = field(default_factory=RLock)
    _last_key: RunnerCacheKey | None = None
    _last_runner: RunnerT | None = None

    def get(self, key: RunnerCacheKey, /) -> RunnerT | None:
        """Return cached runner for one runtime runner cache key."""
        if key == self._last_key:
            return self._last_runner
        with self._lock:
            runner = self._entries.get(key)
            if runner is None:
                return None
            self._entries.move_to_end(key)
        self._last_key = key
        self._last_runner = runner
        return runner

    def set(self, key: RunnerCacheKey, runner: RunnerT, /) -> None:
        """Store one compiled runner under one runtime runner cache key."""
        with self._lock:
            self._entries[key] = runner
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        self._last_key = key
        self._last_runner = runner


__all__ = [
    "BackendProfileCache",
    "RouteOutputIndexCache",
    "RunnerCache",
    "RunnerCacheKey",
    "RuntimeStepSpecializationCache",
    "RuntimeStepCacheKey",
    "SelectionCache",
    "SelectionCacheKey",
]
