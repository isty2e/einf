from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeVar

from ..axis import AxisSide, AxisTerms
from ..reduction.callable import CallableReducerBinding
from ..reduction.schema import (
    CanonicalReducer,
    ReducerName,
    ReducerPlan,
)
from .kind import OperationKind

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class BaseOpCacheKey:
    """Deterministic cache key for one base TensorOp instance."""

    kind: OperationKind
    lhs: AxisSide
    rhs: AxisSide


ReducerToken = ReducerName | CallableReducerBinding
ReducerPlanKey = tuple[tuple[AxisTerms, ReducerToken], ...]


@dataclass(frozen=True, slots=True)
class ConfiguredOpCacheKey:
    """Deterministic cache key for one configured TensorOp instance."""

    base: BaseOpCacheKey
    sizes_items: tuple[tuple[str, int], ...]
    reducer_plan_key: ReducerPlanKey | None


def reducer_plan_to_cache_key(
    reducer_plan: ReducerPlan | None,
    /,
) -> ReducerPlanKey | None:
    """Build deterministic configured-cache key for one reducer plan."""
    if reducer_plan is None:
        return None
    phases: list[tuple[AxisTerms, ReducerToken]] = []
    for phase in reducer_plan:
        phases.append(
            (
                AxisTerms.from_spec(phase.axes),
                _reducer_to_cache_token(phase.reducer),
            )
        )
    return tuple(phases)


def _reducer_to_cache_token(reducer: CanonicalReducer, /) -> ReducerToken:
    """Build stable configured-cache token for one reducer."""
    if isinstance(reducer, ReducerName):
        return reducer
    return reducer


class BoundedTensorOpCache(Generic[KeyT, ValueT]):
    """Thread-safe bounded LRU cache for TensorOp instances."""

    def __init__(self, *, max_size: int) -> None:
        self._max_size = max_size
        self._lock = RLock()
        self._entries: OrderedDict[KeyT, ValueT] = OrderedDict()

    def get_or_create(
        self,
        *,
        key: KeyT,
        builder: Callable[[], ValueT],
    ) -> ValueT:
        """Return cached TensorOp or create-and-cache one under one key."""
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing

        created = builder()

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing

            self._entries[key] = created
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
        return created


@dataclass(slots=True)
class TensorOpFactory(Generic[ValueT]):
    """Two-level TensorOp object factory cache (base + configured)."""

    base_max_size: int
    configured_max_size: int
    _base_cache: BoundedTensorOpCache[BaseOpCacheKey, ValueT] = field(init=False)
    _configured_cache: BoundedTensorOpCache[ConfiguredOpCacheKey, ValueT] = field(
        init=False
    )

    def __post_init__(self) -> None:
        self._base_cache = BoundedTensorOpCache(max_size=self.base_max_size)
        self._configured_cache = BoundedTensorOpCache(max_size=self.configured_max_size)

    def get_base(
        self,
        *,
        key: BaseOpCacheKey,
        builder: Callable[[], ValueT],
    ) -> ValueT:
        """Resolve one base TensorOp through bounded base-op cache."""
        return self._base_cache.get_or_create(key=key, builder=builder)

    def get_configured(
        self,
        *,
        key: ConfiguredOpCacheKey,
        builder: Callable[[], ValueT],
    ) -> ValueT:
        """Resolve one configured TensorOp through bounded configured-op cache."""
        return self._configured_cache.get_or_create(key=key, builder=builder)
