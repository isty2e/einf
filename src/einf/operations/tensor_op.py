from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Generic, TypeVar, final, overload

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..axis import AxisSide, AxisTerms
from ..diagnostics import ErrorCode, ValidationError
from ..lowering import DefaultLoweringProgram
from ..output_normalization import normalize_runtime_outputs
from ..plans.abstract import AbstractPlan, RuntimeSpecializationContext
from ..plans.entrypoint import execute_tensor_op_call, extract_input_shapes
from ..plans.render import PlanDict, build_plan_dict, render_plan_text
from ..reduction.plan import ReducerPlanParser
from ..reduction.schema import Reducer, ReducerCallable, ReducerPlan
from ..signature import Signature
from ..tensor_types import TensorLike
from .policy import OpPolicy, resolve_op_policy

RuntimeTypeKey = tuple[type, ...]
_DEFAULT_LOWERING_PROGRAM = DefaultLoweringProgram()
_BASE_OP_CACHE_MAX_SIZE = 512
_CONFIGURED_OP_CACHE_MAX_SIZE = 2_048


@dataclass(frozen=True, slots=True)
class _BaseOpCacheKey:
    """Deterministic cache key for one base TensorOp instance."""

    name: str
    lhs: AxisSide
    rhs: AxisSide
    supports_reducer: bool


@dataclass(frozen=True, slots=True, eq=False)
class _ReducerCallableToken:
    """Identity token for callable reducers in configured-op cache keys."""

    reducer: ReducerCallable

    def __hash__(self) -> int:
        """Hash by callable identity to keep key stable while object is alive."""
        return id(self.reducer)

    def __eq__(self, other: object) -> bool:
        """Compare callable reducer tokens by identity."""
        if not isinstance(other, _ReducerCallableToken):
            return False
        return self.reducer is other.reducer


_ReducerToken = str | _ReducerCallableToken
_ReducerPlanKey = tuple[tuple[AxisTerms, _ReducerToken], ...]


@dataclass(frozen=True, slots=True)
class _ConfiguredOpCacheKey:
    """Deterministic cache key for one configured TensorOp instance."""

    base: _BaseOpCacheKey
    sizes_items: tuple[tuple[str, int], ...]
    reducer_plan_key: _ReducerPlanKey | None


KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


def _normalize_sizes_items(
    *,
    op_name: str,
    sizes_items: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    """Validate and normalize size bindings to one immutable sorted tuple."""
    merged: dict[str, int] = {}
    for key, value in sizes_items:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"size binding for {key!r} must be an int")
        if value < 0:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: negative with_sizes binding for {key!r}",
                help="provide non-negative with_sizes bindings",
                related=("with_sizes binding",),
                data={"operation": op_name, "dim": key, "value": value},
            )
        merged[key] = value
    return tuple(sorted(merged.items()))


def _reducer_to_cache_token(reducer: Reducer, /) -> _ReducerToken:
    """Build stable configured-cache token for one reducer."""
    if isinstance(reducer, str):
        return reducer
    return _ReducerCallableToken(reducer)


def _reducer_plan_to_cache_key(
    reducer_plan: ReducerPlan | None,
    /,
) -> _ReducerPlanKey | None:
    """Build deterministic configured-cache key for one reducer plan."""
    if reducer_plan is None:
        return None
    phases: list[tuple[AxisTerms, _ReducerToken]] = []
    for phase in reducer_plan:
        phases.append(
            (
                AxisTerms.from_spec(phase.axes),
                _reducer_to_cache_token(phase.reducer),
            )
        )
    return tuple(phases)


@final
@dataclass(frozen=True, slots=True)
class TensorOp:
    """First-class transform operation with immutable configuration."""

    name: str
    lhs: AxisSide
    rhs: AxisSide
    supports_reducer: bool = False
    reducer_plan: ReducerPlan | None = None
    _sizes_items: tuple[tuple[str, int], ...] = ()
    _signature: Signature = field(init=False, repr=False)
    _abstract_plan: AbstractPlan = field(init=False, repr=False)
    _input_arity: int = field(init=False, repr=False)
    _output_arity: int = field(init=False, repr=False)
    _op_policy: OpPolicy = field(init=False, repr=False)
    _shape_free_context: RuntimeSpecializationContext | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _shape_free_single_runners: dict[
        RuntimeTypeKey,
        Callable[[tuple[TensorLike, ...]], TensorLike],
    ] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _shape_free_tuple_runners: dict[
        RuntimeTypeKey,
        Callable[[tuple[TensorLike, ...]], tuple[TensorLike, ...]],
    ] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        """Normalize source-side specs to canonical immutable tuples."""
        normalized = Signature(inputs=self.lhs, outputs=self.rhs)
        op_policy = resolve_op_policy(self.name)
        op_policy.validate_constructor(
            op_name=self.name,
            lhs=normalized.inputs,
            rhs=normalized.outputs,
        )
        normalized_sizes_items = _normalize_sizes_items(
            op_name=self.name,
            sizes_items=self._sizes_items,
        )
        object.__setattr__(self, "lhs", normalized.inputs)
        object.__setattr__(self, "rhs", normalized.outputs)
        object.__setattr__(self, "_sizes_items", normalized_sizes_items)
        object.__setattr__(self, "_signature", normalized)
        object.__setattr__(self, "_input_arity", len(normalized.inputs))
        object.__setattr__(self, "_output_arity", len(normalized.outputs))
        object.__setattr__(self, "_op_policy", op_policy)
        abstract_plan = AbstractPlan(
            op_name=self.name,
            lhs=normalized.inputs,
            rhs=normalized.outputs,
            explicit_sizes_items=normalized_sizes_items,
            lowering=_DEFAULT_LOWERING_PROGRAM.with_reducer_plan(self.reducer_plan),
        )
        object.__setattr__(
            self,
            "_abstract_plan",
            abstract_plan,
        )
        if not abstract_plan.requires_input_shapes(self._input_arity):
            object.__setattr__(
                self,
                "_shape_free_context",
                RuntimeSpecializationContext(
                    input_shapes=tuple(() for _ in range(self._input_arity)),
                    backend_profile=None,
                ),
            )

    @classmethod
    def from_base_spec(
        cls,
        *,
        name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        supports_reducer: bool = False,
    ):
        """Return cached base TensorOp for one normalized constructor spec."""
        base_key = _BaseOpCacheKey(
            name=name,
            lhs=lhs,
            rhs=rhs,
            supports_reducer=supports_reducer,
        )
        return _TENSOR_OP_FACTORY.get_base(
            key=base_key,
            builder=lambda: cls(
                name=name,
                lhs=lhs,
                rhs=rhs,
                supports_reducer=supports_reducer,
            ),
        )

    @property
    def signature(self) -> Signature:
        """Derived normalized signature view of current lhs/rhs."""
        return self._signature

    @property
    def abstract_plan(self) -> AbstractPlan:
        """Canonical abstract execution plan for this TensorOp."""
        return self._abstract_plan

    @property
    def sizes(self) -> dict[str, int]:
        """Return explicit size bindings as a detached mapping copy."""
        return dict(self._sizes_items)

    @property
    def sizes_items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical immutable explicit size bindings."""
        return self._sizes_items

    def _base_cache_key(self) -> _BaseOpCacheKey:
        """Build deterministic base-key for this TensorOp family."""
        return _BaseOpCacheKey(
            name=self.name,
            lhs=self.lhs,
            rhs=self.rhs,
            supports_reducer=self.supports_reducer,
        )

    def _configured_cache_key(
        self,
        *,
        sizes_items: tuple[tuple[str, int], ...],
        reducer_plan: ReducerPlan | None,
    ) -> _ConfiguredOpCacheKey:
        """Build deterministic configured-key for one TensorOp variant."""
        return _ConfiguredOpCacheKey(
            base=self._base_cache_key(),
            sizes_items=sizes_items,
            reducer_plan_key=_reducer_plan_to_cache_key(reducer_plan),
        )

    def with_sizes(self, **sizes: int):
        """Return a new operation with additional dimension bindings.

        Parameters
        ----------
        **sizes
            Non-negative integer dimension bindings by symbol name.

        Returns
        -------
        TensorOp
            New operation instance with merged dimension bindings.

        Raises
        ------
        TypeError
            If a binding value is not an integer.
        ValueError
            If a binding value is negative.
        """
        if not sizes:
            return self

        merged = dict(self._sizes_items)
        for key, value in sizes.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"size binding for {key!r} must be an int")
            if value < 0:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=f"inconsistent dims: negative with_sizes binding for {key!r}",
                    help="provide non-negative with_sizes bindings",
                    related=("with_sizes binding",),
                    data={"operation": self.name, "dim": key, "value": value},
                )
            merged[key] = value

        normalized_sizes_items = tuple(sorted(merged.items()))
        if normalized_sizes_items == self._sizes_items:
            return self

        configured_key = self._configured_cache_key(
            sizes_items=normalized_sizes_items,
            reducer_plan=self.reducer_plan,
        )
        return _TENSOR_OP_FACTORY.get_configured(
            key=configured_key,
            builder=lambda: replace(self, _sizes_items=normalized_sizes_items),
        )

    @overload
    def reduce_by(self, reducer: str) -> Self: ...

    @overload
    def reduce_by(
        self,
        reducer: ReducerCallable,
    ) -> Self: ...

    @overload
    def reduce_by(
        self, reducer: tuple[AxisTerms, str], *phases: tuple[AxisTerms, str]
    ) -> Self: ...

    @overload
    def reduce_by(
        self,
        reducer: tuple[AxisTerms, ReducerCallable],
        *phases: tuple[AxisTerms, ReducerCallable],
    ) -> Self: ...

    @overload
    def reduce_by(
        self,
        reducer: tuple[AxisTerms, str | ReducerCallable],
        *phases: tuple[AxisTerms, str | ReducerCallable],
    ) -> Self: ...

    def reduce_by(
        self,
        reducer: Reducer | tuple[AxisTerms, Reducer],
        *phases: tuple[AxisTerms, Reducer],
    ):
        """Return a new operation with a custom reducer strategy.

        Parameters
        ----------
        reducer
            Reducer name/callable, or one reducer phase tuple `(ax[...], reducer)`.
        *phases
            Additional ordered reducer phases.

        Returns
        -------
        TensorOp
            New operation carrying the reducer.

        Raises
        ------
        AttributeError
            If this operation does not support reducer customization.
        """
        if not self.supports_reducer:
            raise AttributeError(
                f"{self.name} does not support .reduce_by(...) in v0.1"
            )

        if isinstance(reducer, dict):
            raise TypeError(
                "dict reducer plans are not supported; "
                "use ordered phase tuples like reduce_by((ax[h], 'sum'), (ax[d], 'prod'))"
            )
        reducer_plan = ReducerPlanParser(lhs=self.lhs, rhs=self.rhs).parse(
            reducer=reducer,
            phases=phases,
        )
        if reducer_plan == self.reducer_plan:
            return self

        configured_key = self._configured_cache_key(
            sizes_items=self._sizes_items,
            reducer_plan=reducer_plan,
        )
        return _TENSOR_OP_FACTORY.get_configured(
            key=configured_key,
            builder=lambda: replace(self, reducer_plan=reducer_plan),
        )

    def plan_dict(self) -> PlanDict:
        """Return a deterministic symbolic plan payload."""
        return build_plan_dict(
            op_name=self.name,
            lhs=self.lhs,
            rhs=self.rhs,
            sizes=self.sizes,
            abstract_plan=self.abstract_plan,
        )

    def plan(self) -> str:
        """Return a human-readable symbolic plan preview."""
        return render_plan_text(self.plan_dict())

    def _runtime_type_key(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> RuntimeTypeKey:
        """Build one arity-agnostic runtime input type key."""
        tensor_count = len(tensors)
        if tensor_count == 1:
            return (type(tensors[0]),)
        if tensor_count == 2:
            return (
                type(tensors[0]),
                type(tensors[1]),
            )
        return tuple(type(tensor) for tensor in tensors)

    def __call__(self, *tensors: TensorLike) -> TensorLike | tuple[TensorLike, ...]:
        """Execute the operation with exact-arity tensor inputs."""
        match (
            len(tensors) == self._input_arity,
            self._shape_free_context,
            self._output_arity,
        ):
            case (True, RuntimeSpecializationContext() as shape_free_context, 1):
                runtime_type_key = self._runtime_type_key(tensors)
                runner = self._shape_free_single_runners.get(runtime_type_key)
                if runner is None:
                    _ = extract_input_shapes(op_name=self.name, tensors=tensors)
                    runner = self.abstract_plan.resolve_single_output_runner(
                        shape_free_context,
                        tensors,
                    )
                    self._shape_free_single_runners[runtime_type_key] = runner
                return runner(tensors)

            case (True, RuntimeSpecializationContext() as shape_free_context, _):
                runtime_type_key = self._runtime_type_key(tensors)
                tuple_runner = self._shape_free_tuple_runners.get(runtime_type_key)
                if tuple_runner is None:
                    _ = extract_input_shapes(op_name=self.name, tensors=tensors)
                    tuple_runner = self.abstract_plan.resolve_tuple_runner(
                        shape_free_context,
                        tensors,
                    )
                    self._shape_free_tuple_runners[runtime_type_key] = tuple_runner
                raw_outputs = tuple_runner(tensors)
                if len(raw_outputs) == self._output_arity:
                    return raw_outputs
                return normalize_runtime_outputs(
                    op_name=self.name,
                    expected_output_arity=self._output_arity,
                    raw_outputs=raw_outputs,
                )

            case _:
                return execute_tensor_op_call(
                    self.name,
                    self._input_arity,
                    self._output_arity,
                    self._op_policy,
                    self.abstract_plan,
                    tensors,
                )


class _BoundedTensorOpCache(Generic[KeyT, ValueT]):
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
class _TensorOpFactory:
    """Two-level TensorOp object factory cache (base + configured)."""

    base_max_size: int
    configured_max_size: int
    _base_cache: _BoundedTensorOpCache[_BaseOpCacheKey, TensorOp] = field(init=False)
    _configured_cache: _BoundedTensorOpCache[_ConfiguredOpCacheKey, TensorOp] = field(
        init=False
    )

    def __post_init__(self) -> None:
        self._base_cache = _BoundedTensorOpCache(max_size=self.base_max_size)
        self._configured_cache = _BoundedTensorOpCache(
            max_size=self.configured_max_size
        )

    def get_base(
        self,
        *,
        key: _BaseOpCacheKey,
        builder: Callable[[], TensorOp],
    ) -> TensorOp:
        """Resolve one base TensorOp through bounded base-op cache."""
        return self._base_cache.get_or_create(key=key, builder=builder)

    def get_configured(
        self,
        *,
        key: _ConfiguredOpCacheKey,
        builder: Callable[[], TensorOp],
    ) -> TensorOp:
        """Resolve one configured TensorOp through bounded configured-op cache."""
        return self._configured_cache.get_or_create(key=key, builder=builder)


_TENSOR_OP_FACTORY = _TensorOpFactory(
    base_max_size=_BASE_OP_CACHE_MAX_SIZE,
    configured_max_size=_CONFIGURED_OP_CACHE_MAX_SIZE,
)
