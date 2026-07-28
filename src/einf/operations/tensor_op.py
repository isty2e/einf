from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import final, overload

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..axis import AxisSide, AxisTerms
from ..backend import BackendExecutionIdentity
from ..diagnostics import ErrorCode, ValidationError
from ..lowering import DefaultLoweringProgram
from ..lowering.einop.layout import EinopLayoutNormalization
from ..output_normalization import RuntimeOutputContract
from ..plans.abstract import AbstractPlan, RuntimeSpecializationContext
from ..plans.cache import RunnerCache
from ..plans.render import PlanDict, build_plan_dict, render_plan_text
from ..reduction.plan import ReducerPlanParser
from ..reduction.schema import Reducer, ReducerCallable, ReducerPlan
from ..signature import Signature
from ..tensor_types import TensorLike
from .cache import (
    BaseOpCacheKey,
    ConfiguredOpCacheKey,
    TensorOpFactory,
    reducer_plan_to_cache_key,
)
from .execution import execute_tensor_op_call, extract_input_shapes
from .policy import OpPolicy, resolve_op_policy

RuntimeTypeKey = tuple[type[object], ...]
RuntimeRunnerKey = tuple[RuntimeTypeKey, BackendExecutionIdentity]
_DEFAULT_LOWERING_PROGRAM = DefaultLoweringProgram()
_BASE_OP_CACHE_MAX_SIZE = 512
_CONFIGURED_OP_CACHE_MAX_SIZE = 2_048


class _CallMode(Enum):
    """Internal TensorOp execution-mode taxonomy."""

    GENERAL = auto()
    SHAPE_FREE_SINGLE = auto()
    SHAPE_FREE_TUPLE = auto()


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


@dataclass(frozen=True, slots=True)
class TensorOpContract:
    """Canonical immutable TensorOp contract and its normalized products."""

    name: str
    lhs: AxisSide
    rhs: AxisSide
    supports_reducer: bool = False
    reducer_plan: ReducerPlan | None = None
    sizes_items: tuple[tuple[str, int], ...] = ()
    signature: Signature = field(init=False, repr=False)
    abstract_plan: AbstractPlan = field(init=False, repr=False)
    input_arity: int = field(init=False, repr=False)
    output_arity: int = field(init=False, repr=False)
    op_policy: OpPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize constructor inputs and derive the canonical execution contract."""
        normalized = Signature(inputs=self.lhs, outputs=self.rhs)
        op_policy = resolve_op_policy(self.name)
        op_policy.validate_constructor(
            op_name=self.name,
            lhs=normalized.inputs,
            rhs=normalized.outputs,
        )
        normalized_sizes_items = _normalize_sizes_items(
            op_name=self.name,
            sizes_items=self.sizes_items,
        )
        abstract_plan = AbstractPlan(
            op_name=self.name,
            lhs=normalized.inputs,
            rhs=normalized.outputs,
            explicit_sizes_items=normalized_sizes_items,
            lowering=_DEFAULT_LOWERING_PROGRAM.with_reducer_plan(self.reducer_plan),
        )
        object.__setattr__(self, "lhs", normalized.inputs)
        object.__setattr__(self, "rhs", normalized.outputs)
        object.__setattr__(self, "sizes_items", normalized_sizes_items)
        object.__setattr__(self, "signature", normalized)
        object.__setattr__(self, "input_arity", len(normalized.inputs))
        object.__setattr__(self, "output_arity", len(normalized.outputs))
        object.__setattr__(self, "op_policy", op_policy)
        object.__setattr__(self, "abstract_plan", abstract_plan)

    def base_cache_key(self) -> BaseOpCacheKey:
        """Build the deterministic base-op cache key for this contract."""
        return BaseOpCacheKey(
            name=self.name,
            lhs=self.lhs,
            rhs=self.rhs,
            supports_reducer=self.supports_reducer,
        )

    def configured_cache_key(self) -> ConfiguredOpCacheKey:
        """Build the deterministic configured-op cache key for this contract."""
        return ConfiguredOpCacheKey(
            base=self.base_cache_key(),
            sizes_items=self.sizes_items,
            reducer_plan_key=reducer_plan_to_cache_key(self.reducer_plan),
        )

    def with_sizes_items(
        self,
        sizes_items: tuple[tuple[str, int], ...],
        /,
    ) -> "TensorOpContract":
        """Return one contract with updated explicit size bindings."""
        return TensorOpContract(
            name=self.name,
            lhs=self.lhs,
            rhs=self.rhs,
            supports_reducer=self.supports_reducer,
            reducer_plan=self.reducer_plan,
            sizes_items=sizes_items,
        )

    def with_reducer_plan(
        self, reducer_plan: ReducerPlan | None, /
    ) -> "TensorOpContract":
        """Return one contract with updated reducer strategy."""
        return TensorOpContract(
            name=self.name,
            lhs=self.lhs,
            rhs=self.rhs,
            supports_reducer=self.supports_reducer,
            reducer_plan=reducer_plan,
            sizes_items=self.sizes_items,
        )

    def sizes(self) -> dict[str, int]:
        """Return explicit size bindings as one detached mapping."""
        return dict(self.sizes_items)


@dataclass(frozen=True, slots=True)
class TensorOpExecutionStrategy:
    """Stable execution strategy derived from one immutable TensorOp contract."""

    shape_free_context: RuntimeSpecializationContext | None
    call_mode: _CallMode

    @classmethod
    def from_contract(cls, contract: TensorOpContract, /) -> Self:
        """Build one execution strategy from one immutable contract."""
        shape_free_context: RuntimeSpecializationContext | None = None
        if not contract.abstract_plan.specialization_depends_on_input_shapes(
            contract.input_arity
        ):
            shape_free_context = RuntimeSpecializationContext(
                input_shapes=tuple(() for _ in range(contract.input_arity)),
                backend_profile=None,
            )
        if shape_free_context is None:
            call_mode = _CallMode.GENERAL
        elif contract.output_arity == 1:
            call_mode = _CallMode.SHAPE_FREE_SINGLE
        else:
            call_mode = _CallMode.SHAPE_FREE_TUPLE
        return cls(
            shape_free_context=shape_free_context,
            call_mode=call_mode,
        )


@dataclass(slots=True)
class TensorOpRunnerCache:
    """Mutable runner cache for one TensorOp execution strategy."""

    shape_free_single_runners: RunnerCache[
        RuntimeRunnerKey,
        Callable[[tuple[TensorLike, ...]], TensorLike],
    ] = field(default_factory=lambda: RunnerCache())
    shape_free_tuple_runners: RunnerCache[
        RuntimeRunnerKey,
        Callable[[tuple[TensorLike, ...]], tuple[TensorLike, ...]],
    ] = field(default_factory=lambda: RunnerCache())


@final
@dataclass(frozen=True, slots=True)
class TensorOp:
    """First-class transform operation with immutable contract and mutable runtime state."""

    _contract: TensorOpContract
    _execution_strategy: TensorOpExecutionStrategy = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runner_cache: TensorOpRunnerCache = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_output_contract: RuntimeOutputContract = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Attach fresh mutable runtime state to one immutable TensorOp contract."""
        object.__setattr__(
            self,
            "_execution_strategy",
            TensorOpExecutionStrategy.from_contract(self._contract),
        )
        object.__setattr__(self, "_runner_cache", TensorOpRunnerCache())
        object.__setattr__(
            self,
            "_runtime_output_contract",
            RuntimeOutputContract(
                op_name=self._contract.name,
                expected_output_arity=self._contract.output_arity,
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
        contract = TensorOpContract(
            name=name,
            lhs=lhs,
            rhs=rhs,
            supports_reducer=supports_reducer,
        )
        return _TENSOR_OP_FACTORY.get_base(
            key=contract.base_cache_key(),
            builder=lambda: cls(_contract=contract),
        )

    @property
    def name(self) -> str:
        """Public operation name."""
        return self._contract.name

    @property
    def lhs(self) -> AxisSide:
        """Canonical normalized left-hand input signature."""
        return self._contract.lhs

    @property
    def rhs(self) -> AxisSide:
        """Canonical normalized right-hand output signature."""
        return self._contract.rhs

    @property
    def supports_reducer(self) -> bool:
        """Whether this operation accepts `.reduce_by(...)` customization."""
        return self._contract.supports_reducer

    @property
    def reducer_plan(self) -> ReducerPlan | None:
        """Configured reducer plan, when applicable."""
        return self._contract.reducer_plan

    @property
    def signature(self) -> Signature:
        """Derived normalized signature view of current lhs/rhs."""
        return self._contract.signature

    @property
    def abstract_plan(self) -> AbstractPlan:
        """Canonical abstract execution plan for this TensorOp."""
        return self._contract.abstract_plan

    @property
    def sizes(self) -> dict[str, int]:
        """Return explicit size bindings as a detached mapping copy."""
        return self._contract.sizes()

    @property
    def sizes_items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical immutable explicit size bindings."""
        return self._contract.sizes_items

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

        merged = dict(self._contract.sizes_items)
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
        if normalized_sizes_items == self._contract.sizes_items:
            return self

        contract = self._contract.with_sizes_items(normalized_sizes_items)
        return _TENSOR_OP_FACTORY.get_configured(
            key=contract.configured_cache_key(),
            builder=lambda: TensorOp(_contract=contract),
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
        reducer_signature = self.signature
        if self.name == "einop":
            reducer_signature = EinopLayoutNormalization.from_signature(
                reducer_signature
            ).logical
        reducer_parser = ReducerPlanParser(
            lhs=reducer_signature.inputs,
            rhs=reducer_signature.outputs,
        )
        if self.name == "einop" and not reducer_parser.reduced_terms():
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: reduce_by has no logical axes to reduce",
                help="remove reduce_by from einop signatures that preserve every axis",
                related=("einop reducer configuration",),
                data={"operation": "einop"},
            )
        reducer_plan = reducer_parser.parse(
            reducer=reducer,
            phases=phases,
        )
        if reducer_plan == self.reducer_plan:
            return self

        contract = self._contract.with_reducer_plan(reducer_plan)
        return _TENSOR_OP_FACTORY.get_configured(
            key=contract.configured_cache_key(),
            builder=lambda: TensorOp(_contract=contract),
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
        contract = self._contract
        execution_strategy = self._execution_strategy
        runner_cache = self._runner_cache
        output_contract = self._runtime_output_contract
        if (
            len(tensors) != contract.input_arity
            or execution_strategy.call_mode is _CallMode.GENERAL
        ):
            raw_outputs = execute_tensor_op_call(
                contract.name,
                contract.input_arity,
                contract.output_arity,
                contract.op_policy,
                contract.abstract_plan,
                tensors,
            )
            return output_contract.normalize(raw_outputs)

        input_shapes = extract_input_shapes(op_name=contract.name, tensors=tensors)
        contract.abstract_plan.validate_input_shapes(input_shapes)
        runtime_type_key = self._runtime_type_key(tensors)
        backend_profile = contract.abstract_plan.resolve_backend_profile(tensors)
        runner_cache_key = (
            runtime_type_key,
            backend_profile.execution_identity,
        )
        if execution_strategy.call_mode is _CallMode.SHAPE_FREE_SINGLE:
            runner = runner_cache.shape_free_single_runners.get(runner_cache_key)
            if runner is None:
                shape_free_context = execution_strategy.shape_free_context
                if shape_free_context is None:
                    raise RuntimeError(
                        "shape-free single mode requires runtime context"
                    )
                runtime_context = RuntimeSpecializationContext(
                    input_shapes=shape_free_context.input_shapes,
                    backend_profile=backend_profile,
                )
                runner = contract.abstract_plan.resolve_single_output_runner(
                    runtime_context,
                    tensors,
                )
                runner_cache.shape_free_single_runners.set(runner_cache_key, runner)
            return output_contract.normalize(runner(tensors))

        tuple_runner = runner_cache.shape_free_tuple_runners.get(runner_cache_key)
        if tuple_runner is None:
            shape_free_context = execution_strategy.shape_free_context
            if shape_free_context is None:
                raise RuntimeError("shape-free tuple mode requires runtime context")
            runtime_context = RuntimeSpecializationContext(
                input_shapes=shape_free_context.input_shapes,
                backend_profile=backend_profile,
            )
            tuple_runner = contract.abstract_plan.resolve_tuple_runner(
                runtime_context,
                tensors,
            )
            runner_cache.shape_free_tuple_runners.set(runner_cache_key, tuple_runner)
        raw_outputs = tuple_runner(tensors)
        return output_contract.normalize(raw_outputs)


_TENSOR_OP_FACTORY: TensorOpFactory[TensorOp] = TensorOpFactory(
    base_max_size=_BASE_OP_CACHE_MAX_SIZE,
    configured_max_size=_CONFIGURED_OP_CACHE_MAX_SIZE,
)
