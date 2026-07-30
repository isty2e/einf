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
from ..lowering import DefaultLoweringProgram
from ..output_normalization import RuntimeOutputContract
from ..plans.abstract import AbstractPlan, RuntimeSpecializationContext
from ..plans.cache import RunnerCache
from ..plans.render import PlanDict, build_plan_dict, render_plan_text
from ..reduction.schema import Reducer, ReducerCallable, ReducerPlan
from ..signature import Signature
from ..tensor_types import TensorLike
from .cache import (
    BaseOpCacheKey,
    ConfiguredOpCacheKey,
    TensorOpFactory,
    reducer_plan_to_cache_key,
)
from .definition import TensorOpDefinition
from .execution import execute_tensor_op_call, extract_input_shapes
from .kind import OperationKind

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


def _configured_cache_key(definition: TensorOpDefinition) -> ConfiguredOpCacheKey:
    """Project one semantic definition into its runtime cache identity."""
    return ConfiguredOpCacheKey(
        base=BaseOpCacheKey(
            kind=definition.kind,
            lhs=definition.lhs,
            rhs=definition.rhs,
        ),
        sizes_items=definition.sizes_items,
        reducer_plan_key=reducer_plan_to_cache_key(definition.reducer_plan),
    )


@dataclass(frozen=True, slots=True)
class TensorOpExecutionStrategy:
    """Stable execution strategy derived from one planned TensorOp."""

    shape_free_context: RuntimeSpecializationContext | None
    call_mode: _CallMode

    @classmethod
    def from_plan(
        cls,
        *,
        abstract_plan: AbstractPlan,
        input_arity: int,
        output_arity: int,
    ) -> Self:
        """Build one execution strategy from a realized abstract plan."""
        shape_free_context: RuntimeSpecializationContext | None = None
        if not abstract_plan.specialization_depends_on_input_shapes(input_arity):
            shape_free_context = RuntimeSpecializationContext(
                input_shapes=tuple(() for _ in range(input_arity)),
                backend_profile=None,
            )
        if shape_free_context is None:
            call_mode = _CallMode.GENERAL
        elif output_arity == 1:
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
    """Executable transform backed by canonical semantics and runtime state."""

    _definition: TensorOpDefinition
    _abstract_plan: AbstractPlan = field(init=False, repr=False, compare=False)
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
        """Realize one definition as an abstract plan with fresh runtime state."""
        definition = self._definition
        abstract_plan = AbstractPlan(
            op_name=definition.name,
            lhs=definition.lhs,
            rhs=definition.rhs,
            explicit_sizes_items=definition.sizes_items,
            lowering=_DEFAULT_LOWERING_PROGRAM.with_reducer_plan(
                definition.reducer_plan
            ),
        )
        object.__setattr__(self, "_abstract_plan", abstract_plan)
        object.__setattr__(
            self,
            "_execution_strategy",
            TensorOpExecutionStrategy.from_plan(
                abstract_plan=abstract_plan,
                input_arity=definition.input_arity,
                output_arity=definition.output_arity,
            ),
        )
        object.__setattr__(self, "_runner_cache", TensorOpRunnerCache())
        object.__setattr__(
            self,
            "_runtime_output_contract",
            RuntimeOutputContract(
                op_name=definition.name,
                expected_output_arity=definition.output_arity,
            ),
        )

    @classmethod
    def from_base_spec(
        cls,
        *,
        kind: OperationKind,
        lhs: AxisSide,
        rhs: AxisSide,
    ):
        """Return cached base TensorOp for one normalized constructor spec."""
        cache_key = BaseOpCacheKey(
            kind=kind,
            lhs=lhs,
            rhs=rhs,
        )
        return _TENSOR_OP_FACTORY.get_base(
            key=cache_key,
            builder=lambda: cls(
                _definition=TensorOpDefinition(
                    kind=cache_key.kind,
                    lhs=cache_key.lhs,
                    rhs=cache_key.rhs,
                )
            ),
        )

    @property
    def name(self) -> str:
        """Public operation name."""
        return self._definition.name

    @property
    def lhs(self) -> AxisSide:
        """Canonical normalized left-hand input signature."""
        return self._definition.lhs

    @property
    def rhs(self) -> AxisSide:
        """Canonical normalized right-hand output signature."""
        return self._definition.rhs

    @property
    def supports_reducer(self) -> bool:
        """Whether this operation accepts `.reduce_by(...)` customization."""
        return self._definition.supports_reducer

    @property
    def reducer_plan(self) -> ReducerPlan | None:
        """Configured reducer plan, when applicable."""
        return self._definition.reducer_plan

    @property
    def signature(self) -> Signature:
        """Derived normalized signature view of current lhs/rhs."""
        return self._definition.signature

    @property
    def abstract_plan(self) -> AbstractPlan:
        """Canonical abstract execution plan for this TensorOp."""
        return self._abstract_plan

    @property
    def sizes(self) -> dict[str, int]:
        """Return explicit size bindings as a detached mapping copy."""
        return self._definition.sizes()

    @property
    def sizes_items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical immutable explicit size bindings."""
        return self._definition.sizes_items

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
        definition = self._definition.with_sizes(**sizes)
        if definition is self._definition:
            return self

        return _TENSOR_OP_FACTORY.get_configured(
            key=_configured_cache_key(definition),
            builder=lambda: TensorOp(_definition=definition),
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
        definition = self._definition.reduce_by(reducer, *phases)
        if definition is self._definition:
            return self

        return _TENSOR_OP_FACTORY.get_configured(
            key=_configured_cache_key(definition),
            builder=lambda: TensorOp(_definition=definition),
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
        definition = self._definition
        abstract_plan = self._abstract_plan
        execution_strategy = self._execution_strategy
        runner_cache = self._runner_cache
        output_contract = self._runtime_output_contract
        if (
            len(tensors) != definition.input_arity
            or execution_strategy.call_mode is _CallMode.GENERAL
        ):
            raw_outputs = execute_tensor_op_call(
                definition.name,
                definition.input_arity,
                definition.output_arity,
                definition.op_policy,
                abstract_plan,
                tensors,
            )
            return output_contract.normalize(raw_outputs)

        input_shapes = extract_input_shapes(op_name=definition.name, tensors=tensors)
        abstract_plan.validate_input_shapes(input_shapes)
        runtime_type_key = self._runtime_type_key(tensors)
        backend_profile = abstract_plan.resolve_backend_profile(tensors)
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
                runner = abstract_plan.resolve_single_output_runner(
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
            tuple_runner = abstract_plan.resolve_tuple_runner(
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
