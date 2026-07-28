from dataclasses import dataclass
from typing import Literal

from einf.axis import AxisSide, AxisTerms, ScalarAxisTerms, term_size
from einf.backend import (
    BackendProfile,
    get_backend_array_ops,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.reduction.plan import infer_unary_reduced_terms
from einf.reduction.schema import CanonicalReducer, ReducerName
from einf.signature import Signature
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
    UnaryRuntimeProgram,
)
from einf.steps.context import build_runtime_execution_context
from einf.tensor_types import TensorLike

from ..base import AxisSideSymbolicStep
from .build import (
    ReduceAxesResolver,
    _has_reduce_namespace_methods,
    build_reduce_compiled_program,
)
from .runtime import (
    NamespaceReducer,
    ReducerRuntimeContext,
    resolve_namespace_reducer,
)

_DIRECT_TORCH_REDUCER_METHODS: dict[ReducerName, str] = {
    ReducerName.SUM: "sum",
    ReducerName.MEAN: "mean",
    ReducerName.MAX: "amax",
    ReducerName.MIN: "amin",
    ReducerName.ALL: "all",
    ReducerName.ANY: "any",
}

_DIRECT_NUMPY_REDUCER_METHODS: dict[ReducerName, str] = {
    ReducerName.SUM: "sum",
    ReducerName.PROD: "prod",
    ReducerName.MEAN: "mean",
    ReducerName.MAX: "max",
    ReducerName.MIN: "min",
    ReducerName.ALL: "all",
    ReducerName.ANY: "any",
}


@dataclass(frozen=True, slots=True)
class ReduceSymbolicProgram(SymbolicProgram):
    """Precompiled unary reduce program consumed by reduce runtime steps."""

    signature: Signature
    reducer: CanonicalReducer
    reduce_axes: AxisTerms
    is_default_reducer: bool


def build_reduce_symbolic_program(
    lhs: AxisSide,
    rhs: AxisSide,
    reducer: CanonicalReducer,
    reduce_axes: AxisTerms,
    is_default_reducer: bool,
) -> ReduceSymbolicProgram:
    """Build one precompiled unary reduce program from canonical sides."""
    if len(lhs) != 1 or len(rhs) != 1:
        raise ValueError("reduce symbolic step must be 1->1")

    signature = Signature(inputs=lhs, outputs=rhs)
    normalized_reduce_axes = AxisTerms.from_spec(reduce_axes)
    return ReduceSymbolicProgram(
        signature=signature,
        reducer=reducer,
        reduce_axes=normalized_reduce_axes,
        is_default_reducer=is_default_reducer,
    )


class ReduceRuntimeProgram(UnaryRuntimeProgram):
    """Base runtime program for one unary reduce execution strategy."""


@dataclass(frozen=True, slots=True)
class DirectMethodReduceRuntimeProgram(ReduceRuntimeProgram):
    """Shape-invariant unary reduce program bound to one tensor method."""

    reducer: ReducerName
    axes: tuple[int, ...]
    runtime_context: ReducerRuntimeContext
    direct_method_name: str
    direct_axis_keyword: Literal["axis", "dim"]

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one direct-method unary reduce program."""
        if not self.axes:
            return tensor

        try:
            if self.direct_axis_keyword == "dim":
                return getattr(tensor, self.direct_method_name)(dim=self.axes)
            return getattr(tensor, self.direct_method_name)(axis=self.axes)
        except Exception as error:
            raise self.runtime_context.string_reducer_error(
                reducer_name=self.reducer,
                error=error,
            ) from error


@dataclass(frozen=True, slots=True)
class NamespaceReduceRuntimeProgram(ReduceRuntimeProgram):
    """Shape-invariant unary reduce program bound to one namespace reducer."""

    reducer: ReducerName
    axes: tuple[int, ...]
    runtime_context: ReducerRuntimeContext
    reducer_fn: NamespaceReducer

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one namespace-bound unary reduce program."""
        if not self.axes:
            return tensor

        return self.runtime_context.apply_string_reducer(
            reducer_name=self.reducer,
            reducer_fn=self.reducer_fn,
            tensor=tensor,
            axes=self.axes,
        )


@dataclass(frozen=True, slots=True)
class DynamicReduceRuntimeProgram(ReduceRuntimeProgram):
    """Call-time unary reduce program that still depends on runtime context."""

    signature: Signature
    explicit_sizes: dict[str, int]
    reduce_axes: AxisTerms
    reducer: CanonicalReducer
    backend_profile: BackendProfile

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one dynamic unary reduce program."""
        context = build_runtime_execution_context(
            signature=self.signature,
            tensors=(tensor,),
            explicit_sizes=self.explicit_sizes,
        )

        plan = build_reduce_compiled_program(
            tensor=tensor,
            lhs_terms=context.lhs_terms[0],
            expected_output_terms=context.rhs_terms[0],
            axis_sizes=context.axis_sizes,
            pack_sizes=context.pack_sizes,
            pack_ranks=context.pack_ranks,
            reduce_axes=self.reduce_axes,
            reducer=self.reducer,
            backend_profile=self.backend_profile,
        )

        reducer_runtime_context = ReducerRuntimeContext(
            xp=plan.xp,
            backend_ops=plan.backend_ops,
        )
        output = plan.compiled_reducer.apply(
            tensor=tensor,
            axes=plan.axes,
            context=reducer_runtime_context,
        )

        _validate_reduce_output_shape(
            tensor=output,
            terms=context.rhs_terms[0],
            axis_sizes=context.axis_sizes,
        )
        return output


@dataclass(frozen=True, slots=True)
class ReduceRuntimeStep(RuntimeStep[ReduceRuntimeProgram]):
    """Runtime reduce step that executes one unary reduce primitive."""

    name: str
    input_arity: int
    output_arity: int
    program: ReduceRuntimeProgram

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=f"reduce arity mismatch: expected 1 input, got {len(tensors)}",
                help="reduce runtime step requires one tensor",
                related=("reduce runtime",),
                data={"operation": "reduce"},
            )
        return self.program(tensors)

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one unary reduce runtime step."""
        return self.program.run_unary(tensor)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReduceSymbolicStep(AxisSideSymbolicStep[ReduceSymbolicProgram]):
    """Unary symbolic reduce primitive step."""

    program: ReduceSymbolicProgram
    name: str = "reduce"

    def __post_init__(self) -> None:
        AxisSideSymbolicStep.__post_init__(self)
        if self.input_arity != 1 or self.output_arity != 1:
            raise ValueError("reduce symbolic step must be 1->1")
        if self.program.signature.inputs != self.lhs:
            raise ValueError("reduce program lhs does not match step lhs")
        if self.program.signature.outputs != self.rhs:
            raise ValueError("reduce program rhs does not match step rhs")

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        if self.program.is_default_reducer:
            infer_unary_reduced_terms(
                lhs=self.program.signature.inputs[0],
                rhs=self.program.signature.outputs[0],
                op_name="reduce",
            )
        explicit_sizes = self.program.signature.filter_explicit_sizes(
            dict(self.explicit_sizes_items)
        )
        backend_profile = context.backend_profile
        if backend_profile is None:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message=(
                    "backend dispatch unsupported input: reduce runtime requires "
                    "one resolved backend profile"
                ),
                help="execute through AbstractPlan/TensorOp call path to resolve backend profile",
                related=("backend dispatch",),
                data={"operation": "reduce"},
            )
        runtime_program: ReduceRuntimeProgram | None = None
        runtime_backend_ops = get_backend_array_ops(backend_profile.backend_family)
        runtime_xp_candidate = backend_profile.namespace
        if _has_reduce_namespace_methods(runtime_xp_candidate):
            runtime_context = ReducerRuntimeContext(
                xp=runtime_xp_candidate,
                backend_ops=runtime_backend_ops,
            )
            runtime_program = _build_shape_invariant_reduce_runtime_program(
                program=self.program,
                runtime_context=runtime_context,
                backend_family=backend_profile.backend_family,
            )
        if runtime_program is None:
            runtime_program = DynamicReduceRuntimeProgram(
                signature=self.program.signature,
                explicit_sizes=explicit_sizes,
                reduce_axes=self.program.reduce_axes,
                reducer=self.program.reducer,
                backend_profile=backend_profile,
            )

        return ReduceRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=runtime_program,
        )

    def reducer_label(self) -> str:
        """Return one compact reducer label for plan rendering."""
        if self.program.is_default_reducer:
            return "sum(default)"

        reducer = self.program.reducer
        if isinstance(reducer, ReducerName):
            return reducer.value
        return "callable"


def _validate_reduce_output_shape(
    *,
    tensor: TensorLike,
    terms: ScalarAxisTerms,
    axis_sizes: dict[str, int],
) -> None:
    """Validate reducer output shape against reduced-axis contract."""
    try:
        expected_shape = tuple(term_size(term, axis_sizes) for term in terms)
    except Exception as error:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=f"inconsistent dims: reduce output shape inference failed: {error}",
            help="ensure reduced axis expressions evaluate to non-negative integers",
            related=("reduce reducer output",),
            data={"operation": "reduce"},
        ) from error

    actual_shape = tuple(tensor.shape)
    if actual_shape == expected_shape:
        return

    raise ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: reducer output shape does not match reduced-axis contract",
        help="return tensors whose shape matches unreduced rhs terms",
        related=("reduce reducer output",),
        data={
            "operation": "reduce",
            "expected_rank": len(expected_shape),
            "actual_rank": len(actual_shape),
        },
    )


def _build_shape_invariant_reduce_runtime_program(
    *,
    program: ReduceSymbolicProgram,
    runtime_context: ReducerRuntimeContext,
    backend_family: str | None,
) -> ReduceRuntimeProgram | None:
    """Build one static unary reduce program when axis mapping is shape-invariant."""
    reducer = program.reducer
    if not isinstance(reducer, ReducerName):
        return None

    try:
        lhs_terms = ScalarAxisTerms.from_spec(program.signature.inputs[0])
        rhs_terms = ScalarAxisTerms.from_spec(program.signature.outputs[0])
        reduce_terms = ScalarAxisTerms.from_spec(program.reduce_axes)
    except TypeError:
        return None

    resolved = ReduceAxesResolver.resolve(
        current_terms=lhs_terms,
        reduce_terms=reduce_terms,
    )
    if resolved.output_terms != rhs_terms:
        return None
    reduce_axes = resolved.axes

    if not reduce_axes:
        reducer_fn = resolve_namespace_reducer(runtime_context.xp, reducer)
        if reducer_fn is None:
            return None
        return NamespaceReduceRuntimeProgram(
            reducer=reducer,
            axes=reduce_axes,
            runtime_context=runtime_context,
            reducer_fn=reducer_fn,
        )

    reducer_fn = resolve_namespace_reducer(runtime_context.xp, reducer)
    if reducer_fn is None:
        return None
    if backend_family == "torch":
        direct_method_name = _DIRECT_TORCH_REDUCER_METHODS.get(reducer)
        if isinstance(direct_method_name, str):
            return DirectMethodReduceRuntimeProgram(
                reducer=reducer,
                axes=reduce_axes,
                runtime_context=runtime_context,
                direct_method_name=direct_method_name,
                direct_axis_keyword="dim",
            )
    if backend_family == "numpy":
        direct_method_name = _DIRECT_NUMPY_REDUCER_METHODS.get(reducer)
        if isinstance(direct_method_name, str):
            return DirectMethodReduceRuntimeProgram(
                reducer=reducer,
                axes=reduce_axes,
                runtime_context=runtime_context,
                direct_method_name=direct_method_name,
                direct_axis_keyword="axis",
            )

    return NamespaceReduceRuntimeProgram(
        reducer=reducer,
        axes=reduce_axes,
        runtime_context=runtime_context,
        reducer_fn=reducer_fn,
    )


__all__ = [
    "ReduceRuntimeStep",
    "ReduceSymbolicProgram",
    "ReduceSymbolicStep",
    "build_reduce_symbolic_program",
]
