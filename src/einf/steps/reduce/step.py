from collections.abc import Callable
from dataclasses import dataclass

from einf.axis import AxisSide, AxisTerms, ScalarAxisTerms, term_size
from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    BackendProfile,
    get_backend_array_ops,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import build_runtime_execution_context
from einf.reduction.plan import infer_unary_reduced_terms
from einf.reduction.schema import STRING_REDUCERS, Reducer
from einf.signature import Signature
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep, SymbolicProgram
from einf.tensor_types import TensorLike

from ..base import AxisSideSymbolicStep
from .build import ReduceAxesResolver, build_reduce_compiled_program
from .runtime import ReducerRuntimeContext

_DIRECT_TORCH_REDUCER_METHODS: dict[str, str] = {
    "sum": "sum",
    "prod": "prod",
    "mean": "mean",
    "max": "amax",
    "min": "amin",
    "all": "all",
    "any": "any",
}

_DIRECT_NUMPY_REDUCER_METHODS: dict[str, str] = {
    "sum": "sum",
    "prod": "prod",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "all": "all",
    "any": "any",
}


@dataclass(frozen=True, slots=True)
class ReduceSymbolicProgram(SymbolicProgram):
    """Precompiled unary reduce program consumed by reduce runtime steps."""

    signature: Signature
    reducer: Reducer
    reduce_axes: AxisTerms
    is_default_reducer: bool


def build_reduce_symbolic_program(
    lhs: AxisSide,
    rhs: AxisSide,
    reducer: Reducer,
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


@dataclass(frozen=True, slots=True)
class ReduceRuntimeStep(RuntimeStep[ReduceSymbolicProgram]):
    """Runtime reduce step that executes one unary reduce primitive."""

    name: str
    input_arity: int
    output_arity: int
    program: ReduceSymbolicProgram
    explicit_sizes: dict[str, int]
    backend_profile: BackendProfile
    runtime_backend_ops: BackendArrayOps | None = None
    runtime_xp: ArrayNamespace | None = None
    compiled_unary_runner: Callable[[TensorLike], TensorLike] | None = None

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) == 1:
            return (self.run_unary(tensors[0]),)
        return self._run_general(tensors)

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one unary reduce runtime step."""
        compiled_unary_runner = self.compiled_unary_runner
        if compiled_unary_runner is not None:
            return compiled_unary_runner(tensor)

        outputs = self._run_general((tensor,))
        if len(outputs) != 1:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: reduce unary runtime produced invalid output arity",
                help="ensure reduce symbolic step remains unary (1->1)",
                related=("reduce runtime",),
                data={"operation": "reduce"},
            )
        return outputs[0]

    def _run_general(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        """Run one unary reduce runtime step with context normalization."""
        context = build_runtime_execution_context(
            signature=self.program.signature,
            tensors=tensors,
            explicit_sizes=self.explicit_sizes,
        )

        plan = build_reduce_compiled_program(
            tensor=tensors[0],
            lhs_terms=context.lhs_terms[0],
            expected_output_terms=context.rhs_terms[0],
            axis_sizes=context.axis_sizes,
            pack_sizes=context.pack_sizes,
            pack_ranks=context.pack_ranks,
            reduce_axes=self.program.reduce_axes,
            reducer=self.program.reducer,
            backend_profile=self.backend_profile,
        )

        reducer_runtime_context = ReducerRuntimeContext(
            xp=plan.xp,
            backend_ops=plan.backend_ops,
        )
        tensor = plan.compiled_reducer.apply(
            tensor=tensors[0],
            axes=plan.axes,
            context=reducer_runtime_context,
        )

        _validate_reduce_output_shape(
            tensor=tensor,
            terms=context.rhs_terms[0],
            axis_sizes=context.axis_sizes,
        )
        return (tensor,)


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
        runtime_backend_ops = get_backend_array_ops(backend_profile.backend_family)
        runtime_xp = backend_profile.namespace
        compiled_unary_runner = _build_static_reduce_unary_runner(
            program=self.program,
            runtime_backend_ops=runtime_backend_ops,
            runtime_xp=runtime_xp,
        )

        return ReduceRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=self.program,
            explicit_sizes=explicit_sizes,
            backend_profile=backend_profile,
            runtime_backend_ops=runtime_backend_ops,
            runtime_xp=runtime_xp,
            compiled_unary_runner=compiled_unary_runner,
        )

    def reducer_label(self) -> str:
        """Return one compact reducer label for plan rendering."""
        if self.program.is_default_reducer:
            return "sum(default)"

        reducer = self.program.reducer
        if isinstance(reducer, str):
            return reducer
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


def _build_static_reduce_unary_runner(
    *,
    program: ReduceSymbolicProgram,
    runtime_backend_ops: BackendArrayOps | None,
    runtime_xp: ArrayNamespace,
) -> Callable[[TensorLike], TensorLike] | None:
    """Build one static unary reduce runner when axis mapping is shape-invariant."""
    reducer = program.reducer
    if not isinstance(reducer, str) or reducer not in STRING_REDUCERS:
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

        def run_identity(tensor: TensorLike, /) -> TensorLike:
            return tensor

        return run_identity

    if runtime_backend_ops is not None:
        backend_family = runtime_backend_ops.backend_family
        if backend_family == "torch":
            method_name = _DIRECT_TORCH_REDUCER_METHODS.get(reducer)
            if isinstance(method_name, str):

                def run_torch_method(tensor: TensorLike, /) -> TensorLike:
                    reducer_method = getattr(tensor, method_name, None)
                    if not callable(reducer_method):
                        raise ValidationError(
                            code=ErrorCode.INCONSISTENT_DIMS,
                            message=(
                                "inconsistent dims: backend reducer "
                                f"{reducer!r} is unavailable"
                            ),
                            help=(
                                "choose a reducer available on the active backend namespace"
                            ),
                            related=("reduce reducer",),
                            data={},
                        )
                    try:
                        return reducer_method(dim=reduce_axes)
                    except Exception as error:
                        raise ValidationError(
                            code=ErrorCode.INCONSISTENT_DIMS,
                            message=(
                                f"inconsistent dims: backend reducer {reducer!r} failed: {error}"
                            ),
                            help=(
                                "ensure reducer domain is valid for the selected axes "
                                "(for example non-empty domain for max/min)"
                            ),
                            related=("reduce reducer",),
                            data={"reducer": reducer},
                        ) from error

                return run_torch_method
        if backend_family == "numpy":
            method_name = _DIRECT_NUMPY_REDUCER_METHODS.get(reducer)
            if isinstance(method_name, str):

                def run_numpy_method(tensor: TensorLike, /) -> TensorLike:
                    reducer_method = getattr(tensor, method_name, None)
                    if not callable(reducer_method):
                        raise ValidationError(
                            code=ErrorCode.INCONSISTENT_DIMS,
                            message=(
                                "inconsistent dims: backend reducer "
                                f"{reducer!r} is unavailable"
                            ),
                            help=(
                                "choose a reducer available on the active backend namespace"
                            ),
                            related=("reduce reducer",),
                            data={},
                        )
                    try:
                        return reducer_method(axis=reduce_axes)
                    except Exception as error:
                        raise ValidationError(
                            code=ErrorCode.INCONSISTENT_DIMS,
                            message=(
                                f"inconsistent dims: backend reducer {reducer!r} failed: {error}"
                            ),
                            help=(
                                "ensure reducer domain is valid for the selected axes "
                                "(for example non-empty domain for max/min)"
                            ),
                            related=("reduce reducer",),
                            data={"reducer": reducer},
                        ) from error

                return run_numpy_method

        reducer_fn = runtime_backend_ops.reducers.get(reducer)
        if reducer_fn is None:
            return None

        def run_with_backend_ops(tensor: TensorLike, /) -> TensorLike:
            try:
                return reducer_fn(tensor, reduce_axes)
            except Exception as error:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        f"inconsistent dims: backend reducer {reducer!r} failed: {error}"
                    ),
                    help=(
                        "ensure reducer domain is valid for the selected axes "
                        "(for example non-empty domain for max/min)"
                    ),
                    related=("reduce reducer",),
                    data={"reducer": reducer},
                ) from error

        return run_with_backend_ops

    reducer_candidate = getattr(runtime_xp, reducer, None)
    if not callable(reducer_candidate):
        return None

    def run_with_namespace(tensor: TensorLike, /) -> TensorLike:
        try:
            reduced = reducer_candidate(tensor, axis=reduce_axes)
        except Exception as error:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    f"inconsistent dims: backend reducer {reducer!r} failed: {error}"
                ),
                help=(
                    "ensure reducer domain is valid for the selected axes "
                    "(for example non-empty domain for max/min)"
                ),
                related=("reduce reducer",),
                data={"reducer": reducer},
            ) from error

        shape = getattr(reduced, "shape", None)
        if not isinstance(shape, tuple):
            try:
                coerced = runtime_xp.asarray(reduced)
            except Exception as error:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: reducer output must be tensor-like"
                    ),
                    help="return a tensor or scalar value from reducer",
                    related=("reduce reducer output",),
                    data={},
                ) from error
            return coerced
        return reduced

    return run_with_namespace


__all__ = [
    "build_reduce_symbolic_program",
    "ReduceSymbolicProgram",
    "ReduceRuntimeStep",
    "ReduceSymbolicStep",
]
