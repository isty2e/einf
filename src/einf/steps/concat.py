from dataclasses import dataclass

from einf.axis import (
    AxisSide,
    CanonicalScalarExpr,
    ScalarAxisTermBase,
)
from einf.backend import ArrayNamespace, BackendArrayOps
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import PlanSelectionContext
from einf.plans.scoring import numel_from_shape
from einf.signature import Signature
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
    SymbolicStepScore,
)
from einf.tensor_types import TensorLike

from .base import AxisSideSymbolicStep
from .runtime import bind_runtime_backend, coerce_step_outputs

_CONCAT_REQUIRED_METHODS = ("concat",)


@dataclass(frozen=True, slots=True)
class ConcatSymbolicProgram(SymbolicProgram):
    """Precompiled concat program consumed by concat runtime steps."""

    signature: Signature
    concat_axis: int


def build_concat_symbolic_program(
    lhs: AxisSide, rhs: AxisSide
) -> ConcatSymbolicProgram:
    """Build one precompiled concat program from canonical sides."""
    if len(lhs) < 2 or len(rhs) != 1:
        raise ValueError("concat symbolic step must be N->1 with N>=2")
    concat_axis = resolve_concat_axis(lhs, rhs)
    if concat_axis is None:
        raise ValueError("concat symbolic step requires one resolvable concat axis")
    return ConcatSymbolicProgram(
        signature=Signature(inputs=lhs, outputs=rhs),
        concat_axis=concat_axis,
    )


def resolve_concat_axis(lhs: AxisSide, rhs: AxisSide) -> int | None:
    """Return concat axis index when signature is concat, else None."""
    if len(lhs) < 2 or len(rhs) != 1:
        return None

    output_terms = rhs[0]
    output_rank = len(output_terms)
    if output_rank == 0:
        return None
    if any(len(input_terms) != output_rank for input_terms in lhs):
        return None

    concat_axis = -1
    for axis in range(output_rank):
        column_terms = tuple(input_terms[axis] for input_terms in lhs)
        if all(term == output_terms[axis] for term in column_terms):
            continue
        if concat_axis != -1:
            return None
        concat_axis = axis

    if concat_axis == -1:
        return None

    for axis in range(output_rank):
        if axis == concat_axis:
            continue
        if any(input_terms[axis] != output_terms[axis] for input_terms in lhs):
            return None

    try:
        output_concat_term = ScalarAxisTermBase.coerce(output_terms[concat_axis])
        input_concat_terms = tuple(
            ScalarAxisTermBase.coerce(input_terms[concat_axis]) for input_terms in lhs
        )
    except TypeError:
        return None
    output_expr = CanonicalScalarExpr.from_term(output_concat_term)
    input_expr = CanonicalScalarExpr.sum_terms(input_concat_terms)
    if input_expr == output_expr:
        return concat_axis

    return None


@dataclass(frozen=True, slots=True)
class ConcatRuntimeStep(RuntimeStep[ConcatSymbolicProgram]):
    """Runtime concat step with backend-native concat execution."""

    name: str
    input_arity: int
    output_arity: int
    program: ConcatSymbolicProgram
    runtime_backend_ops: BackendArrayOps | None = None
    runtime_xp: ArrayNamespace | None = None

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != self.input_arity:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=(
                    "concat arity mismatch: "
                    f"expected {self.input_arity} inputs, got {len(tensors)}"
                ),
                help="pass exactly the tensors required by this concat step",
                related=("concat runtime",),
                data={"operation": "concat"},
            )
        concat_axis = self.program.concat_axis
        runtime_backend_ops = self.runtime_backend_ops
        if runtime_backend_ops is not None:
            try:
                output = runtime_backend_ops.concat(list(tensors), concat_axis)
            except Exception as error:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: concat runtime failed during backend concat "
                        f"execution: {error}"
                    ),
                    help="ensure non-concat axes are shape-compatible across concat inputs",
                    related=("concat runtime",),
                    data={"operation": "concat"},
                ) from error
            return coerce_step_outputs(output)

        runtime_xp = self.runtime_xp
        if runtime_xp is None:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message=(
                    "backend dispatch unsupported input: "
                    "concat runtime requires one resolved backend namespace/profile"
                ),
                help="execute through AbstractPlan/TensorOp call path to resolve backend profile",
                related=("backend dispatch",),
                data={"operation": "concat"},
            )
        try:
            output = runtime_xp.concat(list(tensors), axis=concat_axis)
        except Exception as error:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: concat runtime failed during backend concat "
                    f"execution: {error}"
                ),
                help="ensure non-concat axes are shape-compatible across concat inputs",
                related=("concat runtime",),
                data={"operation": "concat"},
            ) from error
        return coerce_step_outputs(output)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConcatSymbolicStep(AxisSideSymbolicStep[ConcatSymbolicProgram]):
    """Primitive concat symbolic step (N->1)."""

    program: ConcatSymbolicProgram
    name: str = "concat"

    def __post_init__(self) -> None:
        AxisSideSymbolicStep.__post_init__(self)
        if self.input_arity < 2:
            raise ValueError("concat symbolic step must take at least two inputs")
        if self.output_arity != 1:
            raise ValueError("concat symbolic step must be N->1")
        if self.program.signature.inputs != self.lhs:
            raise ValueError("concat program lhs does not match step lhs")
        if self.program.signature.outputs != self.rhs:
            raise ValueError("concat program rhs does not match step rhs")
        if self.program.concat_axis < 0:
            raise ValueError("concat symbolic step requires a non-negative concat axis")

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        backend_binding = bind_runtime_backend(
            context,
            required_namespace_methods=_CONCAT_REQUIRED_METHODS,
            bind_namespace_when_backend_ops_available=False,
        )
        return ConcatRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=self.program,
            runtime_backend_ops=backend_binding.backend_ops,
            runtime_xp=backend_binding.xp,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        materialize_numel = 0
        for shape in context.input_shapes:
            tensor_numel = numel_from_shape(shape)
            if tensor_numel is None:
                materialize_numel = 0
                break
            materialize_numel += tensor_numel
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=materialize_numel,
            allocation_count=1,
            kernel_count=1,
        )


__all__ = [
    "ConcatRuntimeStep",
    "ConcatSymbolicStep",
    "ConcatSymbolicProgram",
    "build_concat_symbolic_program",
    "resolve_concat_axis",
]
