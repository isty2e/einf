from dataclasses import dataclass

from einf.axis import (
    AxisSide,
    CanonicalScalarExpr,
    ScalarAxisTermBase,
    ScalarAxisTerms,
    term_size,
)
from einf.backend import BackendProfile
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import build_runtime_execution_context
from einf.signature import Signature
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
)
from einf.tensor_types import TensorLike

from ..base import AxisSideSymbolicStep
from .validation import validate_view_outputs


@dataclass(frozen=True, slots=True)
class AxisSliceSymbolicProgram(SymbolicProgram):
    """Precompiled axis-slice program consumed by axis_slice runtime steps."""

    signature: Signature
    split_axis: int
    strict_view: bool = False


def build_axis_slice_symbolic_program(
    lhs: AxisSide, rhs: AxisSide, *, strict_view: bool = False
) -> AxisSliceSymbolicProgram:
    """Build one precompiled axis-slice program from canonical sides."""
    if len(lhs) != 1 or len(rhs) < 2:
        raise ValueError("axis_slice symbolic step must be 1->N with N>=2")
    split_axis = resolve_axis_slice_axis(lhs, rhs)
    return AxisSliceSymbolicProgram(
        signature=Signature(inputs=lhs, outputs=rhs),
        split_axis=-1 if split_axis is None else split_axis,
        strict_view=strict_view,
    )


def resolve_axis_slice_axis(lhs: AxisSide, rhs: AxisSide) -> int | None:
    """Return slice axis index when signature is unary axis-slice, else None."""
    if len(lhs) != 1 or len(rhs) < 2:
        return None

    input_terms = lhs[0]
    input_rank = len(input_terms)
    if input_rank == 0:
        return None
    if any(len(output_terms) != input_rank for output_terms in rhs):
        return None

    split_axis = -1
    for axis in range(input_rank):
        axis_terms = tuple(output_terms[axis] for output_terms in rhs)
        if all(term == input_terms[axis] for term in axis_terms):
            continue
        if split_axis != -1:
            return None
        split_axis = axis

    if split_axis == -1:
        return None

    for axis in range(input_rank):
        if axis == split_axis:
            continue
        if any(output_terms[axis] != input_terms[axis] for output_terms in rhs):
            return None

    try:
        input_split_term = ScalarAxisTermBase.coerce(input_terms[split_axis])
        output_split_terms = tuple(
            ScalarAxisTermBase.coerce(output_terms[split_axis]) for output_terms in rhs
        )
    except TypeError:
        return None

    input_expr = CanonicalScalarExpr.from_term(input_split_term)
    output_expr = CanonicalScalarExpr.sum_terms(output_split_terms)
    if output_expr != input_expr:
        return None
    return split_axis


def _try_run_direct_axis_slice(
    *,
    tensor: TensorLike,
    rhs_terms: tuple[ScalarAxisTerms, ...],
    axis_sizes: dict[str, int],
    split_axis: int,
) -> tuple[TensorLike, ...] | None:
    """Try direct slicing for one prevalidated axis-slice axis."""
    input_rank = len(tensor.shape)
    if any(len(output_terms) != input_rank for output_terms in rhs_terms):
        return None

    try:
        split_sizes = tuple(
            term_size(
                ScalarAxisTermBase.coerce(output_terms[split_axis]),
                axis_sizes,
            )
            for output_terms in rhs_terms
        )
    except Exception:
        return None

    if any(size < 0 for size in split_sizes):
        return None
    if sum(split_sizes) != tensor.shape[split_axis]:
        return None

    prefix = (slice(None),) * split_axis
    suffix = (slice(None),) * (input_rank - split_axis - 1)
    outputs: list[TensorLike] = []
    offset = 0
    for size in split_sizes:
        next_offset = offset + size
        index = prefix + (slice(offset, next_offset),) + suffix
        outputs.append(tensor[index])
        offset = next_offset
    return tuple(outputs)


def _run_direct_axis_slice_by_sizes(
    *,
    tensor: TensorLike,
    split_axis: int,
    split_sizes: tuple[int, ...],
) -> tuple[TensorLike, ...] | None:
    """Run direct slicing from precomputed split sizes."""
    input_rank = len(tensor.shape)
    if split_axis < 0 or split_axis >= input_rank:
        return None
    if any(size < 0 for size in split_sizes):
        return None
    if sum(split_sizes) != tensor.shape[split_axis]:
        return None

    prefix = (slice(None),) * split_axis
    suffix = (slice(None),) * (input_rank - split_axis - 1)
    outputs: list[TensorLike] = []
    offset = 0
    for size in split_sizes:
        next_offset = offset + size
        index = prefix + (slice(offset, next_offset),) + suffix
        outputs.append(tensor[index])
        offset = next_offset
    return tuple(outputs)


@dataclass(frozen=True, slots=True)
class AxisSliceRuntimeStep(RuntimeStep[AxisSliceSymbolicProgram]):
    """Runtime axis-slice step with prevalidated direct slicing."""

    name: str
    input_arity: int
    output_arity: int
    program: AxisSliceSymbolicProgram
    explicit_sizes: dict[str, int]
    backend_profile: BackendProfile | None = None
    precomputed_split_sizes: tuple[int, ...] | None = None

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=f"axis_slice arity mismatch: expected 1 input, got {len(tensors)}",
                help="axis_slice runtime expects one input tensor",
                related=("axis_slice runtime",),
                data={"operation": "axis_slice", "expected": 1, "got": len(tensors)},
            )
        tensor = tensors[0]
        split_axis = self.program.split_axis
        direct_outputs: tuple[TensorLike, ...] | None = None
        if split_axis >= 0:
            precomputed_split_sizes = self.precomputed_split_sizes
            if precomputed_split_sizes is not None:
                direct_outputs = _run_direct_axis_slice_by_sizes(
                    tensor=tensor,
                    split_axis=split_axis,
                    split_sizes=precomputed_split_sizes,
                )
            else:
                context = build_runtime_execution_context(
                    signature=self.program.signature,
                    tensors=tensors,
                    explicit_sizes=self.explicit_sizes,
                )
                direct_outputs = _try_run_direct_axis_slice(
                    tensor=tensor,
                    rhs_terms=context.rhs_terms,
                    axis_sizes=context.axis_sizes,
                    split_axis=split_axis,
                )
            if direct_outputs is not None:
                if self.program.strict_view:
                    backend_profile = self.backend_profile
                    if backend_profile is None:
                        raise ValidationError(
                            code=ErrorCode.NOT_A_VIEW,
                            message=(
                                "not a view: strict view validation requires "
                                "backend profile"
                            ),
                            help=(
                                "run view execution through TensorOp call path "
                                "to resolve backend profile"
                            ),
                            related=("view affine mapping",),
                            data={"operation": "view"},
                        )
                    validate_view_outputs(
                        input_tensor=tensor,
                        outputs=direct_outputs,
                        profile=backend_profile,
                    )
                return direct_outputs
            if self.program.strict_view:
                raise ValidationError(
                    code=ErrorCode.NOT_A_VIEW,
                    message=(
                        "not a view: axis_slice mapping is not representable "
                        "as strict zero-copy split"
                    ),
                    help="restrict view split mappings to one resolvable non-overlapping partition",
                    related=("view affine mapping", "axis_slice"),
                    data={"operation": "view"},
                )
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: axis_slice direct execution failed to resolve "
                    "one valid slicing partition"
                ),
                help="ensure axis_slice lowering emits one resolvable split axis partition",
                related=("axis_slice runtime",),
                data={"operation": "axis_slice"},
            )
        if self.program.strict_view:
            raise ValidationError(
                code=ErrorCode.NOT_A_VIEW,
                message=(
                    "not a view: axis_slice mapping is not one strict non-overlapping partition"
                ),
                help="restrict view split mappings to one resolvable non-overlapping partition",
                related=("view affine mapping", "axis_slice"),
                data={"operation": "view"},
            )
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message=(
                "ambiguous dims: axis_slice lowering has no unique split-axis partition"
            ),
            help="add axis names or with_sizes constraints to make split partition unique",
            related=("axis_slice runtime",),
            data={"operation": "axis_slice"},
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AxisSliceSymbolicStep(AxisSideSymbolicStep[AxisSliceSymbolicProgram]):
    """Primitive axis_slice symbolic step (1->N)."""

    program: AxisSliceSymbolicProgram
    name: str = "axis_slice"

    def __post_init__(self) -> None:
        AxisSideSymbolicStep.__post_init__(self)
        if self.input_arity != 1:
            raise ValueError("axis_slice symbolic step must be 1->N")
        if self.output_arity < 2:
            raise ValueError(
                "axis_slice symbolic step must produce at least two outputs"
            )
        if self.program.signature.inputs != self.lhs:
            raise ValueError("axis_slice program lhs does not match step lhs")
        if self.program.signature.outputs != self.rhs:
            raise ValueError("axis_slice program rhs does not match step rhs")

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        _ = context
        explicit_sizes = self.program.signature.filter_explicit_sizes(
            dict(self.explicit_sizes_items)
        )
        precomputed_split_sizes: tuple[int, ...] | None = None
        split_axis = self.program.split_axis
        if split_axis >= 0:
            try:
                split_terms = tuple(
                    ScalarAxisTermBase.coerce(output_terms[split_axis])
                    for output_terms in self.rhs
                )
                resolved_sizes = tuple(
                    term_size(term, explicit_sizes) for term in split_terms
                )
                if all(size >= 0 for size in resolved_sizes):
                    precomputed_split_sizes = resolved_sizes
            except Exception:
                precomputed_split_sizes = None
        return AxisSliceRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=self.program,
            explicit_sizes=explicit_sizes,
            backend_profile=context.backend_profile,
            precomputed_split_sizes=precomputed_split_sizes,
        )


__all__ = [
    "AxisSliceRuntimeStep",
    "AxisSliceSymbolicStep",
    "AxisSliceSymbolicProgram",
    "build_axis_slice_symbolic_program",
    "resolve_axis_slice_axis",
]
