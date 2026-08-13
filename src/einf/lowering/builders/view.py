from dataclasses import replace
from itertools import permutations

from einf.axis import AxisInt, AxisSide, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.ir import IRProgram, LoweringSignature
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.steps.axis_slice import (
    AxisSliceSymbolicStep,
    build_axis_slice_symbolic_program,
    resolve_axis_slice_axis,
)
from einf.steps.concat import ConcatSymbolicStep
from einf.steps.permute import (
    AxisPermuteSymbolicStep,
    PermuteSymbolicStep,
    build_axis_permute_symbolic_program,
)
from einf.steps.reshape import ReshapeSymbolicStep

from .rearrange import build_rearrange_symbolic_plan


def _not_a_view_lowering_error() -> ValidationError:
    """Build one strict-view lowering failure diagnostic."""
    return ValidationError(
        code=ErrorCode.NOT_A_VIEW,
        message="not a view: mapping is not representable by strict view primitives",
        help=(
            "use affine, zero-copy view mappings representable via permute, "
            "axis_slice, and zero-copy reshape"
        ),
        related=("view lowering",),
        data={"operation": "view"},
    )


def _is_view_safe_rearrange_step(step: object) -> bool:
    """Return whether one rearrange-lowered step is valid for strict view lowering."""
    if isinstance(step, ConcatSymbolicStep):
        return False
    if isinstance(step, AxisSliceSymbolicStep):
        return step.program.split_axis >= 0
    if isinstance(step, ReshapeSymbolicStep):
        return True
    return isinstance(
        step,
        (
            PermuteSymbolicStep,
            AxisPermuteSymbolicStep,
        ),
    )


def _view_reshape_step(step: ReshapeSymbolicStep) -> ReshapeSymbolicStep:
    """Return reshape step configured for strict zero-copy view execution."""
    signature = step.program.signature
    lhs_terms = signature.inputs[0]
    rhs_terms = signature.outputs[0]
    lhs_unit_count = sum(
        1 for term in lhs_terms if isinstance(term, AxisInt) and term.value == 1
    )
    rhs_unit_count = sum(
        1 for term in rhs_terms if isinstance(term, AxisInt) and term.value == 1
    )
    reject_not_a_view = (
        len(rhs_terms) > len(lhs_terms) and rhs_unit_count > lhs_unit_count
    )
    return replace(
        step,
        program=replace(
            step.program,
            zero_copy_mode="require_zero_copy",
            reject_not_a_view=reject_not_a_view,
        ),
    )


def _build_view_safe_plan_from_rearrange(
    rearrange_plan: SymbolicPlan,
) -> SymbolicPlan | None:
    """Convert a rearrange-lowered plan into strict view-safe primitives."""
    if not rearrange_plan.steps:
        return SymbolicPlan(
            source=rearrange_plan.source,
            kind="view",
            steps=(),
        )

    converted_steps = []
    for step in rearrange_plan.steps:
        if not _is_view_safe_rearrange_step(step):
            return None
        if isinstance(step, ReshapeSymbolicStep):
            converted_steps.append(_view_reshape_step(step))
            continue
        if isinstance(step, AxisSliceSymbolicStep):
            converted_steps.append(
                replace(
                    step,
                    program=replace(step.program, strict_view=True),
                )
            )
            continue
        converted_steps.append(step)

    return SymbolicPlan(
        source=rearrange_plan.source,
        kind="view",
        steps=tuple(converted_steps),
    )


def _build_unresolved_axis_slice_view_plan(
    rearrange_plan: SymbolicPlan,
) -> SymbolicPlan | None:
    """Convert one unresolved axis_slice rearrange plan to strict runtime view checks."""
    if len(rearrange_plan.steps) != 1:
        return None
    step = rearrange_plan.steps[0]
    if not isinstance(step, AxisSliceSymbolicStep):
        return None
    return SymbolicPlan(
        source=rearrange_plan.source,
        kind="view",
        steps=(
            replace(
                step,
                program=replace(step.program, strict_view=True),
            ),
        ),
    )


def _build_axis_permute_then_axis_slice_view_plan(
    *,
    source: LoweringSignature,
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
) -> SymbolicPlan | None:
    """Build one unary `permute -> axis_slice` strict view plan when directly splittable."""
    if len(lhs) != 1 or len(rhs) < 2:
        return None
    input_terms = lhs[0]
    input_rank = len(input_terms)
    if input_rank <= 1:
        return None

    identity = tuple(range(input_rank))
    for permutation in permutations(range(input_rank)):
        if permutation == identity:
            continue
        permuted_lhs = AxisSide(
            (AxisTerms(tuple(input_terms[index] for index in permutation)),)
        )
        if resolve_axis_slice_axis(permuted_lhs, rhs) is None:
            continue
        axis_permute_step = AxisPermuteSymbolicStep(
            lhs=lhs,
            rhs=permuted_lhs,
            program=build_axis_permute_symbolic_program(
                lhs=lhs,
                rhs=permuted_lhs,
                explicit_sizes_items=explicit_sizes_items,
            ),
            explicit_sizes_items=explicit_sizes_items,
        )
        axis_slice_step = AxisSliceSymbolicStep(
            lhs=permuted_lhs,
            rhs=rhs,
            program=build_axis_slice_symbolic_program(
                permuted_lhs,
                rhs,
                strict_view=True,
            ),
            explicit_sizes_items=explicit_sizes_items,
        )
        return SymbolicPlan(
            source=source,
            kind="view",
            steps=(axis_permute_step, axis_slice_step),
        )
    return None


def build_view_symbolic_plan(
    ir_program: IRProgram,
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for ``view`` from strict primitives.

    Parameters
    ----------
    ir_program : IRProgram
        Source-bound view IR.
    reducer_plan : ReducerPlan or None
        Unused reducer configuration accepted by the shared builder contract.

    Returns
    -------
    SymbolicPlan
        Strict view plan carrying ``ir_program.source``.

    Raises
    ------
    ValidationError
        If no zero-copy lowering can represent the source.
    """
    _ = reducer_plan
    try:
        rearrange_plan = build_rearrange_symbolic_plan(
            ir_program,
            None,
        )
    except ValidationError as error:
        if error.code == ErrorCode.AXIS_EXPRESSION_TOO_COMPLEX.value:
            raise
        raise _not_a_view_lowering_error() from error

    view_safe_plan = _build_view_safe_plan_from_rearrange(rearrange_plan)
    if view_safe_plan is not None:
        return view_safe_plan
    permute_axis_slice_plan = _build_axis_permute_then_axis_slice_view_plan(
        source=ir_program.source,
        lhs=ir_program.lhs,
        rhs=ir_program.rhs,
        explicit_sizes_items=ir_program.explicit_sizes_items,
    )
    if permute_axis_slice_plan is not None:
        return permute_axis_slice_plan
    unresolved_axis_slice_plan = _build_unresolved_axis_slice_view_plan(rearrange_plan)
    if unresolved_axis_slice_plan is not None:
        return unresolved_axis_slice_plan
    raise _not_a_view_lowering_error()


__all__ = ["build_view_symbolic_plan"]
