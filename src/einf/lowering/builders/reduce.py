from einf.axis import AxisSide, AxisTerms
from einf.ir import IRProgram, build_default_ir_program
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPhase, ReducerPlan
from einf.steps.base import SymbolicStep
from einf.steps.reduce import ReduceSymbolicStep, build_reduce_symbolic_program

from .rearrange import build_rearrange_symbolic_plan


def _build_reduce_steps(
    *,
    lhs: AxisSide,
    reducer_plan: ReducerPlan,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    is_default_reducer: bool,
) -> tuple[list[SymbolicStep], AxisSide]:
    """Build chained unary reduce steps, one primitive step per reducer phase."""
    current_terms = lhs[0]
    steps: list[SymbolicStep] = []
    for phase_index, phase in enumerate(reducer_plan):
        phase_axes = AxisTerms.from_spec(phase.axes)
        next_terms = current_terms - phase_axes
        phase_lhs = AxisSide((current_terms,))
        phase_rhs = AxisSide((next_terms,))
        steps.append(
            ReduceSymbolicStep(
                lhs=phase_lhs,
                rhs=phase_rhs,
                program=build_reduce_symbolic_program(
                    lhs=phase_lhs,
                    rhs=phase_rhs,
                    reducer=phase.reducer,
                    reduce_axes=phase_axes,
                    is_default_reducer=is_default_reducer and phase_index == 0,
                ),
                explicit_sizes_items=explicit_sizes_items,
            )
        )
        current_terms = next_terms
    return steps, AxisSide((current_terms,))


def build_reduce_symbolic_plan(
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for `reduce`."""
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    if len(lhs) != 1 or len(rhs) != 1:
        raise ValueError("reduce symbolic plan requires unary lhs/rhs")

    resolved_reducer_plan = reducer_plan
    is_default_reducer = False
    if resolved_reducer_plan is None:
        is_default_reducer = True
        resolved_reducer_plan = (
            ReducerPhase(
                axes=lhs[0] - rhs[0],
                reducer="sum",
            ),
        )

    steps, reduce_rhs = _build_reduce_steps(
        lhs=lhs,
        reducer_plan=resolved_reducer_plan,
        explicit_sizes_items=explicit_sizes_items,
        is_default_reducer=is_default_reducer,
    )

    if reduce_rhs != rhs:
        rearrange_ir_program = build_default_ir_program(
            op_name="rearrange",
            lhs=reduce_rhs,
            rhs=rhs,
        )
        rearrange_plan = build_rearrange_symbolic_plan(
            rearrange_ir_program,
            explicit_sizes_items,
            None,
        )
        steps.extend(rearrange_plan.steps)

    return SymbolicPlan(
        kind="reduce",
        input_arity=len(lhs),
        output_arity=len(rhs),
        steps=tuple(steps),
    )


__all__ = ["build_reduce_symbolic_plan"]
