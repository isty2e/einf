from einf.axis import AxisSide, AxisTerms
from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.steps.expand import ExpandSymbolicStep
from einf.steps.permute import PermuteSymbolicStep, build_permute_symbolic_program

from ..expand import build_expand_symbolic_program


def build_repeat_symbolic_plan(
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for `repeat`."""
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    _ = reducer_plan
    if len(lhs) != 1 or len(rhs) != 1:
        raise ValueError("repeat primitive lowering requires unary 1->1 signature")

    expand_program = build_expand_symbolic_program(lhs[0], rhs[0])
    compiled = expand_program.compiled
    if compiled is not None and compiled.has_non_identity_permutation:
        permute_step = PermuteSymbolicStep(
            program=build_permute_symbolic_program(compiled.permutation)
        )
        if not compiled.insert_axes and all(
            output_index is not None for output_index in compiled.output_to_input
        ):
            return SymbolicPlan(
                kind="repeat",
                input_arity=len(lhs),
                output_arity=len(rhs),
                steps=(permute_step,),
            )

        permuted_lhs_terms = AxisTerms(
            tuple(lhs[0][index] for index in compiled.permutation)
        )
        permuted_lhs = AxisSide.from_spec(permuted_lhs_terms, side_name="lhs")
        permuted_program = build_expand_symbolic_program(
            lhs_terms=permuted_lhs_terms,
            rhs_terms=rhs[0],
        )
        expand_step = ExpandSymbolicStep(
            lhs=permuted_lhs,
            rhs=rhs,
            explicit_sizes_items=explicit_sizes_items,
            program=permuted_program,
        )
        return SymbolicPlan(
            kind="repeat",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(permute_step, expand_step),
        )

    expand_step = ExpandSymbolicStep(
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=explicit_sizes_items,
        program=expand_program,
    )
    return SymbolicPlan(
        kind="repeat",
        input_arity=len(lhs),
        output_arity=len(rhs),
        steps=(expand_step,),
    )


__all__ = ["build_repeat_symbolic_plan"]
