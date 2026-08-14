from einf.axis import AxisSide, AxisTerms
from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.steps.expand import ExpandSymbolicStep, build_expand_symbolic_program
from einf.steps.permute import PermuteSymbolicStep, build_permute_symbolic_program


def build_repeat_symbolic_plan(
    ir_program: IRProgram,
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for ``repeat``.

    Parameters
    ----------
    ir_program : IRProgram
        Source-bound repeat IR.
    reducer_plan : ReducerPlan or None
        Unused reducer configuration accepted by the shared builder contract.

    Returns
    -------
    SymbolicPlan
        Repeat plan carrying ``ir_program.source``.

    Raises
    ------
    ValueError
        If the source is not unary.
    """
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    explicit_sizes_items = ir_program.explicit_sizes_items
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
                source=ir_program.source,
                kind="repeat",
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
            source=ir_program.source,
            kind="repeat",
            steps=(permute_step, expand_step),
        )

    expand_step = ExpandSymbolicStep(
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=explicit_sizes_items,
        program=expand_program,
    )
    return SymbolicPlan(
        source=ir_program.source,
        kind="repeat",
        steps=(expand_step,),
    )


__all__ = ["build_repeat_symbolic_plan"]
