from einf.diagnostics import ValidationError
from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.steps.einsum import (
    EinsumSymbolicStep,
    build_einsum_symbolic_program_from_equations,
    build_einsum_symbolic_program_from_sides,
)
from einf.steps.einsum.equation import build_contract_equation


def build_contract_symbolic_plan(
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for `contract`."""
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    _ = reducer_plan
    program = build_einsum_symbolic_program_from_sides(
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=explicit_sizes_items,
        allow_native_matmul=True,
    )
    if len(rhs) == 1:
        try:
            equation = build_contract_equation(
                input_axis_lists=lhs,
                output_axis_list=rhs[0],
            )
        except ValidationError:
            pass
        else:
            program = build_einsum_symbolic_program_from_equations(
                input_arity=len(lhs),
                output_arity=1,
                equations=(equation,),
                allow_native_matmul=True,
            )
    step = EinsumSymbolicStep(program=program)
    return SymbolicPlan(
        kind="contract",
        input_arity=len(lhs),
        output_arity=len(rhs),
        steps=(step,),
    )


__all__ = ["build_contract_symbolic_plan"]
