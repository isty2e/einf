from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan

from .builders import (
    build_contract_symbolic_plan,
    build_einop_symbolic_plan,
    build_rearrange_symbolic_plan,
    build_reduce_symbolic_plan,
    build_repeat_symbolic_plan,
    build_view_symbolic_plan,
)

_IR_BUILDERS = {
    "view": build_view_symbolic_plan,
    "reduce": build_reduce_symbolic_plan,
    "contract": build_contract_symbolic_plan,
    "repeat": build_repeat_symbolic_plan,
    "rearrange": build_rearrange_symbolic_plan,
    "einop": build_einop_symbolic_plan,
}


def build_symbolic_candidates_from_ir(
    *,
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> tuple[SymbolicPlan, ...]:
    """Compile one canonical IR program into ordered symbolic-plan candidates."""
    builder = _IR_BUILDERS.get(ir_program.op_name)
    if builder is None:
        return ()
    return (builder(ir_program, explicit_sizes_items, reducer_plan),)


__all__ = ["build_symbolic_candidates_from_ir"]
