from collections.abc import Callable, Mapping
from types import MappingProxyType

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

_SymbolicPlanBuilder = Callable[[IRProgram, ReducerPlan | None], SymbolicPlan]

_IR_BUILDERS: Mapping[str, _SymbolicPlanBuilder] = MappingProxyType(
    {
        "view": build_view_symbolic_plan,
        "reduce": build_reduce_symbolic_plan,
        "contract": build_contract_symbolic_plan,
        "repeat": build_repeat_symbolic_plan,
        "rearrange": build_rearrange_symbolic_plan,
        "einop": build_einop_symbolic_plan,
    }
)


def build_symbolic_candidates_from_ir(
    *,
    ir_program: IRProgram,
    reducer_plan: ReducerPlan | None,
) -> tuple[SymbolicPlan, ...]:
    """Compile one IR program into ordered symbolic candidates.

    Parameters
    ----------
    ir_program : IRProgram
        Source-bound lowering IR.
    reducer_plan : ReducerPlan or None
        Optional reducer policy used by supporting operations.

    Returns
    -------
    tuple[SymbolicPlan, ...]
        Ordered candidates carrying ``ir_program.source``.
    """
    builder = _IR_BUILDERS.get(ir_program.op_name)
    if builder is None:
        return ()
    return (builder(ir_program, reducer_plan),)


__all__ = ["build_symbolic_candidates_from_ir"]
