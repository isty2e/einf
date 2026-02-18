from einf.axis import AxisSide
from einf.diagnostics import ValidationError
from einf.ir import IRProgram, build_default_ir_program
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.signature import Signature
from einf.steps.einsum import (
    EinsumSymbolicStep,
    build_einsum_symbolic_program_from_equations,
    build_einsum_symbolic_program_from_sides,
)

from ..einop import build_einop_execution_plan
from .contract import build_contract_symbolic_plan
from .rearrange import build_rearrange_symbolic_plan
from .reduce import build_reduce_symbolic_plan
from .repeat import build_repeat_symbolic_plan


def _build_reduce_repeat_symbolic_plan(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
) -> SymbolicPlan | None:
    """Build one `reduce -> repeat` symbolic plan for unary signatures."""
    if len(lhs) != 1 or len(rhs) != 1:
        return None

    shared_terms = lhs[0] & rhs[0]
    reduced_rhs = AxisSide.from_spec((shared_terms,), side_name="rhs")
    repeated_lhs = AxisSide.from_spec((shared_terms,), side_name="lhs")
    reduce_plan = build_reduce_symbolic_plan(
        build_default_ir_program(
            op_name="reduce",
            lhs=lhs,
            rhs=reduced_rhs,
        ),
        explicit_sizes_items,
        None,
    )
    repeat_plan = build_repeat_symbolic_plan(
        build_default_ir_program(
            op_name="repeat",
            lhs=repeated_lhs,
            rhs=rhs,
        ),
        explicit_sizes_items,
        None,
    )
    return SymbolicPlan(
        kind="reduce_repeat",
        input_arity=len(lhs),
        output_arity=len(rhs),
        steps=(*reduce_plan.steps, *repeat_plan.steps),
    )


def build_einop_symbolic_plan(
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for `einop` from canonical sides."""
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    analysis_signature = Signature(inputs=lhs, outputs=rhs)
    try:
        execution_plan = build_einop_execution_plan(
            analysis_signature=analysis_signature,
            has_reducer_plan=reducer_plan is not None,
        )
    except ValidationError:
        if len(rhs) != 1:
            raise
        step = EinsumSymbolicStep(
            program=build_einsum_symbolic_program_from_sides(
                lhs=lhs,
                rhs=rhs,
                explicit_sizes_items=explicit_sizes_items,
                allow_native_matmul=True,
            )
        )
        return SymbolicPlan(
            kind="einsum",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(step,),
        )

    if execution_plan.kind == "rearrange":
        return build_rearrange_symbolic_plan(
            build_default_ir_program(
                op_name="rearrange",
                lhs=lhs,
                rhs=rhs,
            ),
            explicit_sizes_items,
            None,
        )

    if execution_plan.kind == "repeat":
        return build_repeat_symbolic_plan(
            build_default_ir_program(
                op_name="repeat",
                lhs=lhs,
                rhs=rhs,
            ),
            explicit_sizes_items,
            None,
        )

    if execution_plan.kind == "reduce":
        return build_reduce_symbolic_plan(
            build_default_ir_program(
                op_name="reduce",
                lhs=lhs,
                rhs=rhs,
            ),
            explicit_sizes_items,
            reducer_plan,
        )

    if execution_plan.kind == "reduce_repeat":
        reduce_repeat_plan = _build_reduce_repeat_symbolic_plan(
            lhs=lhs,
            rhs=rhs,
            explicit_sizes_items=explicit_sizes_items,
        )
        if reduce_repeat_plan is not None:
            return reduce_repeat_plan

    if execution_plan.kind == "contract":
        return build_contract_symbolic_plan(
            build_default_ir_program(
                op_name="contract",
                lhs=lhs,
                rhs=rhs,
            ),
            explicit_sizes_items,
            None,
        )

    if execution_plan.kind == "einsum":
        if len(execution_plan.equations) != 1:
            raise ValueError("einsum primitive lowering requires exactly one equation")
        step = EinsumSymbolicStep(
            program=build_einsum_symbolic_program_from_equations(
                input_arity=len(lhs),
                output_arity=1,
                equations=(execution_plan.equations[0],),
                allow_native_matmul=True,
            )
        )
        return SymbolicPlan(
            kind="einsum",
            input_arity=len(lhs),
            output_arity=1,
            steps=(step,),
        )

    if execution_plan.kind == "einsum_carrier_then_unary":
        if (
            execution_plan.intermediate is not None
            and len(execution_plan.equations) == 1
        ):
            carrier_step = EinsumSymbolicStep(
                program=build_einsum_symbolic_program_from_equations(
                    input_arity=len(lhs),
                    output_arity=1,
                    equations=(execution_plan.equations[0],),
                    allow_native_matmul=True,
                )
            )
            carrier_lhs = AxisSide.from_spec(
                (execution_plan.intermediate,),
                side_name="lhs",
            )
            tail_plan = build_einop_symbolic_plan(
                build_default_ir_program(
                    op_name="einop",
                    lhs=carrier_lhs,
                    rhs=rhs,
                ),
                explicit_sizes_items,
                None,
            )
            if tail_plan.input_arity != 1:
                raise ValueError("carrier tail lowering must be unary")
            return SymbolicPlan(
                kind="einsum_carrier_then_unary",
                input_arity=len(lhs),
                output_arity=len(rhs),
                steps=(carrier_step, *tail_plan.steps),
            )

    if execution_plan.kind == "einsum_chain_then_unary":
        if (
            execution_plan.intermediate is None
            or execution_plan.carrier_index is None
            or not execution_plan.equations
            or len(execution_plan.chain_order) != len(execution_plan.equations)
        ):
            raise ValueError(
                "invalid einsum chain execution plan for symbolic lowering"
            )
        chain_step = EinsumSymbolicStep(
            program=build_einsum_symbolic_program_from_equations(
                input_arity=len(lhs),
                output_arity=1,
                equations=execution_plan.equations,
                chain_order=execution_plan.chain_order,
                carrier_index=execution_plan.carrier_index,
                allow_native_matmul=True,
            )
        )
        carrier_lhs = AxisSide.from_spec(
            (execution_plan.intermediate,),
            side_name="lhs",
        )
        tail_plan = build_einop_symbolic_plan(
            build_default_ir_program(
                op_name="einop",
                lhs=carrier_lhs,
                rhs=rhs,
            ),
            explicit_sizes_items,
            None,
        )
        if tail_plan.input_arity != 1:
            raise ValueError("einsum chain tail lowering must be unary")
        return SymbolicPlan(
            kind="einsum_chain_then_unary",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(chain_step, *tail_plan.steps),
        )

    raise ValueError(f"unsupported einop execution plan kind: {execution_plan.kind}")


__all__ = [
    "build_einop_symbolic_plan",
]
