from einf.axis import AxisSide, AxisTermBase, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.ir import IRProgram, build_default_ir_program
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.plan import infer_unary_reduced_terms
from einf.reduction.schema import ReducerPhase, ReducerPlan
from einf.signature import Signature
from einf.steps.base import StepProgram, SymbolicStep
from einf.steps.einsum import (
    EinsumSymbolicStep,
    build_einsum_symbolic_program_from_equations,
    build_einsum_symbolic_program_from_sides,
)
from einf.steps.tensor_map import TensorMapSymbolicProgram, TensorMapSymbolicStep

from ..einop import EinopLoweringPlan, build_einop_execution_plan
from ..einop.equation import build_einop_equations
from ..einop.layout import EinopLayoutNormalization
from .contract import build_contract_symbolic_plan
from .rearrange import build_rearrange_symbolic_plan
from .reduce import build_reduce_symbolic_plan
from .repeat import build_repeat_symbolic_plan


def _build_direct_einsum_symbolic_plan(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    equations: tuple[str, ...],
) -> SymbolicPlan:
    """Build independent direct einsum outputs from one shared input tuple."""
    if len(equations) != len(rhs):
        raise ValueError(
            "direct einsum lowering requires one equation per output tensor"
        )
    step = EinsumSymbolicStep(
        program=build_einsum_symbolic_program_from_equations(
            input_arity=len(lhs),
            output_arity=len(rhs),
            equations=equations,
            allow_native_matmul=True,
        )
    )
    return SymbolicPlan(
        kind="einsum",
        input_arity=len(lhs),
        output_arity=len(rhs),
        steps=(step,),
    )


def _build_layout_map_steps(
    *,
    sources: AxisSide,
    targets: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
) -> tuple[SymbolicStep[StepProgram], ...]:
    """Build independent unary layout transforms without changing tensor arity."""
    if len(sources) != len(targets):
        raise ValueError("einop layout map cannot change tensor arity")

    chains: list[tuple[SymbolicStep[StepProgram], ...]] = []
    for source, target in zip(sources, targets, strict=True):
        if source == target:
            chains.append(())
            continue
        plan = build_rearrange_symbolic_plan(
            build_default_ir_program(
                op_name="rearrange",
                lhs=AxisSide.from_spec((source,), side_name="lhs"),
                rhs=AxisSide.from_spec((target,), side_name="rhs"),
            ),
            explicit_sizes_items,
            None,
        )
        if plan.input_arity != 1 or plan.output_arity != 1:
            raise ValueError("einop layout transform must lower to a unary plan")
        chains.append(plan.steps)

    normalized_chains = tuple(chains)
    if not any(normalized_chains):
        return ()
    if any(
        step.specialization_depends_on_input_shapes()
        for chain in normalized_chains
        for step in chain
    ):
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message=(
                "ambiguous dims: einop layout transform has no unique axis mapping"
            ),
            help="rename repeated logical axes to make the layout mapping unique",
            related=("einop layout normalization",),
            data={"operation": "einop"},
        )
    if len(normalized_chains) == 1:
        return normalized_chains[0]
    return (
        TensorMapSymbolicStep(
            program=TensorMapSymbolicProgram(chains=normalized_chains)
        ),
    )


def _normalize_layout_reducer_plan(
    *,
    normalization: EinopLayoutNormalization,
    reducer_plan: ReducerPlan | None,
) -> ReducerPlan | None:
    """Project reducer phases from requested composites onto logical axes."""
    if reducer_plan is None:
        return None

    logical = normalization.logical
    if len(logical.inputs) != 1 or len(logical.outputs) != 1:
        return reducer_plan

    logical_reduced = infer_unary_reduced_terms(
        lhs=logical.inputs[0],
        rhs=logical.outputs[0],
        op_name="einop",
    )
    if not logical_reduced:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: reduce_by has no logical axes to reduce",
            help="remove reduce_by from layout-only einop signatures",
            related=("einop layout normalization",),
            data={"operation": "einop"},
        )

    normalized_phases: list[ReducerPhase] = []
    covered_axes: list[AxisTermBase] = []
    logical_reduced_counts = logical_reduced.term_counts()
    for phase in reducer_plan:
        normalized_axes = normalization.normalize_terms(AxisTerms.from_spec(phase.axes))
        if (
            normalized_axes.term_counts()
            != (normalized_axes & logical_reduced).term_counts()
        ):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: reduce_by phase includes a layout-retained axis"
                ),
                help="restrict reducer phases to axes removed by the logical signature",
                related=("einop layout normalization",),
                data={"operation": "einop"},
            )
        normalized_phases.append(
            ReducerPhase(axes=normalized_axes, reducer=phase.reducer)
        )
        covered_axes.extend(normalized_axes)

    if AxisTerms.from_spec(tuple(covered_axes)).term_counts() != logical_reduced_counts:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                "inconsistent dims: reduce_by phases do not partition logical reduced axes"
            ),
            help="cover every logically reduced axis exactly once",
            related=("einop layout normalization",),
            data={"operation": "einop"},
        )
    return tuple(normalized_phases)


def _build_layout_normalized_symbolic_plan(
    *,
    execution_plan: EinopLoweringPlan,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Compose requested layouts around one logical einop plan."""
    normalization = execution_plan.layout_normalization
    if normalization is None:
        raise ValueError("layout-normalized einop plan requires layout metadata")

    requested = normalization.requested
    logical = normalization.logical
    input_steps = _build_layout_map_steps(
        sources=requested.inputs,
        targets=logical.inputs,
        explicit_sizes_items=explicit_sizes_items,
    )
    logical_reducer_plan = _normalize_layout_reducer_plan(
        normalization=normalization,
        reducer_plan=reducer_plan,
    )
    logical_plan: SymbolicPlan | None = None
    if len(logical.outputs) > 1 and logical_reducer_plan is None:
        try:
            direct_equations = build_einop_equations(
                input_axis_lists=logical.inputs,
                output_axis_lists=logical.outputs,
            )
        except ValidationError:
            pass
        else:
            logical_plan = _build_direct_einsum_symbolic_plan(
                lhs=logical.inputs,
                rhs=logical.outputs,
                equations=direct_equations,
            )
    if logical_plan is None:
        logical_plan = build_einop_symbolic_plan(
            build_default_ir_program(
                op_name="einop",
                lhs=logical.inputs,
                rhs=logical.outputs,
            ),
            explicit_sizes_items,
            logical_reducer_plan,
        )
    if any(
        step.specialization_depends_on_input_shapes() for step in logical_plan.steps
    ):
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message=(
                "ambiguous dims: layout-normalized einop has no unique logical "
                "axis mapping"
            ),
            help="rename repeated logical axes to make the layout mapping unique",
            related=("einop layout normalization",),
            data={"operation": "einop"},
        )
    output_steps = _build_layout_map_steps(
        sources=logical.outputs,
        targets=requested.outputs,
        explicit_sizes_items=explicit_sizes_items,
    )
    return SymbolicPlan(
        kind="layout_normalized",
        input_arity=len(requested.inputs),
        output_arity=len(requested.outputs),
        steps=(*input_steps, *logical_plan.steps, *output_steps),
    )


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
    except ValidationError as error:
        if "einop layout normalization" in error.related:
            raise
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

    if execution_plan.kind == "layout_normalized":
        return _build_layout_normalized_symbolic_plan(
            execution_plan=execution_plan,
            explicit_sizes_items=explicit_sizes_items,
            reducer_plan=reducer_plan,
        )

    if execution_plan.kind == "route":
        return SymbolicPlan(
            kind="route",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(),
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
        return _build_direct_einsum_symbolic_plan(
            lhs=lhs,
            rhs=rhs,
            equations=execution_plan.equations,
        )

    if (
        execution_plan.kind == "einsum_carrier_then_unary"
        and execution_plan.intermediate is not None
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
