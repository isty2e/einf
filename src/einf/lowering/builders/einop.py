from einf.axis import AxisSide, AxisTermBase, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.einop_layout import EinopLayoutNormalization
from einf.ir import IRProgram, LoweringSignature
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

from ..einop import (
    CarrierEinopLoweringPlan,
    ChainEinopLoweringPlan,
    DirectEinsumEinopLoweringPlan,
    EinopLoweringPlan,
    EinopPrimitiveRoute,
    LayoutNormalizedEinopLoweringPlan,
    PrimitiveEinopLoweringPlan,
    build_einop_execution_plan,
)
from ..einop.equation import build_einop_equations
from .contract import build_contract_symbolic_plan
from .rearrange import build_rearrange_symbolic_plan
from .reduce import build_reduce_symbolic_plan
from .repeat import build_repeat_symbolic_plan


def _build_direct_einsum_symbolic_plan(
    *,
    source: LoweringSignature,
    equations: tuple[str, ...],
) -> SymbolicPlan:
    """Build independent direct einsum outputs from one shared input tuple."""
    lhs = source.signature.inputs
    rhs = source.signature.outputs
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
        source=source,
        kind="einsum",
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
            IRProgram.from_source(
                LoweringSignature(
                    op_name="rearrange",
                    signature=Signature(
                        inputs=AxisSide.from_spec((source,), side_name="lhs"),
                        outputs=AxisSide.from_spec((target,), side_name="rhs"),
                    ),
                    explicit_sizes_items=explicit_sizes_items,
                )
            ),
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
    source: LoweringSignature,
    execution_plan: LayoutNormalizedEinopLoweringPlan,
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Compose requested layouts around one logical einop plan."""
    explicit_sizes_items = source.explicit_sizes_items
    normalization = execution_plan.normalization
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
                source=LoweringSignature(
                    op_name="einop",
                    signature=logical,
                    explicit_sizes_items=explicit_sizes_items,
                ),
                equations=direct_equations,
            )
    if logical_plan is None:
        logical_plan = build_einop_symbolic_plan(
            IRProgram.from_source(
                LoweringSignature(
                    op_name="einop",
                    signature=logical,
                    explicit_sizes_items=explicit_sizes_items,
                )
            ),
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
        source=source,
        kind="layout_normalized",
        steps=(*input_steps, *logical_plan.steps, *output_steps),
    )


def _build_reduce_repeat_symbolic_plan(
    *,
    source: LoweringSignature,
) -> SymbolicPlan | None:
    """Build one `reduce -> repeat` symbolic plan for unary signatures."""
    lhs = source.signature.inputs
    rhs = source.signature.outputs
    explicit_sizes_items = source.explicit_sizes_items
    if len(lhs) != 1 or len(rhs) != 1:
        return None

    shared_terms = lhs[0] & rhs[0]
    reduced_rhs = AxisSide.from_spec((shared_terms,), side_name="rhs")
    repeated_lhs = AxisSide.from_spec((shared_terms,), side_name="lhs")
    reduce_plan = build_reduce_symbolic_plan(
        IRProgram.from_source(
            LoweringSignature(
                op_name="reduce",
                signature=Signature(inputs=lhs, outputs=reduced_rhs),
                explicit_sizes_items=explicit_sizes_items,
            )
        ),
        None,
    )
    repeat_plan = build_repeat_symbolic_plan(
        IRProgram.from_source(
            LoweringSignature(
                op_name="repeat",
                signature=Signature(inputs=repeated_lhs, outputs=rhs),
                explicit_sizes_items=explicit_sizes_items,
            )
        ),
        None,
    )
    return SymbolicPlan(
        source=source,
        kind="reduce_repeat",
        steps=(*reduce_plan.steps, *repeat_plan.steps),
    )


def _build_selected_einop_symbolic_plan(
    *,
    source: LoweringSignature,
    execution_plan: EinopLoweringPlan,
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan from a canonical einop lowering variant."""
    lhs = source.signature.inputs
    rhs = source.signature.outputs
    explicit_sizes_items = source.explicit_sizes_items
    if isinstance(execution_plan, LayoutNormalizedEinopLoweringPlan):
        return _build_layout_normalized_symbolic_plan(
            source=source,
            execution_plan=execution_plan,
            reducer_plan=reducer_plan,
        )

    if isinstance(execution_plan, PrimitiveEinopLoweringPlan):
        route = execution_plan.route
        if route is EinopPrimitiveRoute.ROUTE:
            return SymbolicPlan(
                source=source,
                kind=execution_plan.symbolic_kind,
                steps=(),
            )
        if route is EinopPrimitiveRoute.REARRANGE:
            primitive_plan = build_rearrange_symbolic_plan(
                IRProgram.from_source(
                    LoweringSignature(
                        op_name="rearrange",
                        signature=source.signature,
                        explicit_sizes_items=explicit_sizes_items,
                    )
                ),
                None,
            )
        elif route is EinopPrimitiveRoute.REPEAT:
            primitive_plan = build_repeat_symbolic_plan(
                IRProgram.from_source(
                    LoweringSignature(
                        op_name="repeat",
                        signature=source.signature,
                        explicit_sizes_items=explicit_sizes_items,
                    )
                ),
                None,
            )
        elif route is EinopPrimitiveRoute.REDUCE:
            primitive_plan = build_reduce_symbolic_plan(
                IRProgram.from_source(
                    LoweringSignature(
                        op_name="reduce",
                        signature=source.signature,
                        explicit_sizes_items=explicit_sizes_items,
                    )
                ),
                reducer_plan,
            )
        elif route is EinopPrimitiveRoute.REDUCE_REPEAT:
            reduce_repeat_plan = _build_reduce_repeat_symbolic_plan(
                source=source,
            )
            if reduce_repeat_plan is None:
                raise ValueError("reduce-repeat einop lowering must be unary")
            return reduce_repeat_plan
        elif route is EinopPrimitiveRoute.CONTRACT:
            primitive_plan = build_contract_symbolic_plan(
                IRProgram.from_source(
                    LoweringSignature(
                        op_name="contract",
                        signature=source.signature,
                        explicit_sizes_items=explicit_sizes_items,
                    )
                ),
                None,
            )
        else:
            raise ValueError(f"unsupported primitive einop route: {route!r}")
        return SymbolicPlan(
            source=source,
            kind=primitive_plan.kind,
            steps=primitive_plan.steps,
        )

    if isinstance(execution_plan, DirectEinsumEinopLoweringPlan):
        return _build_direct_einsum_symbolic_plan(
            source=source,
            equations=execution_plan.equations,
        )

    if isinstance(execution_plan, CarrierEinopLoweringPlan):
        carrier_step = EinsumSymbolicStep(
            program=build_einsum_symbolic_program_from_equations(
                input_arity=len(lhs),
                output_arity=1,
                equations=(execution_plan.equation,),
                allow_native_matmul=True,
            )
        )
        carrier_lhs = AxisSide.from_spec(
            (execution_plan.intermediate,),
            side_name="lhs",
        )
        tail_plan = _build_selected_einop_symbolic_plan(
            source=LoweringSignature(
                op_name="einop",
                signature=Signature(inputs=carrier_lhs, outputs=rhs),
                explicit_sizes_items=explicit_sizes_items,
            ),
            execution_plan=execution_plan.tail,
            reducer_plan=None,
        )
        if tail_plan.input_arity != 1:
            raise ValueError("carrier tail lowering must be unary")
        return SymbolicPlan(
            source=source,
            kind=execution_plan.symbolic_kind,
            steps=(carrier_step, *tail_plan.steps),
        )

    if isinstance(execution_plan, ChainEinopLoweringPlan):
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
        tail_plan = _build_selected_einop_symbolic_plan(
            source=LoweringSignature(
                op_name="einop",
                signature=Signature(inputs=carrier_lhs, outputs=rhs),
                explicit_sizes_items=explicit_sizes_items,
            ),
            execution_plan=execution_plan.tail,
            reducer_plan=None,
        )
        if tail_plan.input_arity != 1:
            raise ValueError("einsum chain tail lowering must be unary")
        return SymbolicPlan(
            source=source,
            kind=execution_plan.symbolic_kind,
            steps=(chain_step, *tail_plan.steps),
        )

    raise TypeError(f"unsupported einop lowering plan: {type(execution_plan).__name__}")


def build_einop_symbolic_plan(
    ir_program: IRProgram,
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for ``einop``.

    Parameters
    ----------
    ir_program : IRProgram
        Source-bound einop IR.
    reducer_plan : ReducerPlan or None
        Optional ordered reducer phases.

    Returns
    -------
    SymbolicPlan
        Einop plan carrying ``ir_program.source``.

    Raises
    ------
    ValidationError
        If no feasible lowering exists within the planning contract.
    """
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    explicit_sizes_items = ir_program.explicit_sizes_items
    analysis_signature = Signature(inputs=lhs, outputs=rhs)
    try:
        execution_plan = build_einop_execution_plan(
            analysis_signature=analysis_signature,
            has_reducer_plan=reducer_plan is not None,
        )
    except ValidationError as error:
        if error.code == ErrorCode.EINOP_PLANNING_TOO_COMPLEX.value:
            raise
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
            source=ir_program.source,
            kind="einsum",
            steps=(step,),
        )

    return _build_selected_einop_symbolic_plan(
        source=ir_program.source,
        execution_plan=execution_plan,
        reducer_plan=reducer_plan,
    )


__all__ = [
    "build_einop_symbolic_plan",
]
