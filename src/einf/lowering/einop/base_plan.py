from einf.diagnostics import ErrorCode, ValidationError
from einf.einop_layout import EinopLayoutNormalization
from einf.signature import Signature
from einf.steps.einsum.equation import build_contract_equation

from .equation import build_einop_equations, has_nary_contraction_candidate
from .model import (
    DirectEinsumEinopLoweringPlan,
    EinopChainSearchRequest,
    EinopLoweringPlan,
    EinopPrimitiveRoute,
    LayoutNormalizedEinopLoweringPlan,
    PrimitiveEinopLoweringPlan,
)


def build_einop_execution_plan_base(
    *,
    analysis_signature: Signature,
    has_reducer_plan: bool,
) -> EinopLoweringPlan | EinopChainSearchRequest:
    """Build an einop plan before carrier-chain search.

    Parameters
    ----------
    analysis_signature : Signature
        Canonical input and output axes to lower.
    has_reducer_plan : bool
        Whether the operation supplies an explicit reducer plan.

    Returns
    -------
    EinopLoweringPlan | EinopChainSearchRequest
        An executable base plan, or a request for carrier-chain search.

    Raises
    ------
    ValidationError
        If the signature cannot be normalized or represented by the selected
        base route.
    """
    if analysis_signature.inputs == analysis_signature.outputs and not has_reducer_plan:
        return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.ROUTE)

    layout_normalization = EinopLayoutNormalization.from_signature(analysis_signature)
    if layout_normalization.is_required:
        duplicate_terms = layout_normalization.generated_duplicate_terms()
        if duplicate_terms:
            raise ValidationError(
                code=ErrorCode.AMBIGUOUS_DIMS,
                message=(
                    "ambiguous dims: composite expansion introduces repeated "
                    "logical axes"
                ),
                help="rename repeated factors to give each logical axis one identity",
                related=("einop layout normalization",),
                data={"operation": "einop", "terms": ",".join(duplicate_terms)},
            )
        return LayoutNormalizedEinopLoweringPlan(
            normalization=layout_normalization,
        )

    if len(analysis_signature.inputs) == 1 and len(analysis_signature.outputs) == 1:
        lhs_terms = analysis_signature.inputs[0]
        rhs_terms = analysis_signature.outputs[0]
        reduced_terms = lhs_terms - rhs_terms
        introduced_terms = rhs_terms - lhs_terms
        has_reduction = has_reducer_plan or bool(reduced_terms)
        has_broadcast = bool(introduced_terms)

        if has_reduction and has_broadcast:
            return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REDUCE_REPEAT)
        if has_reduction:
            return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REDUCE)
        if has_broadcast:
            return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REPEAT)
        return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)

    if analysis_signature.is_atomic() and len(analysis_signature.outputs) == 1:
        _ = build_contract_equation(
            input_axis_lists=analysis_signature.inputs,
            output_axis_list=analysis_signature.outputs[0],
        )
        return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.CONTRACT)

    try:
        equations = build_einop_equations(
            input_axis_lists=analysis_signature.inputs,
            output_axis_lists=analysis_signature.outputs,
        )
    except ValidationError:
        if has_nary_contraction_candidate(analysis_signature):
            return EinopChainSearchRequest()
        return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)

    return DirectEinsumEinopLoweringPlan(equations=equations)


__all__ = ["build_einop_execution_plan_base"]
