from einf.diagnostics import ErrorCode, ValidationError
from einf.einop_layout import EinopLayoutNormalization
from einf.signature import Signature
from einf.steps.einsum.equation import build_contract_equation

from .equation import build_einop_equations, has_nary_contraction_candidate
from .model import EinopLoweringPlan


def build_einop_execution_plan_base(
    *,
    analysis_signature: Signature,
    has_reducer_plan: bool,
) -> EinopLoweringPlan:
    """Build deterministic einop lowering plan before carrier-specific lifting."""
    if analysis_signature.inputs == analysis_signature.outputs and not has_reducer_plan:
        return EinopLoweringPlan(kind="route", equations=())

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
        return EinopLoweringPlan(
            kind="layout_normalized",
            equations=(),
            layout_normalization=layout_normalization,
        )

    if len(analysis_signature.inputs) == 1 and len(analysis_signature.outputs) == 1:
        lhs_terms = analysis_signature.inputs[0]
        rhs_terms = analysis_signature.outputs[0]
        reduced_terms = lhs_terms - rhs_terms
        introduced_terms = rhs_terms - lhs_terms
        has_reduction = has_reducer_plan or bool(reduced_terms)
        has_broadcast = bool(introduced_terms)

        if has_reduction and has_broadcast:
            return EinopLoweringPlan(kind="reduce_repeat", equations=())
        if has_reduction:
            return EinopLoweringPlan(kind="reduce", equations=())
        if has_broadcast:
            return EinopLoweringPlan(kind="repeat", equations=())
        return EinopLoweringPlan(kind="rearrange", equations=())

    if analysis_signature.is_atomic() and len(analysis_signature.outputs) == 1:
        equation = build_contract_equation(
            input_axis_lists=analysis_signature.inputs,
            output_axis_list=analysis_signature.outputs[0],
        )
        return EinopLoweringPlan(kind="contract", equations=(equation,))

    try:
        equations = build_einop_equations(
            input_axis_lists=analysis_signature.inputs,
            output_axis_lists=analysis_signature.outputs,
        )
    except ValidationError:
        if has_nary_contraction_candidate(analysis_signature):
            return EinopLoweringPlan(kind="search_chain", equations=())
        return EinopLoweringPlan(kind="rearrange", equations=())

    return EinopLoweringPlan(kind="einsum", equations=equations)


__all__ = ["build_einop_execution_plan_base"]
