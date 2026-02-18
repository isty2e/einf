from einf.axis import AxisExpr, AxisTermBase, AxisTerms, flatten_add_children
from einf.diagnostics import ValidationError
from einf.signature import Signature

from .base_plan import build_einop_execution_plan_base
from .equation import build_einop_equations
from .model import EinopLoweringPlan


def infer_carrier_candidates(
    *,
    analysis_signature: Signature,
) -> tuple[AxisTerms, ...]:
    """Infer deterministic carrier candidates for multi-output single-einsum lowering."""
    if len(analysis_signature.outputs) <= 1:
        return ()

    output_axis_lists = analysis_signature.outputs
    carrier_anchor_terms: dict[str, AxisExpr] = {}
    for input_axis_list in analysis_signature.inputs:
        for term in input_axis_list:
            if not isinstance(term, AxisExpr):
                continue
            if term.operator != "+":
                continue
            carrier_anchor_terms[term.stable_token()] = term

    if not carrier_anchor_terms:
        return ()

    output_count = len(output_axis_lists)
    output_width = len(output_axis_lists[0])
    if any(len(axis_list) != output_width for axis_list in output_axis_lists):
        return ()

    candidates: list[AxisTerms] = []
    for _, carrier_anchor in sorted(carrier_anchor_terms.items()):
        carrier_anchor_children = set(flatten_add_children(carrier_anchor))
        candidate_terms: list[AxisTermBase] = []
        valid = True
        for column in range(output_width):
            column_terms = tuple(
                output_axis_lists[row][column] for row in range(output_count)
            )
            unique_terms = tuple(dict.fromkeys(column_terms))
            if len(unique_terms) == 1:
                candidate_terms.append(unique_terms[0])
                continue
            if all(term in carrier_anchor_children for term in unique_terms):
                candidate_terms.append(carrier_anchor)
                continue
            valid = False
            break

        if not valid:
            continue

        candidate = AxisTerms(tuple(candidate_terms))
        if candidate in output_axis_lists:
            continue
        if candidate in candidates:
            continue
        candidates.append(candidate)

    return tuple(candidates)


def try_build_carrier_then_unary_plan(
    *,
    analysis_signature: Signature,
) -> EinopLoweringPlan | None:
    """Try one single-einsum carrier plan followed by unary tail lowering."""
    carrier_candidates = infer_carrier_candidates(analysis_signature=analysis_signature)
    for carrier in carrier_candidates:
        try:
            carrier_equation = build_einop_equations(
                input_axis_lists=analysis_signature.inputs,
                output_axis_lists=(carrier,),
            )[0]
        except ValidationError:
            continue

        stage_signature = Signature(
            inputs=(carrier,),
            outputs=analysis_signature.outputs,
        )
        stage_plan = build_einop_execution_plan_base(
            analysis_signature=stage_signature,
            has_reducer_plan=False,
        )
        if stage_plan.kind == "search_chain":
            continue
        if stage_plan.kind == "einsum" and len(stage_plan.equations) > 1:
            continue

        return EinopLoweringPlan(
            kind="einsum_carrier_then_unary",
            equations=(carrier_equation,),
            intermediate=carrier,
        )
    return None


__all__ = [
    "infer_carrier_candidates",
    "try_build_carrier_then_unary_plan",
]
