from einf.axis import (
    AxisExpr,
    AxisTerms,
    ScalarAxisTermBase,
    ScalarAxisTerms,
    expand_products_for_terms,
    flatten_add_children,
)
from einf.diagnostics import ValidationError
from einf.signature import Signature

from .base_plan import build_einop_execution_plan_base
from .equation import (
    all_subset_axis_lists,
    build_einop_equations,
    ordered_unique_axis_terms,
)
from .model import EinopLoweringPlan


def build_symbolic_einsum_chain_plan(
    *,
    analysis_signature: Signature,
) -> EinopLoweringPlan | None:
    """Build symbolic chain plan without runtime tensor probing."""
    input_axis_lists = analysis_signature.inputs
    output_axis_lists = analysis_signature.outputs
    target_terms = {term for axis_list in output_axis_lists for term in axis_list}
    failed_states: set[tuple[AxisTerms, tuple[int, ...]]] = set()

    def is_multi_output_split_feasible(
        carrier_terms: AxisTerms,
    ) -> bool:
        """Return whether one carrier can split directly into multi-outputs."""
        if len(output_axis_lists) < 2:
            return False
        rank = len(carrier_terms)
        if rank == 0:
            return False
        if any(len(output_terms) != rank for output_terms in output_axis_lists):
            return False

        split_axis = -1
        for axis in range(rank):
            axis_terms = tuple(output_terms[axis] for output_terms in output_axis_lists)
            if all(term == carrier_terms[axis] for term in axis_terms):
                continue
            if split_axis != -1:
                return False
            split_axis = axis
        if split_axis == -1:
            return False

        for axis in range(rank):
            if axis == split_axis:
                continue
            if any(
                output_terms[axis] != carrier_terms[axis]
                for output_terms in output_axis_lists
            ):
                return False

        input_split_term = ScalarAxisTermBase.coerce(carrier_terms[split_axis])
        output_split_terms = tuple(
            ScalarAxisTermBase.coerce(output_terms[split_axis])
            for output_terms in output_axis_lists
        )

        expanded_input_split = ScalarAxisTermBase.coerce(
            expand_products_for_terms(ScalarAxisTerms((input_split_term,)))[0]
        )

        def split_children(term: ScalarAxisTermBase) -> tuple[ScalarAxisTermBase, ...]:
            return tuple(
                ScalarAxisTermBase.coerce(child) for child in flatten_add_children(term)
            )

        expected_split_variants: list[tuple[ScalarAxisTermBase, ...]] = [
            split_children(input_split_term),
            split_children(expanded_input_split),
        ]
        if isinstance(input_split_term, AxisExpr) and input_split_term.operator == "*":
            if (
                isinstance(input_split_term.left, AxisExpr)
                and input_split_term.left.operator == "+"
            ):
                expected_split_variants.append(
                    tuple(
                        AxisExpr("*", child, input_split_term.right)
                        for child in split_children(input_split_term.left)
                    )
                )
            if (
                isinstance(input_split_term.right, AxisExpr)
                and input_split_term.right.operator == "+"
            ):
                expected_split_variants.append(
                    tuple(
                        AxisExpr("*", input_split_term.left, child)
                        for child in split_children(input_split_term.right)
                    )
                )

        return output_split_terms in expected_split_variants

    def dfs(
        *,
        carrier_index: int,
        carrier_terms: AxisTerms,
        remaining_indices: tuple[int, ...],
        chain_order: tuple[int, ...],
        equations: tuple[str, ...],
    ) -> EinopLoweringPlan | None:
        state_key = (carrier_terms, remaining_indices)
        if state_key in failed_states:
            return None

        if not remaining_indices:
            stage_signature = Signature(
                inputs=(carrier_terms,),
                outputs=output_axis_lists,
            )
            stage_plan = build_einop_execution_plan_base(
                analysis_signature=stage_signature,
                has_reducer_plan=False,
            )
            if stage_plan.kind == "search_chain":
                failed_states.add(state_key)
                return None
            if stage_plan.kind == "rearrange" and len(output_axis_lists) > 1:
                if not is_multi_output_split_feasible(carrier_terms):
                    failed_states.add(state_key)
                    return None

            return EinopLoweringPlan(
                kind="einsum_chain_then_unary",
                equations=equations,
                intermediate=carrier_terms,
                carrier_index=carrier_index,
                chain_order=chain_order,
                tail_kind=stage_plan.kind,
            )

        for next_index in remaining_indices:
            next_terms = input_axis_lists[next_index]
            next_remaining_indices = tuple(
                index for index in remaining_indices if index != next_index
            )
            remaining_terms = {
                term
                for index in next_remaining_indices
                for term in input_axis_lists[index]
            }
            ordered_terms = ordered_unique_axis_terms(
                carrier_terms,
                next_terms,
            )
            candidate_axis_lists = all_subset_axis_lists(
                ordered_terms=ordered_terms,
                target_terms=target_terms,
                remaining_terms=remaining_terms,
            )
            for candidate_terms in candidate_axis_lists:
                try:
                    equation = build_einop_equations(
                        input_axis_lists=(carrier_terms, next_terms),
                        output_axis_lists=(candidate_terms,),
                    )[0]
                except ValidationError:
                    continue

                found = dfs(
                    carrier_index=carrier_index,
                    carrier_terms=candidate_terms,
                    remaining_indices=next_remaining_indices,
                    chain_order=chain_order + (next_index,),
                    equations=equations + (equation,),
                )
                if found is not None:
                    return found

        failed_states.add(state_key)
        return None

    for carrier_index in range(len(input_axis_lists)):
        remaining_indices = tuple(
            index for index in range(len(input_axis_lists)) if index != carrier_index
        )
        found = dfs(
            carrier_index=carrier_index,
            carrier_terms=input_axis_lists[carrier_index],
            remaining_indices=remaining_indices,
            chain_order=(),
            equations=(),
        )
        if found is not None:
            return found

    return None


__all__ = ["build_symbolic_einsum_chain_plan"]
