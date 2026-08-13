import pytest

from einf import ErrorCode, ValidationError, ax, axes, einop
from einf.axis import AxisTermBase, AxisTerms
from einf.lowering.einop import (
    ChainEinopLoweringPlan,
    EinopLeafLoweringPlan,
    EinopPrimitiveRoute,
    PrimitiveEinopLoweringPlan,
)
from einf.lowering.einop import search_plan as search_plan_module
from einf.signature import Signature


def _terminal_tail(_signature: Signature) -> EinopLeafLoweringPlan:
    return PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)


def _fail_if_subset_space_is_materialized(
    *,
    ordered_terms: AxisTerms,
    target_terms: set[AxisTermBase],
    remaining_terms: set[AxisTermBase],
) -> tuple[AxisTerms, ...]:
    _ = ordered_terms, target_terms, remaining_terms
    raise AssertionError("over-limit subset space must not be materialized")


def _three_input_chain_signature(
    *, retained_terms: AxisTerms | None = None
) -> Signature:
    a, b, c, d = axes("a", "b", "c", "d")
    retained = AxisTerms() if retained_terms is None else retained_terms
    return Signature(
        inputs=(ax[(a, b, *retained)], ax[b, c], ax[c, d]),
        outputs=(ax[(a, d, *retained)],),
    )


def test_chain_search_accepts_exact_shared_candidate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_plan_module,
        "_DEFAULT_EINOP_CHAIN_SEARCH_CANDIDATE_LIMIT",
        16,
    )

    plan = search_plan_module.build_symbolic_einsum_chain_plan(
        analysis_signature=_three_input_chain_signature(),
        tail_builder=_terminal_tail,
    )

    assert isinstance(plan, ChainEinopLoweringPlan)
    assert plan.carrier_index == 0
    assert plan.chain_order == (1, 2)


def test_chain_search_rejects_next_space_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_enumerator = search_plan_module.all_subset_axis_lists
    materialized_candidate_counts: list[int] = []

    def observe_enumerator(
        *,
        ordered_terms: AxisTerms,
        target_terms: set[AxisTermBase],
        remaining_terms: set[AxisTermBase],
    ) -> tuple[AxisTerms, ...]:
        candidates = original_enumerator(
            ordered_terms=ordered_terms,
            target_terms=target_terms,
            remaining_terms=remaining_terms,
        )
        materialized_candidate_counts.append(len(candidates))
        return candidates

    monkeypatch.setattr(
        search_plan_module,
        "_DEFAULT_EINOP_CHAIN_SEARCH_CANDIDATE_LIMIT",
        15,
    )
    monkeypatch.setattr(
        search_plan_module,
        "all_subset_axis_lists",
        observe_enumerator,
    )

    with pytest.raises(ValidationError) as error:
        search_plan_module.build_symbolic_einsum_chain_plan(
            analysis_signature=_three_input_chain_signature(),
            tail_builder=_terminal_tail,
        )

    captured = error.value
    assert captured.code == ErrorCode.EINOP_PLANNING_TOO_COMPLEX.value
    assert captured.external_code == "EINOP_PLANNING_TOO_COMPLEX"
    assert captured.help == (
        "simplify the n-ary einop signature or split it into smaller operations"
    )
    assert captured.related == ("einop lowering",)
    assert captured.data == {
        "operation": "einop",
        "complexity_kind": "chain_search_candidates",
        "limit": 15,
        "attempted": 16,
    }
    assert materialized_candidate_counts == [8]


def test_chain_search_rejects_first_space_before_enumerator_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_plan_module,
        "_DEFAULT_EINOP_CHAIN_SEARCH_CANDIDATE_LIMIT",
        7,
    )
    monkeypatch.setattr(
        search_plan_module,
        "all_subset_axis_lists",
        _fail_if_subset_space_is_materialized,
    )
    a, b, c, d = axes(
        "bounded_public_a",
        "bounded_public_b",
        "bounded_public_c",
        "bounded_public_d",
    )

    with pytest.raises(ValidationError) as error:
        einop(
            (ax[a, b], ax[a, c], ax[a, d]),
            (ax[a], ax[c]),
        )

    assert error.value.data["attempted"] == 8


def test_chain_search_preserves_valid_high_rank_plan_below_default_limit() -> None:
    retained_terms = AxisTerms(axes(*(f"retained_{index}" for index in range(8))))

    plan = search_plan_module.build_symbolic_einsum_chain_plan(
        analysis_signature=_three_input_chain_signature(retained_terms=retained_terms),
        tail_builder=_terminal_tail,
    )

    assert isinstance(plan, ChainEinopLoweringPlan)
    assert plan.carrier_index == 0
    assert plan.chain_order == (1, 2)


def test_chain_search_default_limit_rejects_oversized_first_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_plan_module,
        "all_subset_axis_lists",
        _fail_if_subset_space_is_materialized,
    )
    retained_terms = AxisTerms(axes(*(f"oversized_{index}" for index in range(14))))

    with pytest.raises(ValidationError) as error:
        search_plan_module.build_symbolic_einsum_chain_plan(
            analysis_signature=_three_input_chain_signature(
                retained_terms=retained_terms
            ),
            tail_builder=_terminal_tail,
        )

    assert error.value.data == {
        "operation": "einop",
        "complexity_kind": "chain_search_candidates",
        "limit": 65_536,
        "attempted": 131_072,
    }
