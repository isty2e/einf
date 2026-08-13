import pytest

from einf import ErrorCode, ValidationError, ax, axes, einop
from einf.axis import AxisTermBase, AxisTerms
from einf.lowering.einop import (
    ChainEinopLoweringPlan,
    EinopLeafLoweringPlan,
    EinopPrimitiveRoute,
    PrimitiveEinopLoweringPlan,
    all_subset_axis_lists,
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


def test_public_single_output_preserves_chain_search_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_plan_module,
        "_DEFAULT_EINOP_CHAIN_SEARCH_CANDIDATE_LIMIT",
        7,
    )
    a, b, c, d, contracted, left, right = axes(
        "single_limit_a",
        "single_limit_b",
        "single_limit_c",
        "single_limit_d",
        "single_limit_contracted",
        "single_limit_left",
        "single_limit_right",
    )

    with pytest.raises(ValidationError) as error:
        einop(
            (
                ax[a, a, b, contracted, left + right],
                ax[a, c, contracted],
                ax[a, d, contracted],
            ),
            ax[a],
        )

    assert error.value.code == ErrorCode.EINOP_PLANNING_TOO_COMPLEX.value
    assert error.value.data["attempted"] == 32


def test_chain_search_shares_candidate_budget_across_carriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_plan_module,
        "_DEFAULT_EINOP_CHAIN_SEARCH_CANDIDATE_LIMIT",
        100,
    )
    a, b, c, contracted, retained_left, retained_right = axes(
        "carrier_limit_a",
        "carrier_limit_b",
        "carrier_limit_c",
        "carrier_limit_contracted",
        "carrier_limit_left",
        "carrier_limit_right",
    )
    signature = Signature(
        inputs=(
            ax[a, a, b, contracted, retained_left, retained_right],
            ax[a, c, contracted],
        ),
        outputs=(ax[a, c],),
    )

    with pytest.raises(ValidationError) as error:
        search_plan_module.build_symbolic_einsum_chain_plan(
            analysis_signature=signature,
            tail_builder=_terminal_tail,
        )

    assert error.value.data == {
        "operation": "einop",
        "complexity_kind": "chain_search_candidates",
        "limit": 100,
        "attempted": 128,
    }


class _MembershipProbeAxis(AxisTermBase):
    def __init__(
        self,
        *,
        token: str,
        hash_value: int,
        comparisons: list[int],
    ) -> None:
        self._token = token
        self._hash_value = hash_value
        self._comparisons = comparisons

    def __hash__(self) -> int:
        return self._hash_value

    def __eq__(self, other: object) -> bool:
        self._comparisons[0] += 1
        return self is other

    def to_dsl(self) -> str:
        return self._token

    def stable_token(self) -> str:
        return f"probe:{self._token}"

    def axis_names(self) -> set[str]:
        return set()

    def pack_names(self) -> set[str]:
        return set()


def test_subset_scoring_does_not_repeat_membership_work_per_candidate() -> None:
    ordered_terms = AxisTerms(axes(*(f"ordered_{index}" for index in range(10))))
    target_comparisons = [0]
    remaining_comparisons = [0]
    target_terms: set[AxisTermBase] = {
        _MembershipProbeAxis(
            token=f"target_{index}",
            hash_value=hash(ordered_terms[index % len(ordered_terms)]),
            comparisons=target_comparisons,
        )
        for index in range(30)
    }
    remaining_terms: set[AxisTermBase] = {
        _MembershipProbeAxis(
            token=f"remaining_{index}",
            hash_value=hash(ordered_terms[index % len(ordered_terms)]),
            comparisons=remaining_comparisons,
        )
        for index in range(60)
    }
    target_comparisons[0] = 0
    remaining_comparisons[0] = 0

    candidates = all_subset_axis_lists(
        ordered_terms=ordered_terms,
        target_terms=target_terms,
        remaining_terms=remaining_terms,
    )

    assert len(candidates) == 1_024
    assert target_comparisons[0] <= len(ordered_terms) * len(target_terms)
    assert remaining_comparisons[0] <= len(ordered_terms) * len(remaining_terms)


def test_subset_bitmask_scoring_preserves_candidate_order() -> None:
    a, b, c = axes("subset_order_a", "subset_order_b", "subset_order_c")

    candidates = all_subset_axis_lists(
        ordered_terms=AxisTerms((a, b, c)),
        target_terms={a},
        remaining_terms={b},
    )

    assert candidates == (
        ax[a, b],
        ax[a, b, c],
        ax[a],
        ax[a, c],
        ax[b],
        ax[b, c],
        ax[()],
        ax[c],
    )


def test_subset_scoring_uses_stable_tokens_to_break_score_ties() -> None:
    earlier, middle, later = axes(
        "subset_tie_a",
        "subset_tie_m",
        "subset_tie_z",
    )

    candidates = all_subset_axis_lists(
        ordered_terms=AxisTerms((later, middle, earlier)),
        target_terms=set(),
        remaining_terms=set(),
    )

    assert candidates == (
        ax[()],
        ax[earlier],
        ax[middle],
        ax[later],
        ax[middle, earlier],
        ax[later, earlier],
        ax[later, middle],
        ax[later, middle, earlier],
    )
