import numpy as np
import pytest

from einf import ax, axes, einop
from einf.axis import AxisSide
from einf.lowering.builders import einop as einop_builder_module
from einf.lowering.einop import (
    ChainEinopLoweringPlan,
    DirectEinsumEinopLoweringPlan,
    EinopLoweringPlan,
    build_einop_execution_plan,
)
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.signature import Signature


def test_einop_uses_single_einsum_carrier_path_by_default() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    _ = op(
        np.arange(2 * 3 * 5).reshape(2, 3, 5),
        np.arange(5 * 4).reshape(5, 4),
        np.arange(4 * 6).reshape(4, 6),
    )

    plan = op.plan_dict()
    assert plan["kind"] == "einsum_carrier_then_unary"
    assert len(plan["steps"]) == 2
    assert plan["steps"][0] == {"op": "einsum", "equation": "abc,cd,de->abe"}
    assert plan["steps"][1] == {"op": "axis_slice"}


def test_einop_carrier_builder_consumes_preselected_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Signature, bool]] = []

    def counting_build_einop_execution_plan(
        *,
        analysis_signature: Signature,
        has_reducer_plan: bool,
    ) -> EinopLoweringPlan:
        calls.append((analysis_signature, has_reducer_plan))
        return build_einop_execution_plan(
            analysis_signature=analysis_signature,
            has_reducer_plan=has_reducer_plan,
        )

    monkeypatch.setattr(
        einop_builder_module,
        "build_einop_execution_plan",
        counting_build_einop_execution_plan,
    )
    b, h, w, d, j, k = axes("batch", "height", "width", "depth", "joint", "key")
    op = einop(
        (ax[b, h + w, d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    )

    plan = op.plan_dict()

    assert plan["kind"] == "einsum_carrier_then_unary"
    assert len(calls) == 1


def test_einop_chain_execution_preserves_planner_order() -> None:
    a, b, c, d = axes("chain_a", "chain_b", "chain_c", "chain_d")
    op = einop(
        (ax[a, b], ax[a, c], ax[a, d]),
        (ax[a], ax[c]),
    )
    first = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    second = np.arange(1, 9, dtype=np.float64).reshape(2, 4)
    third = np.arange(1, 11, dtype=np.float64).reshape(2, 5)

    actual = op(first, second, third)

    assert op.plan_dict()["kind"] == "einsum_chain_then_unary"
    np.testing.assert_array_equal(
        actual[0],
        np.einsum("ab,ac,ad->a", first, second, third),
    )
    np.testing.assert_array_equal(
        actual[1],
        np.einsum("ab,ac,ad->c", first, second, third),
    )


def test_einop_chain_builder_consumes_preselected_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b, c, d = axes(
        "chain_call_a",
        "chain_call_b",
        "chain_call_c",
        "chain_call_d",
    )
    signature = Signature(
        inputs=(ax[a, b], ax[a, c], ax[a, d]),
        outputs=(ax[a],),
    )
    stored_tail = DirectEinsumEinopLoweringPlan(equations=("a->a",))
    chain_plan = ChainEinopLoweringPlan(
        equations=("ab,ac->a", "a,ad->a"),
        intermediate=ax[a],
        carrier_index=0,
        chain_order=(1, 2),
        tail=stored_tail,
    )
    projected_plans: list[EinopLoweringPlan] = []
    build_selected_plan = einop_builder_module._build_selected_einop_symbolic_plan

    def observe_selected_plan(
        *,
        execution_plan: EinopLoweringPlan,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
        reducer_plan: ReducerPlan | None,
    ) -> SymbolicPlan:
        projected_plans.append(execution_plan)
        return build_selected_plan(
            execution_plan=execution_plan,
            lhs=lhs,
            rhs=rhs,
            explicit_sizes_items=explicit_sizes_items,
            reducer_plan=reducer_plan,
        )

    monkeypatch.setattr(
        einop_builder_module,
        "_build_selected_einop_symbolic_plan",
        observe_selected_plan,
    )
    symbolic_plan = observe_selected_plan(
        execution_plan=chain_plan,
        lhs=signature.inputs,
        rhs=signature.outputs,
        explicit_sizes_items=(),
        reducer_plan=None,
    )

    assert symbolic_plan.kind == "einsum_chain_then_unary"
    assert projected_plans[0] is chain_plan
    assert projected_plans[1] is stored_tail
