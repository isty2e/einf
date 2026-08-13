import numpy as np
import pytest

from einf import ax, axes, einop
from einf.lowering.builders import einop as einop_builder_module
from einf.lowering.einop import EinopLoweringPlan, build_einop_execution_plan
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
