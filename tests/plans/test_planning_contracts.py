from collections.abc import Mapping

from einf import ax, axes, einop, packs, rearrange, reduce


def _step_ops(plan_dict: Mapping[str, object]) -> list[str]:
    """Extract step operation names from one plan payload."""
    raw_steps = plan_dict["steps"]
    assert isinstance(raw_steps, list)
    ops: list[str] = []
    for step in raw_steps:
        assert isinstance(step, dict)
        op = step.get("op")
        assert isinstance(op, str)
        ops.append(op)
    return ops


def test_planning_contract_einop_carrier_then_unary_kind_is_stable() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    )

    plan_dict = op.plan_dict()
    assert plan_dict["kind"] == "einsum_carrier_then_unary"
    assert _step_ops(plan_dict) == ["einsum", "axis_slice"]


def test_planning_contract_signature_rows_render_as_dsl() -> None:
    b, c = axes("b", "c")
    (T,) = packs("T")
    op = rearrange(ax[b, T, c], ax[c, b, T])

    plan_dict = op.plan_dict()
    lhs = plan_dict["lhs"]
    rhs = plan_dict["rhs"]
    assert lhs == ["ax[b, *T, c]"]
    assert rhs == ["ax[c, b, *T]"]
    assert "Axis(" not in "".join(lhs + rhs)
    assert "AxisPack(" not in "".join(lhs + rhs)


def test_planning_contract_reduce_reducer_labeling_is_stable() -> None:
    b, h, d = axes("b", "h", "d")
    default_op = reduce(ax[b, h, d], ax[b])
    phased_op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "prod"))

    default_steps = default_op.plan_dict()["steps"]
    phased_steps = phased_op.plan_dict()["steps"]
    assert default_steps == [{"op": "reduce", "reducer": "sum(default)"}]
    assert phased_steps == [
        {"op": "reduce", "reducer": "sum"},
        {"op": "reduce", "reducer": "prod"},
    ]
