from collections.abc import Mapping

from einf import ax, axes, einop, rearrange, reduce, repeat
from einf.steps.reduce import ReduceSymbolicStep


def _extract_steps(plan_dict: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract typed plan steps from one `plan_dict` payload."""
    raw_steps = plan_dict["steps"]
    assert isinstance(raw_steps, list)
    normalized_steps: list[dict[str, object]] = []
    for step in raw_steps:
        assert isinstance(step, dict)
        normalized_steps.append(step)
    return normalized_steps


def test_plan_dict_rearrange_reports_one_step() -> None:
    b, n, d = axes("b", "n", "d")
    op = rearrange(ax[b, n, d], ax[b, d, n])

    plan_dict = op.plan_dict()
    assert plan_dict["kind"] == "permute"
    assert plan_dict["executable_now"] is True

    steps = _extract_steps(plan_dict)
    assert steps == [{"op": "permute"}]

    plan_text = op.plan()
    assert "kind: permute" in plan_text
    assert "1. permute" in plan_text


def test_plan_dict_marks_missing_introduced_size_binding() -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r])

    plan_dict = op.plan_dict()
    assert plan_dict["executable_now"] is False

    blockers = plan_dict["blockers"]
    assert isinstance(blockers, list)
    assert blockers == ["missing size binding: r"]


def test_plan_dict_marks_bound_introduced_size_as_executable() -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=3)

    plan_dict = op.plan_dict()
    assert plan_dict["executable_now"] is True

    blockers = plan_dict["blockers"]
    assert isinstance(blockers, list)
    assert blockers == []


def test_plan_dict_einop_reports_carrier_then_unary_steps() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    )

    plan_dict = op.plan_dict()
    assert plan_dict["kind"] == "einsum_carrier_then_unary"

    steps = _extract_steps(plan_dict)
    assert len(steps) == 2
    assert steps[0] == {"op": "einsum", "equation": "abc,cd,de->abe"}
    assert steps[1] == {"op": "axis_slice"}


def test_plan_dict_reduce_reports_default_and_custom_reducer_labels() -> None:
    b, h = axes("b", "h")
    default_op = reduce(ax[b, h], ax[b])
    custom_op = reduce(ax[b, h], ax[b]).reduce_by("max")

    default_plan = default_op.plan_dict()
    custom_plan = custom_op.plan_dict()

    default_steps = _extract_steps(default_plan)
    custom_steps = _extract_steps(custom_plan)
    assert default_steps == [{"op": "reduce", "reducer": "sum(default)"}]
    assert custom_steps == [{"op": "reduce", "reducer": "max"}]


def test_tensorop_exposes_canonical_abstract_plan_surface() -> None:
    b, n, d = axes("b", "n", "d")
    op = rearrange(ax[b, n, d], ax[b, d, n]).with_sizes(n=3)

    abstract = op.abstract_plan
    assert abstract.op_name == "rearrange"
    assert abstract.lhs == op.lhs
    assert abstract.rhs == op.rhs


def test_reduce_by_concretizes_reduce_symbolic_candidate() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).with_sizes(h=3).reduce_by("prod")

    candidate = op.abstract_plan.symbolic_candidates[0]
    step = candidate.steps[0]
    assert isinstance(step, ReduceSymbolicStep)
    assert step.reducer_label() == "prod"
    assert step.explicit_sizes_items == (("h", 3),)


def test_reduce_plan_dict_emits_reduce_then_permute_for_rhs_reorder() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[d, b])

    plan_dict = op.plan_dict()
    steps = _extract_steps(plan_dict)
    assert steps == [
        {"op": "reduce", "reducer": "sum(default)"},
        {"op": "permute"},
    ]
