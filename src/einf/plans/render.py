from typing import Literal

try:
    from typing import NotRequired, TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired, TypedDict

from einf.axis import AxisSide
from einf.signature import Signature
from einf.steps.einsum import EinsumSymbolicStep
from einf.steps.reduce import ReduceSymbolicStep

from .abstract import AbstractPlan
from .symbolic import SymbolicPlan


class PlanStepDict(TypedDict):
    op: str
    equation: NotRequired[str]
    reducer: NotRequired[str]


class PlanDict(TypedDict):
    schema_version: str
    op: str
    lhs: list[str]
    rhs: list[str]
    kind: str
    resolution: Literal["symbolic", "concrete"]
    executable_now: bool
    blockers: list[str]
    ir: list[str]
    steps: list[PlanStepDict]


def build_plan_dict(
    *,
    op_name: str,
    lhs: AxisSide,
    rhs: AxisSide,
    sizes: dict[str, int],
    abstract_plan: AbstractPlan,
) -> PlanDict:
    """Build deterministic plan payload from abstract and symbolic plans."""
    signature = Signature(inputs=lhs, outputs=rhs)
    blockers = _build_plan_blockers(
        lhs=lhs,
        rhs=rhs,
        sizes=sizes,
    )
    candidate = _select_preview_candidate(
        abstract_plan=abstract_plan,
    )
    if candidate is None:
        blockers.append("no symbolic plan candidates")
        kind = "invalid"
        steps: list[PlanStepDict] = [{"op": "invalid"}]
    else:
        steps = _symbolic_steps_to_plan_steps(
            symbolic_plan=candidate,
        )
        kind = _derive_plan_kind(symbolic_plan=candidate, steps=steps)

    return {
        "schema_version": "v1",
        "op": op_name,
        "lhs": [axis_terms.to_dsl() for axis_terms in lhs],
        "rhs": [axis_terms.to_dsl() for axis_terms in rhs],
        "kind": kind,
        "resolution": _plan_resolution(signature=signature, sizes=sizes),
        "executable_now": len(blockers) == 0 and candidate is not None,
        "blockers": blockers,
        "ir": list(abstract_plan.ir_program.node_kinds()),
        "steps": steps,
    }


def render_plan_text(plan: PlanDict) -> str:
    """Render one plan payload as a compact human-readable preview."""
    lhs_text = (
        plan["lhs"][0] if len(plan["lhs"]) == 1 else f"({', '.join(plan['lhs'])})"
    )
    rhs_text = (
        plan["rhs"][0] if len(plan["rhs"]) == 1 else f"({', '.join(plan['rhs'])})"
    )

    lines: list[str] = [
        f"op: {plan['op']}",
        f"signature: {lhs_text} -> {rhs_text}",
        f"kind: {plan['kind']}",
        f"resolution: {plan['resolution']}",
        f"executable_now: {'true' if plan['executable_now'] else 'false'}",
    ]

    blockers = plan["blockers"]
    if blockers:
        lines.append("blockers:")
        for blocker in blockers:
            lines.append(f"- {blocker}")

    steps = plan["steps"]
    ir_kinds = plan["ir"]
    if ir_kinds:
        lines.append(f"ir: {' -> '.join(ir_kinds)}")
    if steps:
        lines.append("steps:")
        for index, step in enumerate(steps, start=1):
            details: list[str] = []
            equation = step.get("equation")
            reducer = step.get("reducer")
            if equation is not None:
                details.append(f"equation={equation}")
            if reducer is not None:
                details.append(f"reducer={reducer}")
            if details:
                lines.append(f"{index}. {step['op']} ({', '.join(details)})")
            else:
                lines.append(f"{index}. {step['op']}")

    return "\n".join(lines)


def _select_preview_candidate(
    *,
    abstract_plan: AbstractPlan,
) -> SymbolicPlan | None:
    """Pick one deterministic symbolic candidate from one abstract plan."""
    if not abstract_plan.symbolic_candidates:
        return None
    return abstract_plan.symbolic_candidates[0]


def _derive_plan_kind(
    *,
    symbolic_plan: SymbolicPlan,
    steps: list[PlanStepDict],
) -> str:
    """Derive displayed plan kind from symbolic candidate and rendered steps."""
    if len(steps) == 1:
        return steps[0]["op"]
    if symbolic_plan.kind:
        return symbolic_plan.kind
    return "invalid"


def _symbolic_steps_to_plan_steps(
    *,
    symbolic_plan: SymbolicPlan,
) -> list[PlanStepDict]:
    """Convert symbolic steps to plan step dictionaries."""
    plan_steps: list[PlanStepDict] = []
    for step in symbolic_plan.steps:
        if isinstance(step, EinsumSymbolicStep):
            equations = step.preview_equations()
            if equations:
                for equation in equations:
                    plan_steps.append({"op": step.name, "equation": equation})
                continue
            plan_steps.append({"op": step.name})
            continue
        if isinstance(step, ReduceSymbolicStep):
            plan_steps.append({"op": "reduce", "reducer": step.reducer_label()})
            continue
        plan_steps.append({"op": step.name})
    return plan_steps


def _build_plan_blockers(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    sizes: dict[str, int],
) -> list[str]:
    """Collect deterministic symbolic blockers for one plan payload."""
    blockers: list[str] = []
    lhs_axis_names, lhs_pack_names = lhs.symbol_names()
    rhs_axis_names, rhs_pack_names = rhs.symbol_names()
    unresolved_sizes = sorted((rhs_axis_names - lhs_axis_names) - set(sizes))
    for name in unresolved_sizes:
        blockers.append(f"missing size binding: {name}")
    unresolved_packs = sorted(rhs_pack_names - lhs_pack_names)
    for pack_name in unresolved_packs:
        blockers.append(f"missing pack binding: {pack_name}")
    return blockers


def _plan_resolution(
    *,
    signature: Signature,
    sizes: dict[str, int],
) -> Literal["symbolic", "concrete"]:
    """Classify plan resolution based on unresolved symbols."""
    if signature.pack_names():
        return "symbolic"
    if signature.axis_names() - set(sizes):
        return "symbolic"
    return "concrete"


__all__ = [
    "PlanDict",
    "PlanStepDict",
    "build_plan_dict",
    "render_plan_text",
]
