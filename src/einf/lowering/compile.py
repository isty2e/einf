from dataclasses import dataclass

from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan

from .builders import (
    build_contract_symbolic_plan,
    build_einop_symbolic_plan,
    build_rearrange_symbolic_plan,
    build_reduce_symbolic_plan,
    build_repeat_symbolic_plan,
    build_view_symbolic_plan,
)


@dataclass(frozen=True, slots=True)
class _IRShapeRule:
    """Required IR node shape contract for one operation."""

    required_nodes: frozenset[str]
    must_start_with_assemble: bool = False


_IR_SHAPE_RULES: dict[str, _IRShapeRule] = {
    "view": _IRShapeRule(
        required_nodes=frozenset({"transform", "route", "gather"}),
        must_start_with_assemble=True,
    ),
    "reduce": _IRShapeRule(required_nodes=frozenset({"transform"})),
    "contract": _IRShapeRule(required_nodes=frozenset({"transform"})),
    "repeat": _IRShapeRule(
        required_nodes=frozenset({"transform", "route"}),
        must_start_with_assemble=True,
    ),
    "rearrange": _IRShapeRule(
        required_nodes=frozenset({"transform", "route", "gather"}),
        must_start_with_assemble=True,
    ),
    "einop": _IRShapeRule(
        required_nodes=frozenset({"transform", "route", "gather"}),
        must_start_with_assemble=True,
    ),
}

_IR_BUILDERS = {
    "view": build_view_symbolic_plan,
    "reduce": build_reduce_symbolic_plan,
    "contract": build_contract_symbolic_plan,
    "repeat": build_repeat_symbolic_plan,
    "rearrange": build_rearrange_symbolic_plan,
    "einop": build_einop_symbolic_plan,
}


def _validate_ir_program_shape(ir_program: IRProgram) -> None:
    """Validate that one IR program satisfies operation-level node-shape rules."""
    rule = _IR_SHAPE_RULES.get(ir_program.op_name)
    if rule is None:
        return

    node_kinds = ir_program.node_kinds()
    node_kind_set = set(node_kinds)
    missing = tuple(sorted(rule.required_nodes - node_kind_set))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "lowering IR is missing required node kinds for "
            f"{ir_program.op_name}: {missing_text}"
        )

    if rule.must_start_with_assemble:
        if not node_kinds or node_kinds[0] != "assemble":
            raise ValueError(
                f"lowering IR must start with 'assemble' for {ir_program.op_name}"
            )


def build_symbolic_candidates_from_ir(
    *,
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> tuple[SymbolicPlan, ...]:
    """Compile one canonical IR program into ordered symbolic-plan candidates."""
    _validate_ir_program_shape(ir_program)
    builder = _IR_BUILDERS.get(ir_program.op_name)
    if builder is None:
        return ()
    return (builder(ir_program, explicit_sizes_items, reducer_plan),)


__all__ = ["build_symbolic_candidates_from_ir"]
