from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from einf.axis import AxisSide


@dataclass(frozen=True, slots=True)
class TransformIR:
    """Canonical transform intent on one `(lhs, rhs)` side pair."""

    op_name: str
    lhs: AxisSide
    rhs: AxisSide
    kind: Literal["transform"] = "transform"


@dataclass(frozen=True, slots=True)
class RouteIR:
    """Tensor routing intent between lowering stages."""

    input_arity: int
    output_arity: int
    reason: str
    kind: Literal["route"] = "route"


@dataclass(frozen=True, slots=True)
class GatherIR:
    """Indexed extraction intent before primitive materialization."""

    input_arity: int
    output_arity: int
    reason: str
    kind: Literal["gather"] = "gather"


@dataclass(frozen=True, slots=True)
class AssembleIR:
    """Macro composition node for deterministic lowering assembly."""

    parts: tuple[Literal["transform", "route", "gather"], ...]
    reason: str
    kind: Literal["assemble"] = "assemble"


IRNode: TypeAlias = TransformIR | RouteIR | GatherIR | AssembleIR


@dataclass(frozen=True, slots=True)
class IRProgram:
    """Lowering IR program for one abstract operation."""

    op_name: str
    lhs: AxisSide
    rhs: AxisSide
    nodes: tuple[IRNode, ...]
    input_arity: int = field(init=False)
    output_arity: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_arity", len(self.lhs))
        object.__setattr__(self, "output_arity", len(self.rhs))

    def node_kinds(self) -> tuple[str, ...]:
        """Return deterministic node-kind sequence for this IR program."""
        return tuple(node.kind for node in self.nodes)


def build_default_ir_program(
    *, op_name: str, lhs: AxisSide, rhs: AxisSide
) -> IRProgram:
    """Build deterministic default IR program from normalized operation sides."""
    input_arity = len(lhs)
    output_arity = len(rhs)
    transform_node = TransformIR(
        op_name=op_name,
        lhs=lhs,
        rhs=rhs,
    )

    if op_name in {"reduce", "contract"}:
        nodes: tuple[IRNode, ...] = (transform_node,)
        return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)

    if op_name == "view":
        nodes = (
            AssembleIR(
                parts=("transform", "route", "gather"),
                reason="view lowering assembly",
            ),
            transform_node,
            RouteIR(
                input_arity=input_arity,
                output_arity=output_arity,
                reason="view routing",
            ),
            GatherIR(
                input_arity=output_arity,
                output_arity=output_arity,
                reason="view output gather",
            ),
        )
        return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)

    if op_name == "repeat":
        nodes = (
            AssembleIR(
                parts=("transform", "route"),
                reason="repeat lowering assembly",
            ),
            transform_node,
            RouteIR(
                input_arity=input_arity,
                output_arity=output_arity,
                reason="repeat routing",
            ),
        )
        return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)

    if op_name == "rearrange":
        nodes = (
            AssembleIR(
                parts=("transform", "route", "gather"),
                reason="rearrange lowering assembly",
            ),
            transform_node,
            RouteIR(
                input_arity=input_arity,
                output_arity=output_arity,
                reason="rearrange routing",
            ),
            GatherIR(
                input_arity=output_arity,
                output_arity=output_arity,
                reason="rearrange output gather",
            ),
        )
        return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)

    if op_name == "einop":
        nodes = (
            AssembleIR(
                parts=("transform", "route", "gather"),
                reason="einop lowering assembly",
            ),
            transform_node,
            RouteIR(
                input_arity=input_arity,
                output_arity=output_arity,
                reason="einop operand routing",
            ),
            GatherIR(
                input_arity=output_arity,
                output_arity=output_arity,
                reason="einop output gather",
            ),
        )
        return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)

    nodes = (transform_node,)
    return IRProgram(op_name=op_name, lhs=lhs, rhs=rhs, nodes=nodes)


__all__ = [
    "AssembleIR",
    "GatherIR",
    "IRNode",
    "IRProgram",
    "RouteIR",
    "TransformIR",
    "build_default_ir_program",
]
