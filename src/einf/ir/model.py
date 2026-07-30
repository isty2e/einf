from dataclasses import dataclass, field
from enum import Enum

from einf.axis import AxisSide


class LoweringTraceStage(str, Enum):
    """Non-semantic stage label used to explain default lowering."""

    ASSEMBLE = "assemble"
    TRANSFORM = "transform"
    ROUTE = "route"
    GATHER = "gather"


@dataclass(frozen=True, slots=True)
class IRProgram:
    """Canonical lowering input plus a non-semantic observability trace.

    `op_name`, `lhs`, and `rhs` are the authoritative compiler inputs. `trace`
    describes the default lowering path for rendering only and must not affect
    compilation.
    """

    op_name: str
    lhs: AxisSide
    rhs: AxisSide
    trace: tuple[LoweringTraceStage, ...] = ()
    input_arity: int = field(init=False)
    output_arity: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_arity", len(self.lhs))
        object.__setattr__(self, "output_arity", len(self.rhs))


_TRANSFORM_TRACE = (LoweringTraceStage.TRANSFORM,)
_COMPOSITE_TRACE = (
    LoweringTraceStage.ASSEMBLE,
    LoweringTraceStage.TRANSFORM,
    LoweringTraceStage.ROUTE,
    LoweringTraceStage.GATHER,
)
_LOWERING_TRACE_BY_OPERATION = {
    "view": _COMPOSITE_TRACE,
    "repeat": (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
    ),
    "rearrange": _COMPOSITE_TRACE,
    "einop": _COMPOSITE_TRACE,
}


def build_default_ir_program(
    *, op_name: str, lhs: AxisSide, rhs: AxisSide
) -> IRProgram:
    """Build canonical lowering input with its default observability trace."""
    return IRProgram(
        op_name=op_name,
        lhs=lhs,
        rhs=rhs,
        trace=_LOWERING_TRACE_BY_OPERATION.get(op_name, _TRANSFORM_TRACE),
    )


__all__ = [
    "IRProgram",
    "LoweringTraceStage",
    "build_default_ir_program",
]
