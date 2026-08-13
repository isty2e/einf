from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from einf.axis import AxisSide
from einf.signature import Signature


class LoweringTraceStage(str, Enum):
    """Non-semantic stage label used to explain default lowering."""

    ASSEMBLE = "assemble"
    TRANSFORM = "transform"
    ROUTE = "route"
    GATHER = "gather"


@dataclass(frozen=True, slots=True)
class LoweringSignature:
    """Identify the structural operation consumed by lowering.

    Parameters
    ----------
    op_name : str
        Operation name used to select lowering rules.
    signature : Signature
        Normalized input and output axis signature.
    explicit_sizes_items : tuple[tuple[str, int], ...]
        Canonical explicit axis-size bindings.
    """

    op_name: str
    signature: Signature
    explicit_sizes_items: tuple[tuple[str, int], ...] = ()

    @property
    def input_arity(self) -> int:
        """Return the number of input tensors.

        Returns
        -------
        int
            Number of input tensors declared by the axis signature.
        """
        return self.signature.input_arity

    @property
    def output_arity(self) -> int:
        """Return the number of output tensors.

        Returns
        -------
        int
            Number of output tensors declared by the axis signature.
        """
        return self.signature.output_arity


@dataclass(frozen=True, slots=True)
class IRProgram:
    """Carry one lowering signature and its observability trace.

    Parameters
    ----------
    source : LoweringSignature
        Structural operation consumed by lowering.
    trace : tuple[LoweringTraceStage, ...]
        Non-semantic stages used to render the default lowering path.
    """

    source: LoweringSignature
    trace: tuple[LoweringTraceStage, ...] = ()

    @classmethod
    def from_source(cls, source: LoweringSignature, /) -> "IRProgram":
        """Build an IR program with the source's default trace.

        Parameters
        ----------
        source : LoweringSignature
            Structural operation consumed by lowering.

        Returns
        -------
        IRProgram
            Source-bound IR with the default observability trace.
        """
        return cls(
            source=source,
            trace=_LOWERING_TRACE_BY_OPERATION.get(
                source.op_name,
                _TRANSFORM_TRACE,
            ),
        )

    @property
    def op_name(self) -> str:
        """Return the operation name used by lowering.

        Returns
        -------
        str
            Operation name from the canonical lowering signature.
        """
        return self.source.op_name

    @property
    def lhs(self) -> AxisSide:
        """Return normalized input axis terms.

        Returns
        -------
        AxisSide
            Input side of the canonical axis signature.
        """
        return self.source.signature.inputs

    @property
    def rhs(self) -> AxisSide:
        """Return normalized output axis terms.

        Returns
        -------
        AxisSide
            Output side of the canonical axis signature.
        """
        return self.source.signature.outputs

    @property
    def explicit_sizes_items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical explicit axis-size bindings.

        Returns
        -------
        tuple[tuple[str, int], ...]
            Axis names and explicit sizes in canonical order.
        """
        return self.source.explicit_sizes_items

    @property
    def input_arity(self) -> int:
        """Return the number of input tensors.

        Returns
        -------
        int
            Number of input tensors declared by the source signature.
        """
        return self.source.input_arity

    @property
    def output_arity(self) -> int:
        """Return the number of output tensors.

        Returns
        -------
        int
            Number of output tensors declared by the source signature.
        """
        return self.source.output_arity


_TRANSFORM_TRACE = (LoweringTraceStage.TRANSFORM,)
_COMPOSITE_TRACE = (
    LoweringTraceStage.ASSEMBLE,
    LoweringTraceStage.TRANSFORM,
    LoweringTraceStage.ROUTE,
    LoweringTraceStage.GATHER,
)
_LOWERING_TRACE_BY_OPERATION: Mapping[
    str,
    tuple[LoweringTraceStage, ...],
] = MappingProxyType(
    {
        "view": _COMPOSITE_TRACE,
        "repeat": (
            LoweringTraceStage.ASSEMBLE,
            LoweringTraceStage.TRANSFORM,
            LoweringTraceStage.ROUTE,
        ),
        "rearrange": _COMPOSITE_TRACE,
        "einop": _COMPOSITE_TRACE,
    }
)


__all__ = [
    "IRProgram",
    "LoweringSignature",
    "LoweringTraceStage",
]
