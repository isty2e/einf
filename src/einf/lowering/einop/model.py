from dataclasses import dataclass

from einf.axis import AxisTerms


@dataclass(frozen=True, slots=True)
class EinopLoweringPlan:
    """Cached lowering artifact."""

    kind: str
    equations: tuple[str, ...]
    intermediate: AxisTerms | None = None
    carrier_index: int | None = None
    chain_order: tuple[int, ...] = ()
    tail_kind: str | None = None


__all__ = ["EinopLoweringPlan"]
