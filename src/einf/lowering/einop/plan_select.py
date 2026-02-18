from einf.diagnostics import ErrorCode, ValidationError
from einf.signature import Signature

from .base_plan import build_einop_execution_plan_base
from .carrier_plan import try_build_carrier_then_unary_plan
from .model import EinopLoweringPlan
from .search_plan import build_symbolic_einsum_chain_plan


def build_einop_execution_plan(
    *,
    analysis_signature: Signature,
    has_reducer_plan: bool,
) -> EinopLoweringPlan:
    """Build deterministic einop lowering plan with carrier-first chain lifting."""
    execution_plan = build_einop_execution_plan_base(
        analysis_signature=analysis_signature,
        has_reducer_plan=has_reducer_plan,
    )

    if execution_plan.kind not in {"einsum", "search_chain"}:
        return execution_plan

    if execution_plan.kind == "einsum" and len(execution_plan.equations) <= 1:
        return execution_plan

    carrier_plan = try_build_carrier_then_unary_plan(
        analysis_signature=analysis_signature
    )
    if carrier_plan is not None:
        return carrier_plan

    chain_plan = build_symbolic_einsum_chain_plan(analysis_signature=analysis_signature)
    if chain_plan is not None:
        return chain_plan

    raise ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: einop exhaustive chain search found no valid lowering",
        help=(
            "ensure the signature is representable via non-view staged lowering "
            "(einsum/contract/reduce/repeat/rearrange)"
        ),
        related=("einop lowering",),
        data={"operation": "einop"},
    )


__all__ = [
    "build_einop_execution_plan",
]
