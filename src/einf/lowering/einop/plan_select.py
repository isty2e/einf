from einf.diagnostics import ErrorCode, ValidationError
from einf.signature import Signature

from .base_plan import build_einop_execution_plan_base
from .carrier_plan import try_build_carrier_then_unary_plan
from .model import (
    DirectEinsumEinopLoweringPlan,
    EinopChainSearchRequest,
    EinopLoweringPlan,
)
from .search_plan import build_symbolic_einsum_chain_plan


def build_einop_execution_plan(
    *,
    analysis_signature: Signature,
    has_reducer_plan: bool,
) -> EinopLoweringPlan:
    """Build a complete deterministic einop lowering plan.

    Parameters
    ----------
    analysis_signature : Signature
        Canonical input and output axes to lower.
    has_reducer_plan : bool
        Whether the operation supplies an explicit reducer plan.

    Returns
    -------
    EinopLoweringPlan
        The selected executable plan.

    Raises
    ------
    ValidationError
        If the signature cannot be normalized or no lowering is valid.
    """
    execution_plan = build_einop_execution_plan_base(
        analysis_signature=analysis_signature,
        has_reducer_plan=has_reducer_plan,
    )

    if not isinstance(
        execution_plan,
        (DirectEinsumEinopLoweringPlan, EinopChainSearchRequest),
    ):
        return execution_plan

    if isinstance(execution_plan, DirectEinsumEinopLoweringPlan) and (
        len(execution_plan.equations) <= 1 or len(analysis_signature.inputs) == 1
    ):
        return execution_plan

    carrier_plan = try_build_carrier_then_unary_plan(
        analysis_signature=analysis_signature
    )
    if carrier_plan is not None:
        return carrier_plan

    chain_plan = build_symbolic_einsum_chain_plan(
        analysis_signature=analysis_signature,
        tail_builder=lambda signature: build_einop_execution_plan(
            analysis_signature=signature,
            has_reducer_plan=False,
        ),
    )
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
