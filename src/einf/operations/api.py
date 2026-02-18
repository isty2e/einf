from ..axis import AxisSide, AxisTerms
from ..signature import Signature
from ..steps.einsum import validate_contract_atomic_terms
from .tensor_op import TensorOp


def view(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `view` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        name="view",
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


def rearrange(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `rearrange` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        name="rearrange",
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


def repeat(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `repeat` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        name="repeat",
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


def reduce(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `reduce` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        name="reduce",
        lhs=lhs_specs,
        rhs=rhs_specs,
        supports_reducer=True,
    )


def contract(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `contract` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    validate_contract_atomic_terms(Signature(inputs=lhs_specs, outputs=rhs_specs))
    return TensorOp.from_base_spec(
        name="contract",
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


def einop(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `einop` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        name="einop",
        lhs=lhs_specs,
        rhs=rhs_specs,
        supports_reducer=True,
    )


__all__ = ["view", "rearrange", "repeat", "reduce", "contract", "einop"]
