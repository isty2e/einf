from ..axis import AxisSide, AxisTerms
from .kind import OperationKind
from .tensor_op import TensorOp


def view(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `view` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        kind=OperationKind.VIEW,
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
        kind=OperationKind.REARRANGE,
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
        kind=OperationKind.REPEAT,
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
        kind=OperationKind.REDUCE,
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


def contract(
    lhs: AxisTerms | AxisSide,
    rhs: AxisTerms | AxisSide,
) -> TensorOp:
    """Create a `TensorOp` scaffold for `contract` transforms."""
    lhs_specs = AxisSide.from_spec(lhs, side_name="lhs")
    rhs_specs = AxisSide.from_spec(rhs, side_name="rhs")
    return TensorOp.from_base_spec(
        kind=OperationKind.CONTRACT,
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
        kind=OperationKind.EINOP,
        lhs=lhs_specs,
        rhs=rhs_specs,
    )


__all__ = ["TensorOp", "contract", "einop", "rearrange", "reduce", "repeat", "view"]
