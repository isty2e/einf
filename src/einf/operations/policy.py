from dataclasses import dataclass

from ..axis import AxisSide
from ..diagnostics import ErrorCode, ValidationError
from .kind import OperationKind


@dataclass(frozen=True, slots=True)
class OpPolicy:
    """Canonical constructor/call arity policy for one operation."""

    supports_reducer: bool = False
    unary_call_only: bool = False
    require_unary_lhs: bool = False
    require_unary_rhs: bool = False

    def validate_constructor(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
    ) -> None:
        """Validate operation constructor arity contract."""
        if self.require_unary_lhs and len(lhs) != 1:
            raise ValidationError(
                code=ErrorCode.MULTI_INPUT_NOT_ALLOWED,
                message=(
                    "multi-input not allowed: "
                    f"{op_name} expects exactly one input, got {len(lhs)}"
                ),
                help=f"use unary lhs for {op_name} in v0.1",
                related=(f"{op_name} schema",),
                data={
                    "operation": op_name,
                    "expected": 1,
                    "got": len(lhs),
                },
            )
        if self.require_unary_rhs and len(rhs) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=(
                    f"{op_name} arity mismatch: expected exactly 1 output, got {len(rhs)}"
                ),
                help=f"use {op_name} signatures with exactly one output axis-list",
                related=(f"{op_name} schema",),
                data={
                    "operation": op_name,
                    "expected": 1,
                    "got": len(rhs),
                },
            )

    def validate_call(
        self,
        *,
        op_name: str,
        expected_input_arity: int,
        input_arity: int,
    ) -> None:
        """Validate operation call arity contract."""
        if self.unary_call_only and input_arity > 1:
            raise ValidationError(
                code=ErrorCode.MULTI_INPUT_NOT_ALLOWED,
                message=(
                    "multi-input not allowed: "
                    f"{op_name} expects exactly one input, got {input_arity}"
                ),
                help=f"pass exactly one input tensor to {op_name} in v0.1",
                related=(f"{op_name} schema",),
                data={
                    "operation": op_name,
                    "expected": 1,
                    "got": input_arity,
                },
            )
        if input_arity != expected_input_arity:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=(
                    f"{op_name} arity mismatch: expected {expected_input_arity} inputs, "
                    f"got {input_arity}"
                ),
                help="pass exactly the number of inputs required by this signature",
                related=("TensorOp call contract",),
                data={
                    "operation": op_name,
                    "expected": expected_input_arity,
                    "got": input_arity,
                },
            )


_POLICIES_BY_KIND = {
    OperationKind.VIEW: OpPolicy(unary_call_only=True, require_unary_lhs=True),
    OperationKind.REARRANGE: OpPolicy(),
    OperationKind.REPEAT: OpPolicy(
        unary_call_only=True,
        require_unary_lhs=True,
        require_unary_rhs=True,
    ),
    OperationKind.REDUCE: OpPolicy(
        supports_reducer=True,
        unary_call_only=True,
        require_unary_lhs=True,
        require_unary_rhs=True,
    ),
    OperationKind.CONTRACT: OpPolicy(require_unary_rhs=True),
    OperationKind.EINOP: OpPolicy(supports_reducer=True),
}


def resolve_op_policy(kind: OperationKind, /) -> OpPolicy:
    """Resolve canonical operation policy by closed operation kind."""
    return _POLICIES_BY_KIND[kind]


__all__ = ["OpPolicy", "resolve_op_policy"]
