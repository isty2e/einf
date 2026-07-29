import pytest

from einf import ax, axes
from einf.axis import AxisSide
from einf.operations.kind import OperationKind
from einf.operations.policy import resolve_op_policy
from einf.operations.tensor_op import TensorOpContract
from einf.reduction.schema import ReducerPhase


def test_every_operation_kind_has_a_canonical_policy() -> None:
    policies = {kind: resolve_op_policy(kind) for kind in OperationKind}

    assert set(policies) == set(OperationKind)


def test_reducer_capability_is_derived_from_operation_kind() -> None:
    reducer_kinds = {
        kind for kind in OperationKind if resolve_op_policy(kind).supports_reducer
    }

    assert reducer_kinds == {OperationKind.REDUCE, OperationKind.EINOP}


def test_contract_rejects_reducer_plan_for_non_reducer_kind() -> None:
    b, c = axes("b", "c")

    with pytest.raises(ValueError, match="view does not support reducer plans"):
        TensorOpContract(
            kind=OperationKind.VIEW,
            lhs=AxisSide.from_spec(ax[b, c], side_name="lhs"),
            rhs=AxisSide.from_spec(ax[b], side_name="rhs"),
            reducer_plan=(ReducerPhase(axes=ax[c], reducer="sum"),),
        )
