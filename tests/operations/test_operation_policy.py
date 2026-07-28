from einf.operations.kind import OperationKind
from einf.operations.policy import resolve_op_policy


def test_every_operation_kind_has_a_canonical_policy() -> None:
    policies = {kind: resolve_op_policy(kind) for kind in OperationKind}

    assert set(policies) == set(OperationKind)


def test_reducer_capability_is_derived_from_operation_kind() -> None:
    reducer_kinds = {
        kind for kind in OperationKind if resolve_op_policy(kind).supports_reducer
    }

    assert reducer_kinds == {OperationKind.REDUCE, OperationKind.EINOP}
