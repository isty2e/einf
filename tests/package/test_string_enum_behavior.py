import json
import pickle
from enum import StrEnum

import pytest

from einf import ErrorCode, ReducerName
from einf.ir import LoweringTraceStage
from einf.lowering.einop import EinopPrimitiveRoute
from einf.operations.kind import OperationKind


@pytest.mark.parametrize(
    "member",
    [
        member
        for kind in (
            ErrorCode,
            LoweringTraceStage,
            EinopPrimitiveRoute,
            OperationKind,
            ReducerName,
        )
        for member in kind
    ],
)
def test_string_enum_values_preserve_identity_serialization_and_display(
    member: StrEnum,
) -> None:
    qualified_name = f"{type(member).__name__}.{member.name}"
    value = member.value

    assert member == value
    assert hash(member) == hash(value)
    assert type(member)(value) is member
    assert json.loads(json.dumps(member)) == value
    assert pickle.loads(pickle.dumps(member)) is member

    assert str(member) == qualified_name
    assert f"{member}" == qualified_name
    assert format(member, ">40") == qualified_name.rjust(40)
