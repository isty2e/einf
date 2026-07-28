from enum import Enum


class OperationKind(str, Enum):
    """Closed set of tensor operation identities."""

    VIEW = "view"
    REARRANGE = "rearrange"
    REPEAT = "repeat"
    REDUCE = "reduce"
    CONTRACT = "contract"
    EINOP = "einop"


__all__ = ["OperationKind"]
