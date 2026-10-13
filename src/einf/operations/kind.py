from enum import Enum, StrEnum


class OperationKind(StrEnum):
    """Closed set of tensor operation identities."""

    __str__ = Enum.__str__

    def __format__(self, format_spec: str) -> str:
        """Format the qualified member name.

        Parameters
        ----------
        format_spec : str
            String format specification applied to the member name.

        Returns
        -------
        str
            Formatted qualified member name.
        """
        return Enum.__format__(self, format_spec)

    VIEW = "view"
    REARRANGE = "rearrange"
    REPEAT = "repeat"
    REDUCE = "reduce"
    CONTRACT = "contract"
    EINOP = "einop"


__all__ = ["OperationKind"]
