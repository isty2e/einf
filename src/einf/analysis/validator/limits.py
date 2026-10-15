from dataclasses import dataclass
from typing import Literal

ValidationLimitResource = Literal["discovery_entries", "file_bytes", "total_bytes"]


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    """Configure opt-in resource limits for one validator invocation.

    Parameters
    ----------
    max_discovery_entries : int or None, optional
        Maximum explicit target attempts plus encountered directory entries.
    max_file_bytes : int or None, optional
        Maximum raw source bytes for any one file.
    max_total_bytes : int or None, optional
        Maximum cumulative raw source bytes captured by the validator.

    Raises
    ------
    TypeError
        If a supplied limit is not an integer, or is a boolean.
    ValueError
        If a supplied limit is not positive.

    Notes
    -----
    Each unspecified limit is unrestricted. These limits do not bound process
    memory, filesystem syscall duration, parser time, or checker-internal work.
    """

    max_discovery_entries: int | None = None
    max_file_bytes: int | None = None
    max_total_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_discovery_entries", self.max_discovery_entries),
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer or None")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ValidationLimitFailure:
    """Describe one resource limit that stopped validation.

    Parameters
    ----------
    resource : ValidationLimitResource
        Resource whose configured limit was exceeded.
    limit : int
        Configured maximum for that resource.
    observed : int
        Entry count, observed file size, or cumulative byte count exceeding it.
    path : str
        Requested discovery root or normalized source file being admitted.
    """

    resource: ValidationLimitResource
    limit: int
    observed: int
    path: str


__all__ = ["ValidationLimitFailure", "ValidationLimitResource", "ValidationLimits"]
