"""Shared helpers for benchmark entrypoints."""

from .metadata import available_libraries, version_or_missing
from .outputs import as_single_array

__all__ = [
    "as_single_array",
    "available_libraries",
    "version_or_missing",
]
