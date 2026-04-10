"""Shared output coercion helpers for benchmark runners."""

from benchmarks.harness.types import Array, Output


def as_single_array(output: Output) -> Array:
    """Return one single-array output and reject multi-output tuples."""
    if isinstance(output, tuple):
        if len(output) != 1:
            raise ValueError(
                f"expected single output tensor, got tuple of length {len(output)}"
            )
        return output[0]
    return output
