from dataclasses import dataclass
from math import prod


@dataclass(frozen=True, slots=True, order=True)
class SymbolicPlanScore:
    """Deterministic symbolic-plan score."""

    peak_einsum_numel: int
    pre_einsum_materialize_numel: int
    post_einsum_materialize_numel: int
    allocation_count: int
    kernel_count: int
    step_count: int


def numel_from_shape(shape: tuple[int, ...]) -> int | None:
    """Return one shape numel or None when shape is not strict int tuple."""
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None
        if dim < 0:
            return None
        dims.append(dim)
    return prod(dims, start=1)


def einsum_peak_numel(
    *,
    equation: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> int | None:
    """Return peak operand/output numel for one einsum equation and operand shapes."""
    try:
        lhs_text, rhs_text = equation.split("->", 1)
    except ValueError:
        return None

    lhs_operands = tuple(text.strip() for text in lhs_text.split(","))
    if len(lhs_operands) != len(operand_shapes):
        return None

    symbol_sizes: dict[str, int] = {}
    operand_numels: list[int] = []
    for operand_spec, shape in zip(lhs_operands, operand_shapes, strict=True):
        if len(operand_spec) != len(shape):
            return None
        operand_numel = numel_from_shape(shape)
        if operand_numel is None:
            return None
        operand_numels.append(operand_numel)

        for symbol, dim in zip(operand_spec, shape, strict=True):
            if symbol in symbol_sizes:
                if symbol_sizes[symbol] != dim:
                    return None
                continue
            symbol_sizes[symbol] = dim

    output_numel = 1
    for symbol in rhs_text.strip():
        dim = symbol_sizes.get(symbol)
        if dim is None:
            return None
        output_numel *= dim

    return max((*operand_numels, output_numel))


def einsum_output_shape(
    *,
    equation: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[int, ...] | None:
    """Return einsum output shape for one equation and operand shapes."""
    try:
        lhs_text, rhs_text = equation.split("->", 1)
    except ValueError:
        return None

    lhs_operands = tuple(text.strip() for text in lhs_text.split(","))
    if len(lhs_operands) != len(operand_shapes):
        return None

    symbol_sizes: dict[str, int] = {}
    for operand_spec, shape in zip(lhs_operands, operand_shapes, strict=True):
        if len(operand_spec) != len(shape):
            return None
        for symbol, dim in zip(operand_spec, shape, strict=True):
            if isinstance(dim, bool) or not isinstance(dim, int):
                return None
            if symbol in symbol_sizes:
                if symbol_sizes[symbol] != dim:
                    return None
                continue
            symbol_sizes[symbol] = dim

    output_shape: list[int] = []
    for symbol in rhs_text.strip():
        dim = symbol_sizes.get(symbol)
        if dim is None:
            return None
        output_shape.append(dim)
    return tuple(output_shape)


__all__ = [
    "SymbolicPlanScore",
    "einsum_output_shape",
    "einsum_peak_numel",
    "numel_from_shape",
]
