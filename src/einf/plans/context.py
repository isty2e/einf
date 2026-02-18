from dataclasses import dataclass

from einf.axis import Axis, AxisPack, AxisTerms, ScalarAxisTermBase, ScalarAxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.signature import Signature
from einf.solver import solve_dimensions
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class PlanSelectionContext:
    """Shape-only candidate-selection context."""

    input_shapes: tuple[tuple[int, ...], ...]
    explicit_sizes: dict[str, int]


@dataclass(frozen=True, slots=True)
class RuntimeExecutionContext:
    """Canonical runtime context derived from one signature and input shapes."""

    axis_sizes: dict[str, int]
    pack_sizes: dict[str, tuple[int, ...]]
    lhs_terms: tuple[ScalarAxisTerms, ...]
    rhs_terms: tuple[ScalarAxisTerms, ...]
    pack_ranks: tuple[tuple[str, int], ...]


def _pack_axis_name(pack_name: str, index: int) -> str:
    """Build stable synthetic scalar-axis name for one pack position."""
    return f"_einf_pack_{pack_name}_{index}"


def _validate_internal_pack_axis_collisions(
    *,
    axis_sizes: dict[str, int],
    pack_sizes: dict[str, tuple[int, ...]],
) -> None:
    """Reject user axis names that collide with internal synthetic pack names."""
    generated_names = {
        _pack_axis_name(pack_name, index)
        for pack_name, shape in pack_sizes.items()
        for index in range(len(shape))
    }
    collisions = tuple(sorted(set(axis_sizes) & generated_names))
    if collisions:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                "inconsistent dims: user axis names collide with internal pack expansion names"
            ),
            help="avoid axis names starting with '_einf_pack_'",
            related=("pack expansion",),
            data={"count": len(collisions)},
        )


def expand_pack_terms(
    axis_terms: AxisTerms,
    pack_sizes: dict[str, tuple[int, ...]],
    axis_sizes: dict[str, int],
) -> ScalarAxisTerms:
    """Expand pack terms to deterministic synthetic scalar axes."""
    expanded: list[ScalarAxisTermBase] = []
    for term in axis_terms:
        if isinstance(term, AxisPack):
            for index, dim in enumerate(pack_sizes[term.name]):
                name = _pack_axis_name(term.name, index)
                axis_sizes[name] = dim
                expanded.append(Axis(name))
            continue
        if isinstance(term, ScalarAxisTermBase):
            expanded.append(term)
            continue
        raise TypeError("unsupported axis term in pack expansion")
    return ScalarAxisTerms(tuple(expanded))


def build_runtime_execution_context(
    *,
    signature: Signature,
    tensors: tuple[TensorLike, ...],
    explicit_sizes: dict[str, int],
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> RuntimeExecutionContext:
    """Solve dimensions and normalize both signature sides to scalar-axis terms."""
    normalized_input_shapes = input_shapes
    if normalized_input_shapes is None:
        normalized_input_shapes = tuple(tensor.shape for tensor in tensors)

    solved = solve_dimensions(
        signature,
        input_shapes=normalized_input_shapes,
        explicit_sizes=explicit_sizes,
    )

    axis_sizes = dict(solved.axis_sizes)
    _validate_internal_pack_axis_collisions(
        axis_sizes=axis_sizes,
        pack_sizes=solved.pack_sizes,
    )

    lhs_terms = tuple(
        expand_pack_terms(axis_list, solved.pack_sizes, axis_sizes)
        for axis_list in signature.inputs
    )
    rhs_terms = tuple(
        expand_pack_terms(axis_list, solved.pack_sizes, axis_sizes)
        for axis_list in signature.outputs
    )
    pack_ranks = tuple(
        (pack_name, len(pack_shape))
        for pack_name, pack_shape in sorted(solved.pack_sizes.items())
    )
    return RuntimeExecutionContext(
        axis_sizes=axis_sizes,
        pack_sizes=dict(solved.pack_sizes),
        lhs_terms=lhs_terms,
        rhs_terms=rhs_terms,
        pack_ranks=pack_ranks,
    )


__all__ = [
    "PlanSelectionContext",
    "RuntimeExecutionContext",
    "build_runtime_execution_context",
    "expand_pack_terms",
]
