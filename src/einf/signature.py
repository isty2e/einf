from dataclasses import dataclass

from .axis import Axis, AxisSide, AxisTerms


@dataclass(frozen=True, slots=True, init=False)
class Signature:
    """Normalized transform signature with explicit input/output axis-term tuples."""

    inputs: AxisSide
    outputs: AxisSide

    def __init__(
        self,
        inputs: tuple[AxisTerms, ...],
        outputs: tuple[AxisTerms, ...],
    ) -> None:
        object.__setattr__(self, "inputs", AxisSide.coerce(inputs))
        object.__setattr__(self, "outputs", AxisSide.coerce(outputs))

    @property
    def input_arity(self) -> int:
        """Return expected number of input tensors."""
        return len(self.inputs)

    @property
    def output_arity(self) -> int:
        """Return declared number of output tensors."""
        return len(self.outputs)

    def is_atomic(self) -> bool:
        """Return whether every signature term is one atomic axis label."""
        for axis_list in self.inputs:
            for term in axis_list:
                if not isinstance(term, Axis):
                    return False
        for axis_list in self.outputs:
            for term in axis_list:
                if not isinstance(term, Axis):
                    return False
        return True

    def axis_names(self) -> set[str]:
        """Collect scalar axis names referenced by this signature."""
        return self.inputs.axis_names() | self.outputs.axis_names()

    def pack_names(self) -> set[str]:
        """Collect axis-pack names referenced by this signature."""
        return self.inputs.pack_names() | self.outputs.pack_names()

    def filter_explicit_sizes(
        self,
        explicit_sizes: dict[str, int],
        /,
    ) -> dict[str, int]:
        """Keep only explicit sizes referenced by this signature."""
        referenced_axes = self.axis_names()
        return {
            axis_name: axis_size
            for axis_name, axis_size in explicit_sizes.items()
            if axis_name in referenced_axes
        }


__all__ = ["Signature"]
