from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import prod

from .config import BenchSizes, DimensionName
from .generator import TensorGenerator

Shape = tuple[int, ...]
ShapeFactory = Callable[[Mapping[DimensionName, int]], tuple[Shape, ...]]


class DimensionMode(str, Enum):
    """How one workload dimension is resolved for each dynamic batch."""

    FIXED = "fixed"
    SAMPLED = "sampled"


@dataclass(frozen=True, slots=True)
class WorkloadDimension:
    """One used workload dimension and its configured value range."""

    name: DimensionName
    mode: DimensionMode
    base: int
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class DynamicWorkloadMetadata:
    """Derived shape and dimension facts for one dynamic workload profile."""

    dimensions: tuple[WorkloadDimension, ...]
    base_input_shapes: tuple[Shape, ...]
    base_output_shapes: tuple[Shape, ...]
    base_input_elements: int
    base_output_elements: int

    @property
    def sampled_dimensions(self) -> tuple[WorkloadDimension, ...]:
        """Return dimensions sampled independently for each generated batch."""
        return tuple(
            dimension
            for dimension in self.dimensions
            if dimension.mode is DimensionMode.SAMPLED
        )

    @property
    def fixed_dimensions(self) -> tuple[WorkloadDimension, ...]:
        """Return dimensions held at their profile base value."""
        return tuple(
            dimension
            for dimension in self.dimensions
            if dimension.mode is DimensionMode.FIXED
        )

    def compare_to(
        self,
        reference: "DynamicWorkloadMetadata",
        /,
        *,
        scale: str,
        reference_scale: str,
    ) -> "DynamicWorkloadComparison":
        """Compare exact dimension and tensor-volume ratios across profiles."""
        current_signature = tuple(
            (dimension.name, dimension.mode) for dimension in self.dimensions
        )
        reference_signature = tuple(
            (dimension.name, dimension.mode) for dimension in reference.dimensions
        )
        if current_signature != reference_signature:
            raise ValueError("workload dimension contracts differ across scales")
        if tuple(map(len, self.base_input_shapes)) != tuple(
            map(len, reference.base_input_shapes)
        ) or tuple(map(len, self.base_output_shapes)) != tuple(
            map(len, reference.base_output_shapes)
        ):
            raise ValueError("workload tensor ranks differ across scales")

        return DynamicWorkloadComparison(
            scale=scale,
            reference_scale=reference_scale,
            dimension_ratios=tuple(
                (
                    current.name,
                    Fraction(current.base, baseline.base),
                )
                for current, baseline in zip(
                    self.dimensions,
                    reference.dimensions,
                    strict=True,
                )
            ),
            base_input_elements_ratio=Fraction(
                self.base_input_elements,
                reference.base_input_elements,
            ),
            base_output_elements_ratio=Fraction(
                self.base_output_elements,
                reference.base_output_elements,
            ),
        )


@dataclass(frozen=True, slots=True)
class DynamicWorkloadComparison:
    """Exact profile ratios for one dynamic workload."""

    scale: str
    reference_scale: str
    dimension_ratios: tuple[tuple[DimensionName, Fraction], ...]
    base_input_elements_ratio: Fraction
    base_output_elements_ratio: Fraction


class _DimensionValues(Mapping[DimensionName, int]):
    def __init__(self, values: Mapping[DimensionName, int]) -> None:
        self._values = dict(values)
        self._accessed: list[DimensionName] = []

    @property
    def accessed(self) -> tuple[DimensionName, ...]:
        return tuple(self._accessed)

    def __getitem__(self, name: DimensionName) -> int:
        value = self._values[name]
        if name not in self._accessed:
            self._accessed.append(name)
        return value

    def __iter__(self) -> Iterator[DimensionName]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True, slots=True)
class DynamicShapeWorkload:
    """Canonical shape contract for dynamic benchmark batches."""

    sampled_dimensions: tuple[DimensionName, ...]
    input_shapes: ShapeFactory
    output_shapes: ShapeFactory

    def __post_init__(self) -> None:
        """Reject duplicate sampling declarations."""
        if len(set(self.sampled_dimensions)) != len(self.sampled_dimensions):
            raise ValueError("sampled workload dimensions must be unique")

    def metadata(self, *, sizes: BenchSizes) -> DynamicWorkloadMetadata:
        """Derive canonical workload metadata from one size profile."""
        values = _DimensionValues(dict(sizes.items()))
        input_shapes = self._validate_shapes(self.input_shapes(values))
        output_shapes = self._validate_shapes(self.output_shapes(values))
        used_dimensions = values.accessed
        unused_sampled = set(self.sampled_dimensions) - set(used_dimensions)
        if unused_sampled:
            raise ValueError(
                "sampled workload dimensions are not used by its shape contract: "
                f"{sorted(unused_sampled)}"
            )

        sampled = frozenset(self.sampled_dimensions)
        dimensions = tuple(
            self._dimension_metadata(
                name=name,
                base=sizes.value(name),
                sampled=name in sampled,
            )
            for name in used_dimensions
        )
        return DynamicWorkloadMetadata(
            dimensions=dimensions,
            base_input_shapes=input_shapes,
            base_output_shapes=output_shapes,
            base_input_elements=sum(prod(shape) for shape in input_shapes),
            base_output_elements=sum(prod(shape) for shape in output_shapes),
        )

    def draw_input_shapes(
        self,
        *,
        sizes: BenchSizes,
        generator: TensorGenerator,
        metadata: DynamicWorkloadMetadata,
    ) -> tuple[Shape, ...]:
        """Draw dimensions and return input shapes for one dynamic batch."""
        resolved_values = dict(sizes.items())
        base_by_name = {
            dimension.name: dimension.base for dimension in metadata.dimensions
        }
        if any(
            dimension.base != sizes.value(dimension.name)
            for dimension in metadata.dimensions
        ):
            raise ValueError("workload metadata does not match the configured sizes")
        for name in self.sampled_dimensions:
            try:
                base = base_by_name[name]
            except KeyError:
                raise ValueError(
                    "workload metadata does not cover every sampled dimension"
                ) from None
            resolved_values[name] = generator.draw_dimension(base=base)

        values = _DimensionValues(resolved_values)
        input_shapes = self._validate_shapes(self.input_shapes(values))
        output_shapes = self._validate_shapes(self.output_shapes(values))
        expected_dimensions = tuple(dimension.name for dimension in metadata.dimensions)
        if values.accessed != expected_dimensions:
            raise ValueError(
                "workload dimension usage must not depend on sampled values"
            )
        if self._shape_ranks(input_shapes) != self._shape_ranks(
            metadata.base_input_shapes
        ) or self._shape_ranks(output_shapes) != self._shape_ranks(
            metadata.base_output_shapes
        ):
            raise ValueError("workload tensor ranks must not depend on sampled values")
        return input_shapes

    @staticmethod
    def _dimension_metadata(
        *,
        name: DimensionName,
        base: int,
        sampled: bool,
    ) -> WorkloadDimension:
        if sampled:
            minimum, maximum = TensorGenerator.dimension_bounds(base=base)
            mode = DimensionMode.SAMPLED
        else:
            minimum = maximum = base
            mode = DimensionMode.FIXED
        return WorkloadDimension(
            name=name,
            mode=mode,
            base=base,
            minimum=minimum,
            maximum=maximum,
        )

    @staticmethod
    def _validate_shapes(shapes: tuple[Shape, ...]) -> tuple[Shape, ...]:
        if not isinstance(shapes, tuple):
            raise TypeError("dynamic workload shapes must be a tuple")
        if not shapes:
            raise ValueError("dynamic workload requires at least one tensor shape")
        for shape in shapes:
            if not isinstance(shape, tuple):
                raise TypeError("each dynamic workload shape must be a tuple")
            for dimension in shape:
                if type(dimension) is not int:
                    raise TypeError("dynamic workload tensor dimensions must be ints")
                if dimension < 1:
                    raise ValueError(
                        "dynamic workload tensor dimensions must be positive"
                    )
        return shapes

    @staticmethod
    def _shape_ranks(shapes: tuple[Shape, ...]) -> tuple[int, ...]:
        return tuple(map(len, shapes))
