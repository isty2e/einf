from collections.abc import Callable
from dataclasses import dataclass, field

from .config import BenchSizes
from .generator import TensorGenerator
from .types import Array, Reference, Runner
from .workload import DynamicShapeWorkload, DynamicWorkloadMetadata

RunnerFactory = Callable[[], Runner]


@dataclass(frozen=True, slots=True)
class CaseCalls:
    """Display strings for one benchmark case across libraries."""

    einf: str
    einops: str
    einx: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Case definition with call forms and library runner factories."""

    name: str
    description: str
    calls: CaseCalls
    reference: Reference
    make_einf_runner: RunnerFactory
    make_einops_runner: RunnerFactory
    make_einx_runner: RunnerFactory


@dataclass(frozen=True, slots=True)
class FixedCaseSpec:
    """Fixed benchmark case spec with one static input batch."""

    case: BenchmarkCase
    inputs: tuple[Array, ...]


@dataclass(frozen=True, slots=True)
class DynamicCaseSpec:
    """Dynamic benchmark case spec with one canonical shape workload."""

    case: BenchmarkCase
    sizes: BenchSizes
    workload: DynamicShapeWorkload
    workload_metadata: DynamicWorkloadMetadata = field(init=False)

    def __post_init__(self) -> None:
        """Bind workload metadata to this case's configured size profile."""
        object.__setattr__(
            self,
            "workload_metadata",
            self.workload.metadata(sizes=self.sizes),
        )

    def make_batch(self, generator: TensorGenerator, /) -> tuple[Array, ...]:
        """Generate one standard-normal backend batch from the shape workload."""
        input_shapes = self.workload.draw_input_shapes(
            sizes=self.sizes,
            generator=generator,
            metadata=self.workload_metadata,
        )
        return generator.backend_batch(
            tuple(generator.randn_numpy(shape) for shape in input_shapes)
        )
