from collections.abc import Callable
from dataclasses import dataclass

from .generator import TensorGenerator
from .types import Array, Reference, Runner

RunnerFactory = Callable[[], Runner]
BatchFactory = Callable[[TensorGenerator], tuple[Array, ...]]


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
    """Dynamic benchmark case spec with per-batch factory."""

    case: BenchmarkCase
    batch_factory: BatchFactory
