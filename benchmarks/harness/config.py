from dataclasses import dataclass
from typing import Literal

from .types import BackendName

ScaleName = Literal["small", "medium", "large"]
DynamicScaleName = Literal["medium", "large"]


@dataclass(frozen=True, slots=True)
class BenchSizes:
    """Shape constants for one benchmark scale."""

    b: int
    n: int
    d: int
    h: int
    w: int
    r: int
    j: int


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Shared benchmark task configuration."""

    backend: BackendName
    scale: str
    seed: int


@dataclass(frozen=True, slots=True)
class FixedTaskConfig(TaskConfig):
    """Fixed-shape benchmark configuration."""

    rounds: int
    cold_repeats: int
    warmup: int
    warm_repeats: int
    warm_iterations: int


@dataclass(frozen=True, slots=True)
class DynamicTaskConfig(TaskConfig):
    """Dynamic-shape benchmark configuration."""

    batches: int
    warmup_batches: int
    repeats: int
    rounds: int
    round_order_seed: int
    parity_checks: int


def fixed_sizes_for_scale(scale: ScaleName) -> BenchSizes:
    """Resolve fixed benchmark sizes from named scale."""
    if scale == "small":
        return BenchSizes(
            b=8,
            n=96,
            d=64,
            h=24,
            w=16,
            r=12,
            j=80,
        )
    if scale == "medium":
        return BenchSizes(
            b=16,
            n=192,
            d=96,
            h=32,
            w=24,
            r=16,
            j=128,
        )
    if scale == "large":
        return BenchSizes(
            b=24,
            n=384,
            d=128,
            h=48,
            w=32,
            r=24,
            j=192,
        )
    raise ValueError(f"unsupported scale: {scale}")


def dynamic_sizes_for_scale(scale: DynamicScaleName) -> BenchSizes:
    """Resolve dynamic benchmark sizes from named scale."""
    if scale == "medium":
        return BenchSizes(
            b=16,
            n=192,
            d=96,
            h=32,
            w=24,
            r=16,
            j=128,
        )
    if scale == "large":
        return BenchSizes(
            b=24,
            n=384,
            d=128,
            h=48,
            w=32,
            r=24,
            j=192,
        )
    raise ValueError(f"unsupported scale: {scale}")
