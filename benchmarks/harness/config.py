from dataclasses import dataclass
from typing import Literal

ScaleName = Literal["small", "medium", "large"]
DynamicScaleName = Literal["medium", "large"]
DimensionName = Literal["b", "n", "d", "h", "w", "r", "j"]
DIMENSION_NAMES: tuple[DimensionName, ...] = ("b", "n", "d", "h", "w", "r", "j")


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

    def __post_init__(self) -> None:
        """Reject non-positive benchmark dimensions."""
        for name, value in self.items():
            if type(value) is not int:
                raise TypeError(f"benchmark dimension {name} must be an int")
            if value < 1:
                raise ValueError(f"benchmark dimension {name} must be positive")

    def items(self) -> tuple[tuple[DimensionName, int], ...]:
        """Return dimensions in canonical benchmark order."""
        return tuple((name, self.value(name)) for name in DIMENSION_NAMES)

    def value(self, name: DimensionName, /) -> int:
        """Return one named benchmark dimension."""
        if name == "b":
            return self.b
        if name == "n":
            return self.n
        if name == "d":
            return self.d
        if name == "h":
            return self.h
        if name == "w":
            return self.w
        if name == "r":
            return self.r
        if name == "j":
            return self.j
        raise ValueError(f"unsupported benchmark dimension: {name}")


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Shared benchmark task configuration."""

    scale: str
    seed: int


@dataclass(frozen=True, slots=True)
class FixedTaskConfig(TaskConfig):
    """Fixed-shape benchmark configuration."""

    rounds: int
    warmup: int
    repeats: int
    iterations: int


@dataclass(frozen=True, slots=True)
class DynamicTaskConfig(TaskConfig):
    """Dynamic-shape benchmark configuration."""

    batches: int
    warmup_batches: int
    repeats: int
    rounds: int
    round_order_seed: int


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
