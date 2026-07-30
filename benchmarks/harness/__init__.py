"""Shared benchmark harness components."""

from .backend import BackendSpec
from .case import BenchmarkCase, CaseCalls, DynamicCaseSpec, FixedCaseSpec
from .config import (
    BenchSizes,
    DimensionName,
    DynamicScaleName,
    DynamicTaskConfig,
    FixedTaskConfig,
    ScaleName,
    dynamic_sizes_for_scale,
    fixed_sizes_for_scale,
)
from .generator import TensorGenerator
from .profiler import Profiler
from .render import MarkdownPrinter
from .result import (
    DynamicCaseResult,
    DynamicObservation,
    DynamicRun,
    FixedCaseResult,
    FixedRun,
    PairedComparison,
    TestResult,
    TimingSummary,
    UnavailableRun,
)
from .runner import BenchmarkRunner
from .types import (
    Array,
    BackendName,
    LibraryName,
    NumpyArray,
    NumpyOutput,
    Output,
    Reference,
    Runner,
    torch,
)
from .workload import (
    DimensionMode,
    DynamicShapeWorkload,
    DynamicWorkloadComparison,
    DynamicWorkloadMetadata,
    WorkloadDimension,
)

__all__ = [
    "Array",
    "BackendName",
    "BackendSpec",
    "BenchSizes",
    "BenchmarkCase",
    "BenchmarkRunner",
    "CaseCalls",
    "DimensionMode",
    "DimensionName",
    "DynamicCaseResult",
    "DynamicCaseSpec",
    "DynamicObservation",
    "DynamicRun",
    "DynamicScaleName",
    "DynamicShapeWorkload",
    "DynamicTaskConfig",
    "DynamicWorkloadComparison",
    "DynamicWorkloadMetadata",
    "FixedCaseResult",
    "FixedCaseSpec",
    "FixedRun",
    "FixedTaskConfig",
    "LibraryName",
    "MarkdownPrinter",
    "NumpyArray",
    "NumpyOutput",
    "Output",
    "PairedComparison",
    "Profiler",
    "Reference",
    "Runner",
    "ScaleName",
    "TensorGenerator",
    "TestResult",
    "TimingSummary",
    "UnavailableRun",
    "WorkloadDimension",
    "dynamic_sizes_for_scale",
    "fixed_sizes_for_scale",
    "torch",
]
