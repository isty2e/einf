"""Shared benchmark harness components."""

from .backend import BackendSpec
from .case import BenchmarkCase, CaseCalls, DynamicCaseSpec, FixedCaseSpec
from .config import (
    BenchSizes,
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

__all__ = [
    "Array",
    "BackendName",
    "BackendSpec",
    "BenchSizes",
    "BenchmarkCase",
    "BenchmarkRunner",
    "CaseCalls",
    "DynamicCaseResult",
    "DynamicCaseSpec",
    "DynamicObservation",
    "DynamicRun",
    "DynamicScaleName",
    "DynamicTaskConfig",
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
    "dynamic_sizes_for_scale",
    "fixed_sizes_for_scale",
    "torch",
]
