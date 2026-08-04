#!/usr/bin/env python3
"""Profile call-path overhead decomposition for core TensorOp workloads.

This script reports per-call median latency and stage-level exclusive timing for:
- rearrange (flatten/split)
- repeat
- reduce
- contract
- einop(contract)

Scenarios:
- fixed medium
- fixed large
- dynamic medium
- dynamic large
"""

import argparse
import hashlib
import importlib
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar, cast
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_info

from benchmarks.guardrail.policy import (
    OVERHEAD_REPORT_SCHEMA_VERSION,
    OverheadCpuAllocationDict,
    OverheadCpuBandwidthLimitDict,
    OverheadCpuWeightHierarchyDict,
    OverheadEnvironmentDict,
    OverheadExecutionResourcesDict,
    OverheadNativeThreadPoolDict,
    OverheadPythonRuntimeDict,
    OverheadTorchThreadsDict,
)
from benchmarks.shared import (
    einf_source_receipt_metadata,
    require_einf_source_root,
    require_stable_einf_source_content,
    version_or_missing,
)
from benchmarks.shared.artifacts import publish_receipt
from einf import TensorLike, ax, axes, contract, einop, rearrange, reduce, repeat

try:
    import torch
except ImportError:
    torch = None

Array = NDArray[np.float32]
BackendName = Literal["numpy", "torch"]
_CgroupVersion = Literal[1, 2]
_TensorFamily = TypeVar("_TensorFamily", bound=TensorLike)

_CGROUP_MEMBERSHIP_PATH = Path("/proc/self/cgroup")
_CGROUP_MOUNTINFO_PATH = Path("/proc/self/mountinfo")

Tensor = TensorLike
TensorBatch = tuple[Tensor, ...]
TensorOutput = Tensor | TensorBatch

STAGES = (
    "__call__",
    "input_shape",
    "solve",
    "normalize_context",
    "backend_checks",
    "plan_select",
    "step_specialize",
    "runner_resolve",
    "fusion",
    "step_run",
    "primitive",
    "kernel",
)

STAGE_TARGETS: dict[str, tuple[str, ...]] = {
    "input_shape": (
        "einf.operations.execution.extract_input_shapes",
        "einf.operations.execution._extract_input_shape",
    ),
    "solve": (
        "einf.solver.api.solve_dimensions",
        "einf.solver.solve_dimensions",
        "einf.steps.context.solve_dimensions",
        "einf.steps.expand.solve.solve_dimensions",
    ),
    "normalize_context": (
        "einf.steps.context.build_runtime_execution_context",
        "einf.steps.context.expand_pack_terms",
    ),
    "backend_checks": (
        "einf.plans.abstract.AbstractPlan.resolve_backend_profile",
        "einf.backend.dispatch.BackendResolver.resolve",
        "einf.backend.dispatch.BackendResolver.lookup",
        "einf.backend.dispatch.BackendPolicy.validate_profile",
    ),
    "plan_select": (
        "einf.plans.abstract.AbstractPlan.select_symbolic_plan",
        "einf.plans.symbolic.SymbolicPlan.score",
    ),
    "step_specialize": (
        "einf.plans.symbolic.SymbolicPlan._specialize_runtime_steps",
        "einf.plans.cache.RuntimeStepSpecializationCache.get",
        "einf.plans.cache.RuntimeStepSpecializationCache.set",
        "einf.steps.einsum.step.EinsumSymbolicStep.specialize",
        "einf.steps.reduce.step.ReduceSymbolicStep.specialize",
        "einf.steps.expand.step.ExpandSymbolicStep.specialize",
        "einf.steps.reshape.step.ReshapeSymbolicStep.specialize",
        "einf.steps.permute.PermuteSymbolicStep.specialize",
        "einf.steps.permute.AxisPermuteSymbolicStep.specialize",
        "einf.steps.concat.ConcatSymbolicStep.specialize",
        "einf.steps.axis_slice.step.AxisSliceSymbolicStep.specialize",
    ),
    "runner_resolve": (
        "einf.plans.abstract.AbstractPlan.resolve_tuple_runner",
        "einf.plans.abstract.AbstractPlan.resolve_single_output_runner",
        "einf.plans.abstract.AbstractPlan._build_runner_cache_key",
        "einf.plans.abstract.AbstractPlan._resolve_runtime_context",
        "einf.plans.abstract.AbstractPlan._select_runtime_symbolic_plan",
        "einf.plans.abstract.AbstractPlan._build_runner_kernel",
        "einf.plans.abstract.AbstractPlan._build_step_chain_runner_kernel",
        "einf.plans.abstract.AbstractPlan._resolve_route_output_indices",
        "einf.plans.runners.RouteRunnerKernel.build_tuple_runner",
        "einf.plans.runners.RouteRunnerKernel.build_single_output_runner",
        "einf.plans.runners.StepChainRunnerKernel.build_tuple_runner",
        "einf.plans.runners.StepChainRunnerKernel.build_single_output_runner",
        "einf.plans.runners.run_runtime_step",
        "einf.plans.cache.RunnerCache.get",
        "einf.plans.cache.RunnerCache.set",
        "einf.plans.cache.RouteOutputIndexCache.get",
        "einf.plans.cache.RouteOutputIndexCache.set",
    ),
    "fusion": (
        "einf.plans.fusion.engine.discover_step_fusion",
        "einf.plans.fusion.engine.discover_step_fusions",
    ),
    "step_run": (
        "einf.operations.execution._execute_abstract_plan",
        "einf.plans.abstract.AbstractPlan.execute",
        "einf.plans.symbolic.SymbolicPlan.execute",
        "einf.plans.symbolic.SymbolicPlan._run_runtime_steps",
    ),
    "primitive": (
        "einf.steps.runtime.bind_runtime_backend",
        "einf.steps.einsum.step.EinsumRuntimeStep.run",
        "einf.steps.einsum.step.EinsumRuntimeStep._run_direct",
        "einf.steps.einsum.step.EinsumRuntimeStep._run_chain",
        "einf.steps.permute.PermuteRuntimeStep.run",
        "einf.steps.permute.PermuteRuntimeStep.run_unary",
        "einf.steps.concat.ConcatRuntimeStep.run",
        "einf.steps.axis_slice.step.AxisSliceRuntimeStep.run",
        "einf.steps.expand.step.ExpandRuntimeStep.run",
        "einf.steps.expand.step.ExpandRuntimeStep.run_unary",
        "einf.steps.expand.runtime.run_expand_program",
        "einf.steps.reshape.step.ReshapeRuntimeStep.run",
        "einf.steps.reshape.step.ReshapeRuntimeStep.run_unary",
        "einf.steps.reshape.step.ReshapeRuntimeStep._run_unary_allow_copy",
        "einf.steps.reshape.step.ReshapeRuntimeStep._run_unary_zero_copy",
        "einf.steps.reshape.runtime.run_reshape_program",
        "einf.steps.reshape.runtime.try_run_reshape_program",
        "einf.steps.reduce.step.ReduceRuntimeStep.run",
        "einf.steps.reduce.step.ReduceRuntimeStep.run_unary",
        "einf.steps.reduce.step.DirectMethodReduceRuntimeProgram.run_unary",
        "einf.steps.reduce.step.NamespaceReduceRuntimeProgram.run_unary",
        "einf.steps.reduce.step.DynamicReduceRuntimeProgram.run_unary",
        "einf.steps.reduce.build.build_reduce_compiled_program",
        "einf.steps.reduce.build._compile_reduce_runtime_phase",
        "einf.steps.reduce.runtime.ReducerCompiler.compile",
        "einf.steps.reduce.runtime.CompiledStringReducer.apply",
        "einf.steps.reduce.runtime.CompiledCallableReducer.apply",
    ),
    "kernel": (
        "einf.steps.einsum.step.opt_einsum.contract",
        "einf.steps.einsum.native.try_native_contract_einsum",
        "einf.steps.reduce.runtime.ReducerRuntimeContext.apply_string_reducer",
        "einf.backend.runtime.BackendArrayOps.reduce",
    ),
}


@dataclass(frozen=True, slots=True)
class BenchSizes:
    b: int
    n: int
    d: int
    h: int
    w: int
    r: int
    j: int


@dataclass(frozen=True, slots=True)
class OverheadCase:
    name: str
    call_repr: str
    build_invoke: Callable[[], Callable[[], TensorOutput]]
    loops: int


@dataclass(frozen=True, slots=True)
class _CgroupMount:
    root: PurePosixPath
    mount_point: Path


@dataclass(frozen=True, slots=True)
class RearrangeSplitDynamicBatch:
    tensor: Tensor
    h: int
    w: int


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    call_repr: str
    loops: int
    unpatched_call_ms: float
    instrumented_call_ms: float
    stage_ms_per_call: dict[str, float]
    residual_ms_per_call: float


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: str
    mode: str
    scale: str
    cases: tuple[CaseResult, ...]


class _ExclusiveStageClock:
    """Compute exclusive stage timing under nested wrapped calls."""

    def __init__(self) -> None:
        self.self_ns: dict[str, int] = {
            stage: 0 for stage in STAGES if stage != "__call__"
        }
        self._stack: list[list[Any]] = []

    def reset(self) -> None:
        for key in self.self_ns:
            self.self_ns[key] = 0
        self._stack.clear()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        frame: list[Any] = [stage, time.perf_counter_ns(), 0]
        self._stack.append(frame)
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            _stage, start_ns, child_ns = self._stack.pop()
            total_ns = end_ns - start_ns
            self_ns = total_ns - child_ns
            self.self_ns[stage] += self_ns
            if self._stack:
                self._stack[-1][2] += total_ns


def _to_backend_tensor(
    array: Array,
    *,
    backend: BackendName,
) -> Tensor:
    """Convert deterministic numpy-generated data to selected backend tensors."""
    if backend == "numpy":
        return array
    if torch is None:
        raise RuntimeError("torch backend selected without torch installed")
    return torch.from_numpy(np.ascontiguousarray(array))


def _touch_tensor(
    tensor: Tensor,
) -> None:
    _ = tensor.shape

    if isinstance(tensor, np.ndarray):
        if tensor.size > 0:
            if tensor.ndim == 0:
                _ = float(tensor.item())
            else:
                _ = float(tensor[(0,) * tensor.ndim])
        return

    if torch is not None and isinstance(tensor, torch.Tensor):
        if tensor.numel() > 0:
            if tensor.ndim == 0:
                _ = float(tensor.item())
            else:
                _ = float(tensor[(0,) * tensor.ndim].item())
        return

    raise TypeError(f"unsupported tensor type: {type(tensor)!r}")


def _touch_output(
    output: TensorOutput,
) -> None:
    """Touch outputs to reduce accidental laziness in timing loops."""
    if isinstance(output, tuple):
        for tensor in output:
            _touch_tensor(tensor)
        return

    _touch_tensor(output)


def _measure_call_median_ms(
    *,
    invoke: Callable[[], TensorOutput],
    loops: int,
    warmup: int,
    repeats: int,
) -> float:
    """Measure median per-call latency without monkeypatch overhead."""
    for _ in range(warmup):
        _touch_output(invoke())

    samples_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(loops):
            _touch_output(invoke())
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / float(loops)
        samples_ms.append(elapsed_ms)
    return statistics.median(samples_ms)


def _resolve_target(target: str) -> tuple[object, str, Callable[..., object]]:
    """Resolve one patch target into parent object, attr name, and callable."""
    parts = target.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        attr_path = parts[index:]
        if not attr_path:
            break

        parent: object = module
        for attr in attr_path[:-1]:
            parent = getattr(parent, attr)
        attr_name = attr_path[-1]
        original = getattr(parent, attr_name)
        if not callable(original):
            raise TypeError(f"target {target!r} is not callable")
        return parent, attr_name, original

    raise ModuleNotFoundError(f"cannot resolve patch target: {target}")


def _resolved_stage_target_names() -> dict[str, tuple[str, ...]]:
    """Resolve the instrumentation coverage used by this profiler process."""
    resolved_by_stage: dict[str, tuple[str, ...]] = {}
    for stage, targets in STAGE_TARGETS.items():
        resolved_names: list[str] = []
        resolved_slots: set[tuple[int, str]] = set()
        for target in targets:
            try:
                parent, attr_name, _ = _resolve_target(target)
            except (AttributeError, ModuleNotFoundError, TypeError):
                continue
            slot = (id(parent), attr_name)
            if slot in resolved_slots:
                continue
            resolved_slots.add(slot)
            resolved_names.append(target)
        if not resolved_names:
            raise RuntimeError(
                f"no instrumentation target resolved for stage {stage!r}"
            )
        resolved_by_stage[stage] = tuple(resolved_names)
    return resolved_by_stage


def _stage_wrapper(
    *,
    stage_clock: _ExclusiveStageClock,
    stage: str,
    original: Callable[..., object],
) -> Callable[..., object]:
    """Build one wrapped callable that records exclusive stage timing."""

    def wrapped(*args: object, **kwargs: object) -> object:
        with stage_clock.measure(stage):
            return original(*args, **kwargs)

    return wrapped


@contextmanager
def _patch_stage_targets(stage_clock: _ExclusiveStageClock) -> Iterator[None]:
    """Patch stage targets to collect exclusive stage timing."""
    with ExitStack() as stack:
        for stage, targets in STAGE_TARGETS.items():
            patched_slots: set[tuple[int, str]] = set()
            for target in targets:
                try:
                    module, attr_name, original = _resolve_target(target)
                except (AttributeError, ModuleNotFoundError, TypeError):
                    continue
                slot = (id(module), attr_name)
                if slot in patched_slots:
                    continue
                patched_slots.add(slot)
                stack.enter_context(
                    patch.object(
                        module,
                        attr_name,
                        _stage_wrapper(
                            stage_clock=stage_clock,
                            stage=stage,
                            original=original,
                        ),
                    )
                )
        yield


def _measure_stage_ms_per_call(
    *,
    invoke: Callable[[], TensorOutput],
    loops: int,
    warmup: int,
) -> tuple[dict[str, float], float, float]:
    """Measure exclusive stage timing (ms/call) using monkeypatch wrappers."""
    stage_clock = _ExclusiveStageClock()
    with _patch_stage_targets(stage_clock):
        for _ in range(warmup):
            _touch_output(invoke())
        stage_clock.reset()
        started = time.perf_counter()
        for _ in range(loops):
            _touch_output(invoke())
        instrumented_call_ms = (time.perf_counter() - started) * 1000.0 / float(loops)

    stage_ms_per_call = {
        stage: stage_clock.self_ns[stage] / float(loops) / 1_000_000.0
        for stage in stage_clock.self_ns
    }
    max_stage_ms = max(stage_ms_per_call.values(), default=0.0)
    if max_stage_ms > instrumented_call_ms * 1.5:
        raise RuntimeError(
            "stage timing sanity check failed: stage exceeds instrumented call by >50%; "
            "check duplicate patch targets"
        )
    residual_ms_per_call = instrumented_call_ms - sum(stage_ms_per_call.values())
    return stage_ms_per_call, instrumented_call_ms, residual_ms_per_call


def _profile_case(case: OverheadCase, /) -> CaseResult:
    """Profile one case using fresh invoke state for each timing regime."""
    warmup = max(20, case.loops // 20)
    unpatched_invoke = case.build_invoke()
    median_call_ms = _measure_call_median_ms(
        invoke=unpatched_invoke,
        loops=case.loops,
        warmup=warmup,
        repeats=5,
    )
    instrumented_invoke = case.build_invoke()
    stage_ms, instrumented_call_ms, residual_ms_per_call = _measure_stage_ms_per_call(
        invoke=instrumented_invoke,
        loops=case.loops,
        warmup=warmup,
    )
    return CaseResult(
        name=case.name,
        call_repr=case.call_repr,
        loops=case.loops,
        unpatched_call_ms=median_call_ms,
        instrumented_call_ms=instrumented_call_ms,
        stage_ms_per_call=stage_ms,
        residual_ms_per_call=residual_ms_per_call,
    )


def _fixed_sizes(scale: str) -> BenchSizes:
    if scale == "medium":
        return BenchSizes(b=16, n=192, d=96, h=32, w=24, r=16, j=128)
    if scale == "large":
        return BenchSizes(b=24, n=384, d=128, h=48, w=32, r=24, j=192)
    raise ValueError(f"unsupported scale: {scale}")


def _dynamic_draw(
    *,
    random_state: np.random.RandomState,
    base: int,
    floor: int = 1,
) -> int:
    low = max(floor, int(base * 0.6))
    high = max(low + 1, int(base * 1.4) + 1)
    return int(random_state.randint(low, high))


def _build_cycled_invoke(
    *,
    op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]],
    batches: list[tuple[_TensorFamily, ...]],
) -> Callable[[], _TensorFamily | tuple[_TensorFamily, ...]]:
    """Build one fresh cycling invoke closure over precomputed tensor batches."""
    cursor = 0

    def invoke() -> _TensorFamily | tuple[_TensorFamily, ...]:
        nonlocal cursor
        payload = batches[cursor]
        cursor = (cursor + 1) % len(batches)
        return op(*payload)

    return invoke


def _build_fixed_unary_invoke(
    *,
    op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]],
    tensor: _TensorFamily,
) -> Callable[[], _TensorFamily | tuple[_TensorFamily, ...]]:
    """Build one fixed unary invoke with operation construction outside timing."""

    def invoke(
        op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]] = op,
    ) -> _TensorFamily | tuple[_TensorFamily, ...]:
        return op(tensor)

    return invoke


def _build_fixed_binary_invoke(
    *,
    op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]],
    lhs: _TensorFamily,
    rhs: _TensorFamily,
) -> Callable[[], _TensorFamily | tuple[_TensorFamily, ...]]:
    """Build one fixed binary invoke with operation construction outside timing."""

    def invoke(
        op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]] = op,
    ) -> _TensorFamily | tuple[_TensorFamily, ...]:
        return op(lhs, rhs)

    return invoke


def _build_prebound_unary_invoke(
    *,
    op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]],
    tensor: _TensorFamily,
) -> Callable[[], _TensorFamily | tuple[_TensorFamily, ...]]:
    """Build one unary invoke with operation and tensor bound as call defaults."""

    def invoke(
        op: Callable[..., _TensorFamily | tuple[_TensorFamily, ...]] = op,
        tensor: _TensorFamily = tensor,
    ) -> _TensorFamily | tuple[_TensorFamily, ...]:
        return op(tensor)

    return invoke


def _build_dynamic_rearrange_split_invoke(
    *,
    split_invokes: tuple[Callable[[], TensorOutput], ...],
) -> Callable[[], TensorOutput]:
    """Build one fresh dynamic split invoke over prebound batch-specific calls."""
    cursor = 0

    def invoke() -> TensorOutput:
        nonlocal cursor
        output = split_invokes[cursor]()
        cursor = (cursor + 1) % len(split_invokes)
        return output

    return invoke


def _fixed_cases(
    *,
    sizes: BenchSizes,
    seed: int,
    scale: str,
    backend: BackendName,
) -> tuple[OverheadCase, ...]:
    random_state = np.random.RandomState(seed)
    x_bhwd_np = random_state.randn(sizes.b, sizes.h, sizes.w, sizes.d).astype(
        np.float32
    )
    x_bhwd = _to_backend_tensor(
        x_bhwd_np,
        backend=backend,
    )
    x_bflatd = _to_backend_tensor(
        x_bhwd_np.reshape((sizes.b, sizes.h * sizes.w, sizes.d)),
        backend=backend,
    )
    x_bd = _to_backend_tensor(
        random_state.randn(sizes.b, sizes.d).astype(np.float32),
        backend=backend,
    )
    x_bnd = _to_backend_tensor(
        random_state.randn(sizes.b, sizes.n, sizes.d).astype(np.float32),
        backend=backend,
    )
    w_dj = _to_backend_tensor(
        random_state.randn(sizes.d, sizes.j).astype(np.float32),
        backend=backend,
    )

    b, n, d, h, w, r, j = axes("b", "n", "d", "h", "w", "r", "j")
    h1_size = sizes.h // 2
    h2_size = sizes.h - h1_size
    h1, h2 = axes("h1", "h2")
    x_split_contract = _to_backend_tensor(
        random_state.randn(sizes.b, (h1_size + h2_size) * sizes.r, sizes.n).astype(
            np.float32
        ),
        backend=backend,
    )
    w_nd = _to_backend_tensor(
        random_state.randn(sizes.n, sizes.d).astype(np.float32),
        backend=backend,
    )

    if scale == "medium":
        loops = {
            "rearrange_flatten": 8_000,
            "rearrange_split": 8_000,
            "repeat": 10_000,
            "reduce": 3_000,
            "contract": 700,
            "einop_contract": 700,
            "einop_contract_split": 300,
        }
    else:
        loops = {
            "rearrange_flatten": 2_500,
            "rearrange_split": 2_500,
            "repeat": 3_000,
            "reduce": 1_000,
            "contract": 250,
            "einop_contract": 250,
            "einop_contract_split": 120,
        }

    return (
        OverheadCase(
            name="rearrange_flatten",
            call_repr="rearrange(ax[b, h, w, d], ax[b, (h * w), d])(x)",
            build_invoke=lambda: _build_fixed_unary_invoke(
                op=rearrange(ax[b, h, w, d], ax[b, (h * w), d]),
                tensor=x_bhwd,
            ),
            loops=loops["rearrange_flatten"],
        ),
        OverheadCase(
            name="rearrange_split",
            call_repr="rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=h, w=w)(x)",
            build_invoke=lambda: _build_fixed_unary_invoke(
                op=rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(
                    h=sizes.h,
                    w=sizes.w,
                ),
                tensor=x_bflatd,
            ),
            loops=loops["rearrange_split"],
        ),
        OverheadCase(
            name="repeat",
            call_repr="repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)",
            build_invoke=lambda: _build_fixed_unary_invoke(
                op=repeat(ax[b, d], ax[b, d, r]).with_sizes(r=sizes.r),
                tensor=x_bd,
            ),
            loops=loops["repeat"],
        ),
        OverheadCase(
            name="reduce",
            call_repr="reduce(ax[b, h, w, d], ax[b, d])(x)",
            build_invoke=lambda: _build_fixed_unary_invoke(
                op=reduce(ax[b, h, w, d], ax[b, d]),
                tensor=x_bhwd,
            ),
            loops=loops["reduce"],
        ),
        OverheadCase(
            name="contract",
            call_repr="contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: _build_fixed_binary_invoke(
                op=contract((ax[b, n, d], ax[d, j]), ax[b, n, j]),
                lhs=x_bnd,
                rhs=w_dj,
            ),
            loops=loops["contract"],
        ),
        OverheadCase(
            name="einop_contract",
            call_repr="einop((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: _build_fixed_binary_invoke(
                op=einop((ax[b, n, d], ax[d, j]), ax[b, n, j]),
                lhs=x_bnd,
                rhs=w_dj,
            ),
            loops=loops["einop_contract"],
        ),
        OverheadCase(
            name="einop_contract_split",
            call_repr=(
                "einop((ax[b, (h1 + h2) * r, n], ax[n, d]), "
                "(ax[b, h1 * r, d], ax[b, h2 * r, d]))"
                ".with_sizes(h1=h1, h2=h2, r=r)(lhs, rhs)"
            ),
            build_invoke=lambda: _build_fixed_binary_invoke(
                op=einop(
                    (ax[b, ((h1 + h2) * r), n], ax[n, d]),
                    (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
                ).with_sizes(h1=h1_size, h2=h2_size, r=sizes.r),
                lhs=x_split_contract,
                rhs=w_nd,
            ),
            loops=loops["einop_contract_split"],
        ),
    )


def _dynamic_batches(
    *,
    sizes: BenchSizes,
    seed: int,
    count: int,
    backend: BackendName,
) -> tuple[dict[str, list[TensorBatch]], list[RearrangeSplitDynamicBatch]]:
    random_state = np.random.RandomState(seed)
    batches: dict[str, list[TensorBatch]] = {
        "rearrange_flatten": [],
        "repeat": [],
        "reduce": [],
        "contract": [],
        "einop_contract": [],
        "einop_contract_split": [],
    }
    split_batches: list[RearrangeSplitDynamicBatch] = []
    h1_size = sizes.h // 2
    h2_size = sizes.h - h1_size
    flat_split_dim = (h1_size + h2_size) * sizes.r
    for _ in range(count):
        b_dim = _dynamic_draw(random_state=random_state, base=sizes.b)
        h_dim = _dynamic_draw(random_state=random_state, base=sizes.h)
        w_dim = _dynamic_draw(random_state=random_state, base=sizes.w)
        d_dim = _dynamic_draw(random_state=random_state, base=sizes.d)
        n_dim = _dynamic_draw(random_state=random_state, base=sizes.n)
        j_dim = _dynamic_draw(random_state=random_state, base=sizes.j)

        x_bhwd = _to_backend_tensor(
            random_state.randn(b_dim, h_dim, w_dim, d_dim).astype(np.float32),
            backend=backend,
        )
        x_split = _to_backend_tensor(
            random_state.randn(b_dim, h_dim * w_dim, d_dim).astype(np.float32),
            backend=backend,
        )
        x_bd = _to_backend_tensor(
            random_state.randn(b_dim, d_dim).astype(np.float32),
            backend=backend,
        )
        lhs = _to_backend_tensor(
            random_state.randn(b_dim, n_dim, d_dim).astype(np.float32),
            backend=backend,
        )
        rhs = _to_backend_tensor(
            random_state.randn(d_dim, j_dim).astype(np.float32),
            backend=backend,
        )
        split_lhs = _to_backend_tensor(
            random_state.randn(b_dim, flat_split_dim, n_dim).astype(np.float32),
            backend=backend,
        )
        split_rhs = _to_backend_tensor(
            random_state.randn(n_dim, d_dim).astype(np.float32),
            backend=backend,
        )

        batches["rearrange_flatten"].append((x_bhwd,))
        split_batches.append(
            RearrangeSplitDynamicBatch(tensor=x_split, h=h_dim, w=w_dim)
        )
        batches["repeat"].append((x_bd,))
        batches["reduce"].append((x_bhwd,))
        batches["contract"].append((lhs, rhs))
        batches["einop_contract"].append((lhs, rhs))
        batches["einop_contract_split"].append((split_lhs, split_rhs))
    return batches, split_batches


def _dynamic_cases(
    *,
    sizes: BenchSizes,
    seed: int,
    scale: str,
    backend: BackendName,
) -> tuple[OverheadCase, ...]:
    batches, split_batches = _dynamic_batches(
        sizes=sizes,
        seed=seed,
        count=48 if scale == "medium" else 36,
        backend=backend,
    )
    b, n, d, h, w, r, j = axes("b", "n", "d", "h", "w", "r", "j")
    h1_size = sizes.h // 2
    h2_size = sizes.h - h1_size
    h1, h2 = axes("h1", "h2")

    if scale == "medium":
        loops = {
            "rearrange_flatten": 4_000,
            "rearrange_split": 4_000,
            "repeat": 5_000,
            "reduce": 2_000,
            "contract": 500,
            "einop_contract": 500,
            "einop_contract_split": 220,
        }
    else:
        loops = {
            "rearrange_flatten": 1_500,
            "rearrange_split": 1_500,
            "repeat": 2_000,
            "reduce": 700,
            "contract": 220,
            "einop_contract": 220,
            "einop_contract_split": 90,
        }

    return (
        OverheadCase(
            name="rearrange_flatten",
            call_repr="rearrange(ax[b, h, w, d], ax[b, (h * w), d])(x)",
            build_invoke=lambda: _build_cycled_invoke(
                op=rearrange(ax[b, h, w, d], ax[b, (h * w), d]),
                batches=batches["rearrange_flatten"],
            ),
            loops=loops["rearrange_flatten"],
        ),
        OverheadCase(
            name="rearrange_split",
            call_repr="rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=h_i, w=w_i)(x)",
            build_invoke=lambda: _build_dynamic_rearrange_split_invoke(
                split_invokes=tuple(
                    _build_prebound_unary_invoke(
                        op=rearrange(
                            ax[b, (h * w), d],
                            ax[b, h, w, d],
                        ).with_sizes(h=batch.h, w=batch.w),
                        tensor=batch.tensor,
                    )
                    for batch in split_batches
                ),
            ),
            loops=loops["rearrange_split"],
        ),
        OverheadCase(
            name="repeat",
            call_repr="repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)",
            build_invoke=lambda: _build_cycled_invoke(
                op=repeat(ax[b, d], ax[b, d, r]).with_sizes(r=sizes.r),
                batches=batches["repeat"],
            ),
            loops=loops["repeat"],
        ),
        OverheadCase(
            name="reduce",
            call_repr="reduce(ax[b, h, w, d], ax[b, d])(x)",
            build_invoke=lambda: _build_cycled_invoke(
                op=reduce(ax[b, h, w, d], ax[b, d]),
                batches=batches["reduce"],
            ),
            loops=loops["reduce"],
        ),
        OverheadCase(
            name="contract",
            call_repr="contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: _build_cycled_invoke(
                op=contract((ax[b, n, d], ax[d, j]), ax[b, n, j]),
                batches=batches["contract"],
            ),
            loops=loops["contract"],
        ),
        OverheadCase(
            name="einop_contract",
            call_repr="einop((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: _build_cycled_invoke(
                op=einop((ax[b, n, d], ax[d, j]), ax[b, n, j]),
                batches=batches["einop_contract"],
            ),
            loops=loops["einop_contract"],
        ),
        OverheadCase(
            name="einop_contract_split",
            call_repr=(
                "einop((ax[b, (h1 + h2) * r, n], ax[n, d]), "
                "(ax[b, h1 * r, d], ax[b, h2 * r, d]))"
                ".with_sizes(h1=h1, h2=h2, r=r)(lhs, rhs)"
            ),
            build_invoke=lambda: _build_cycled_invoke(
                op=einop(
                    (ax[b, ((h1 + h2) * r), n], ax[n, d]),
                    (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
                ).with_sizes(h1=h1_size, h2=h2_size, r=sizes.r),
                batches=batches["einop_contract_split"],
            ),
            loops=loops["einop_contract_split"],
        ),
    )


def _profile_scenario(
    *,
    scenario: str,
    mode: str,
    scale: str,
    seed: int,
    backend: BackendName,
) -> ScenarioResult:
    sizes = _fixed_sizes(scale)
    if mode == "fixed":
        cases = _fixed_cases(
            sizes=sizes,
            seed=seed,
            scale=scale,
            backend=backend,
        )
    elif mode == "dynamic":
        cases = _dynamic_cases(
            sizes=sizes,
            seed=seed,
            scale=scale,
            backend=backend,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    case_results: list[CaseResult] = []
    for case in cases:
        case_results.append(_profile_case(case))
    return ScenarioResult(
        scenario=scenario,
        mode=mode,
        scale=scale,
        cases=tuple(case_results),
    )


def _scenario_markdown(result: ScenarioResult) -> str:
    lines = [
        f"### {result.scenario}",
        "",
        "| Case | Call | Loop count | __call__ ms (unpatched) | __call__ ms (instrumented) | input_shape ms | solve ms | normalize ms | backend ms | plan_select ms | step_specialize ms | runner_resolve ms | fusion ms | step_run ms | primitive ms | kernel ms | residual ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result.cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.name,
                    f"`{case.call_repr}`",
                    str(case.loops),
                    f"{case.unpatched_call_ms:.4f}",
                    f"{case.instrumented_call_ms:.4f}",
                    f"{case.stage_ms_per_call.get('input_shape', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('solve', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('normalize_context', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('backend_checks', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('plan_select', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('step_specialize', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('runner_resolve', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('fusion', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('step_run', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('primitive', 0.0):.4f}",
                    f"{case.stage_ms_per_call.get('kernel', 0.0):.4f}",
                    f"{case.residual_ms_per_call:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _to_json(
    result: tuple[ScenarioResult, ...],
    *,
    backend: BackendName,
    seed: int,
    resolved_stage_targets: dict[str, tuple[str, ...]],
    harness_source_sha256: str,
    subject_source: dict[str, str | bool | None],
    execution_resources: OverheadExecutionResourcesDict,
    python_runtime: OverheadPythonRuntimeDict,
) -> dict[str, object]:
    return {
        "schema_version": OVERHEAD_REPORT_SCHEMA_VERSION,
        "meta": {
            "capture_id": str(uuid4()),
            "harness_source_sha256": harness_source_sha256,
            "subject_source": subject_source,
            "execution_target": {
                "backend": backend,
                "requested_device": "cpu",
                "resolved_device": "cpu",
            },
            "host": _host_metadata(),
            "execution_resources": execution_resources,
            "environment": _environment_metadata(
                backend,
                python_runtime=python_runtime,
            ),
            "seed": seed,
            "stages": list(STAGES),
            "resolved_stage_targets": {
                stage: list(targets)
                for stage, targets in resolved_stage_targets.items()
            },
        },
        "scenarios": [
            {
                "scenario": scenario.scenario,
                "mode": scenario.mode,
                "scale": scenario.scale,
                "cases": [
                    {
                        "name": case.name,
                        "call_repr": case.call_repr,
                        "loops": case.loops,
                        "unpatched_call_ms": case.unpatched_call_ms,
                        "instrumented_call_ms": case.instrumented_call_ms,
                        "stage_ms_per_call": case.stage_ms_per_call,
                        "residual_ms_per_call": case.residual_ms_per_call,
                    }
                    for case in scenario.cases
                ],
            }
            for scenario in result
        ],
    }


def _harness_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _subject_source_metadata() -> dict[str, str | bool | None]:
    return einf_source_receipt_metadata()


def _require_stable_receipt_sources(
    *,
    harness_source_sha256: str,
    subject_content_sha256: str,
) -> None:
    if _harness_source_sha256() != harness_source_sha256:
        raise RuntimeError("overhead profiler source changed during measurement")
    require_stable_einf_source_content(subject_content_sha256)


def _command_stdout(*command: str) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        return (
            _command_stdout("sysctl", "-n", "machdep.cpu.brand_string")
            or _command_stdout("sysctl", "-n", "hw.model")
            or platform.processor()
            or "unknown"
        )
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"model name", "Hardware"}:
                    return value.strip()
        except OSError:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"


def _host_metadata() -> dict[str, str | int]:
    """Return stable host facts required for CPU latency comparison."""
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None:
        raise RuntimeError("cannot determine logical CPU count")
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    cpu_model = _cpu_model()
    if not system or not release or not machine or cpu_model == "unknown":
        raise RuntimeError("cannot determine complete host CPU identity")
    return {
        "system": system,
        "release": release,
        "machine": machine,
        "cpu_model": cpu_model,
        "logical_cpu_count": logical_cpu_count,
    }


def _process_cpu_affinity() -> list[int] | None:
    get_affinity = cast(
        Callable[[int], set[int]] | None,
        getattr(os, "sched_getaffinity", None),
    )
    if get_affinity is None:
        return None
    try:
        affinity = sorted(get_affinity(0))
    except OSError as error:
        raise RuntimeError("cannot determine process CPU affinity") from error
    if not affinity:
        raise RuntimeError("process CPU affinity is empty")
    return affinity


def _decode_mountinfo_path(value: str) -> str:
    """Decode the escapes permitted in procfs mountinfo path fields."""
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _cgroup_cpu_membership(text: str) -> tuple[_CgroupVersion, PurePosixPath] | None:
    """Return the process membership for the controller governing CPU bandwidth."""
    unified_membership: PurePosixPath | None = None
    for raw_line in text.splitlines():
        if not raw_line:
            continue
        hierarchy, separator, remainder = raw_line.partition(":")
        controllers, second_separator, raw_path = remainder.partition(":")
        if not separator or not second_separator:
            raise RuntimeError("cannot parse /proc/self/cgroup")
        membership = PurePosixPath(raw_path)
        if not membership.is_absolute():
            raise RuntimeError("cgroup membership path must be absolute")
        if "cpu" in controllers.split(","):
            return 1, membership
        if hierarchy == "0" and not controllers:
            unified_membership = membership
    if unified_membership is None:
        return None
    return 2, unified_membership


def _cgroup_cpu_mounts(
    text: str,
    *,
    version: _CgroupVersion,
) -> tuple[_CgroupMount, ...]:
    """Return cgroup mounts that can expose the selected CPU controller."""
    mounts: list[_CgroupMount] = []
    for raw_line in text.splitlines():
        before_separator, separator, after_separator = raw_line.partition(" - ")
        if not separator:
            continue
        mount_fields = before_separator.split()
        filesystem_fields = after_separator.split()
        if len(mount_fields) < 6 or len(filesystem_fields) < 3:
            raise RuntimeError("cannot parse /proc/self/mountinfo")
        filesystem_type = filesystem_fields[0]
        if version == 2:
            if filesystem_type != "cgroup2":
                continue
        else:
            mount_options = set(mount_fields[5].split(","))
            super_options = set(filesystem_fields[2].split(","))
            if filesystem_type != "cgroup" or "cpu" not in (
                mount_options | super_options
            ):
                continue

        root = PurePosixPath(_decode_mountinfo_path(mount_fields[3]))
        mount_point = Path(_decode_mountinfo_path(mount_fields[4]))
        if not root.is_absolute() or not mount_point.is_absolute():
            raise RuntimeError("cgroup mount paths must be absolute")
        mounts.append(
            _CgroupMount(
                root=root,
                mount_point=mount_point,
            )
        )
    return tuple(sorted(mounts, key=lambda mount: len(mount.root.parts), reverse=True))


def _candidate_cgroup_directories(
    *,
    membership: PurePosixPath,
    mount: _CgroupMount,
) -> tuple[Path, ...]:
    """Return host- and cgroup-namespace interpretations of one membership."""
    candidates: list[Path] = []
    try:
        relative_membership = membership.relative_to(mount.root)
    except ValueError:
        pass
    else:
        candidates.append(mount.mount_point.joinpath(*relative_membership.parts))

    namespace_relative = mount.mount_point.joinpath(*membership.parts[1:])
    if namespace_relative not in candidates:
        candidates.append(namespace_relative)
    return tuple(candidates)


def _resolve_cgroup_directory(
    *,
    version: _CgroupVersion,
    membership: PurePosixPath,
    mounts: tuple[_CgroupMount, ...],
) -> tuple[Path, Path]:
    """Resolve the process cgroup directory and visible hierarchy root."""
    control_name = "cpu.max" if version == 2 else "cpu.cfs_quota_us"
    existing_candidates: list[tuple[Path, Path]] = []
    for mount in mounts:
        for candidate in _candidate_cgroup_directories(
            membership=membership,
            mount=mount,
        ):
            if not candidate.is_dir():
                continue
            resolved = (candidate, mount.mount_point)
            if (candidate / control_name).is_file():
                return resolved
            existing_candidates.append(resolved)
    if existing_candidates:
        return existing_candidates[0]
    raise RuntimeError("cannot resolve the process CPU cgroup directory")


def _read_optional_control(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"cannot read cgroup CPU control {path.name}") from error
    if not value:
        raise RuntimeError(f"cgroup CPU control {path.name} is empty")
    return value


def _required_control_integer(
    value: str,
    *,
    name: str,
    minimum: int,
) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"cgroup CPU control {name} is not an integer") from error
    if parsed < minimum:
        raise RuntimeError(f"cgroup CPU control {name} is below {minimum}")
    return parsed


def _v2_cpu_bandwidth_limit(directory: Path) -> OverheadCpuBandwidthLimitDict | None:
    raw_max = _read_optional_control(directory / "cpu.max")
    if raw_max is None:
        return None
    max_fields = raw_max.split()
    if len(max_fields) != 2:
        raise RuntimeError("cgroup v2 cpu.max must contain quota and period")
    quota_field, period_field = max_fields
    period_us = _required_control_integer(
        period_field,
        name="cpu.max period",
        minimum=1,
    )
    raw_burst = _read_optional_control(directory / "cpu.max.burst")
    burst_us = (
        0
        if raw_burst is None
        else _required_control_integer(raw_burst, name="cpu.max.burst", minimum=0)
    )
    if quota_field == "max":
        return None
    return OverheadCpuBandwidthLimitDict(
        quota_us=_required_control_integer(
            quota_field,
            name="cpu.max quota",
            minimum=1,
        ),
        period_us=period_us,
        burst_us=burst_us,
    )


def _v1_cpu_bandwidth_limit(directory: Path) -> OverheadCpuBandwidthLimitDict | None:
    raw_quota = _read_optional_control(directory / "cpu.cfs_quota_us")
    raw_period = _read_optional_control(directory / "cpu.cfs_period_us")
    if raw_quota is None and raw_period is None:
        return None
    if raw_quota is None or raw_period is None:
        raise RuntimeError("cgroup v1 CPU quota and period must both be available")
    quota_us = _required_control_integer(
        raw_quota,
        name="cpu.cfs_quota_us",
        minimum=-1,
    )
    period_us = _required_control_integer(
        raw_period,
        name="cpu.cfs_period_us",
        minimum=1,
    )
    raw_burst = _read_optional_control(directory / "cpu.cfs_burst_us")
    burst_us = (
        0
        if raw_burst is None
        else _required_control_integer(
            raw_burst,
            name="cpu.cfs_burst_us",
            minimum=0,
        )
    )
    if quota_us == -1:
        return None
    if quota_us == 0:
        raise RuntimeError("cgroup v1 CPU quota must be positive or -1")
    return OverheadCpuBandwidthLimitDict(
        quota_us=quota_us,
        period_us=period_us,
        burst_us=burst_us,
    )


def _cgroup_cpu_weight(
    directory: Path,
    *,
    version: _CgroupVersion,
) -> int | None:
    control_name = "cpu.weight" if version == 2 else "cpu.shares"
    raw_weight = _read_optional_control(directory / control_name)
    if raw_weight is None:
        return None
    weight = _required_control_integer(
        raw_weight,
        name=control_name,
        minimum=0 if version == 2 else 2,
    )
    maximum = 10_000 if version == 2 else 262_144
    if weight > maximum:
        raise RuntimeError(f"cgroup CPU control {control_name} is above {maximum}")
    return weight


def _cgroup_cpu_controls() -> tuple[
    list[OverheadCpuBandwidthLimitDict],
    OverheadCpuWeightHierarchyDict | None,
]:
    """Return CPU bandwidth and hierarchical weight controls for this process."""
    if platform.system() != "Linux":
        return [], None
    try:
        membership_text = _CGROUP_MEMBERSHIP_PATH.read_text()
        mountinfo_text = _CGROUP_MOUNTINFO_PATH.read_text()
    except OSError as error:
        raise RuntimeError("cannot inspect process cgroup CPU allocation") from error
    membership = _cgroup_cpu_membership(membership_text)
    if membership is None:
        return [], None
    version, cgroup_path = membership
    mounts = _cgroup_cpu_mounts(mountinfo_text, version=version)
    if not mounts:
        raise RuntimeError("cannot locate the process CPU cgroup mount")
    directory, hierarchy_root = _resolve_cgroup_directory(
        version=version,
        membership=cgroup_path,
        mounts=mounts,
    )
    if directory != hierarchy_root and hierarchy_root not in directory.parents:
        raise RuntimeError("resolved CPU cgroup escapes its visible hierarchy")

    limits: set[tuple[int, int, int]] = set()
    weights: list[int] = []
    current = directory
    while True:
        limit = (
            _v2_cpu_bandwidth_limit(current)
            if version == 2
            else _v1_cpu_bandwidth_limit(current)
        )
        if limit is not None:
            limits.add((limit["quota_us"], limit["period_us"], limit["burst_us"]))
        weight = _cgroup_cpu_weight(current, version=version)
        if weight is not None:
            weights.append(weight)
        if current == hierarchy_root:
            break
        current = current.parent

    bandwidth_limits = [
        OverheadCpuBandwidthLimitDict(
            quota_us=quota_us,
            period_us=period_us,
            burst_us=burst_us,
        )
        for quota_us, period_us, burst_us in sorted(limits)
    ]
    weight_hierarchy = (
        OverheadCpuWeightHierarchyDict(
            version=version,
            child_to_root=weights,
        )
        if weights
        else None
    )
    return bandwidth_limits, weight_hierarchy


def _cpu_allocation_metadata() -> OverheadCpuAllocationDict:
    bandwidth_limits, weight_hierarchy = _cgroup_cpu_controls()
    return OverheadCpuAllocationDict(
        process_cpu_affinity=_process_cpu_affinity(),
        cgroup_cpu_bandwidth_limits=bandwidth_limits,
        cgroup_cpu_weight_hierarchy=weight_hierarchy,
    )


def _native_threadpool_metadata(
    backend: BackendName,
) -> list[OverheadNativeThreadPoolDict]:
    threadpools: list[OverheadNativeThreadPoolDict] = []
    for index, raw_threadpool in enumerate(threadpool_info()):
        context = f"native thread pool {index}"
        required_strings: dict[str, str] = {}
        for field_name in ("user_api", "internal_api", "prefix"):
            value = raw_threadpool.get(field_name)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"{context} has no valid {field_name}")
            required_strings[field_name] = value
        if backend == "numpy" and required_strings["user_api"] != "blas":
            continue
        num_threads = raw_threadpool.get("num_threads")
        if type(num_threads) is not int or num_threads < 1:
            raise RuntimeError(f"{context} has no valid num_threads")

        optional_strings: dict[str, str | None] = {}
        for field_name in ("version", "threading_layer", "architecture"):
            value = raw_threadpool.get(field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeError(f"{context} has invalid {field_name}")
            optional_strings[field_name] = value
        threadpools.append(
            OverheadNativeThreadPoolDict(
                user_api=required_strings["user_api"],
                internal_api=required_strings["internal_api"],
                prefix=required_strings["prefix"],
                num_threads=num_threads,
                version=optional_strings["version"],
                threading_layer=optional_strings["threading_layer"],
                architecture=optional_strings["architecture"],
            )
        )
    threadpools.sort(
        key=lambda threadpool: (
            threadpool["user_api"],
            threadpool["internal_api"],
            threadpool["prefix"],
            threadpool["version"] or "",
            threadpool["threading_layer"] or "",
            threadpool["architecture"] or "",
            threadpool["num_threads"],
        )
    )
    return threadpools


def _execution_resources_metadata(
    backend: BackendName,
    *,
    cpu_allocation: OverheadCpuAllocationDict,
) -> OverheadExecutionResourcesDict:
    resources = OverheadExecutionResourcesDict(
        cpu_allocation=cpu_allocation,
        native_threadpools=_native_threadpool_metadata(backend),
    )
    if backend == "torch":
        if torch is None:
            raise RuntimeError("torch backend selected but torch is not installed")
        resources["torch_threads"] = OverheadTorchThreadsDict(
            intra_op=torch.get_num_threads(),
            inter_op=torch.get_num_interop_threads(),
        )
    return resources


def _require_stable_execution_resources(
    expected: OverheadExecutionResourcesDict,
    *,
    backend: BackendName,
) -> None:
    current = _execution_resources_metadata(
        backend,
        cpu_allocation=_cpu_allocation_metadata(),
    )
    if current != expected:
        raise RuntimeError("execution resources changed during overhead measurement")


def _fixed_python_hash_seed() -> int:
    """Validate and return this process's fixed PYTHONHASHSEED setting."""
    raw_seed = os.environ.get("PYTHONHASHSEED")
    if raw_seed is None or raw_seed == "random":
        raise RuntimeError(
            "receipt capture requires a fixed PYTHONHASHSEED in [0, 4294967295]"
        )
    if not raw_seed.isascii() or not raw_seed.isdecimal():
        raise RuntimeError("PYTHONHASHSEED must be an unsigned decimal integer")
    seed = int(raw_seed)
    if not 0 <= seed <= 4_294_967_295:
        raise RuntimeError("PYTHONHASHSEED is outside [0, 4294967295]")
    if sys.flags.ignore_environment:
        raise RuntimeError("Python was started with environment variables disabled")
    expected_randomization = int(seed != 0)
    if sys.flags.hash_randomization != expected_randomization:
        raise RuntimeError(
            "PYTHONHASHSEED does not match the active hash-randomization mode"
        )
    return seed


def _python_debug_build() -> bool | None:
    raw_value = sysconfig.get_config_var("Py_DEBUG")
    if raw_value is None:
        return None
    if raw_value in (0, "0"):
        return False
    if raw_value in (1, "1"):
        return True
    raise RuntimeError("Python Py_DEBUG build setting is not canonical")


def _python_runtime_metadata() -> OverheadPythonRuntimeDict:
    implementation = sys.implementation
    implementation_version = implementation.version
    cache_tag = implementation.cache_tag
    if cache_tag is not None and (not isinstance(cache_tag, str) or not cache_tag):
        raise RuntimeError("Python implementation cache tag is invalid")
    abi_flags = getattr(sys, "abiflags", "")
    if not isinstance(abi_flags, str):
        raise TypeError("Python ABI flags are invalid")
    return OverheadPythonRuntimeDict(
        implementation_name=implementation.name,
        implementation_version=(
            f"{implementation_version.major}.{implementation_version.minor}."
            f"{implementation_version.micro}-{implementation_version.releaselevel}."
            f"{implementation_version.serial}"
        ),
        language_version=platform.python_version(),
        build=sys.version,
        cache_tag=cache_tag,
        abi_flags=abi_flags,
        optimize=sys.flags.optimize,
        debug=sys.flags.debug,
        py_debug=_python_debug_build(),
        hash_seed=_fixed_python_hash_seed(),
        hash_witness=(
            hash("einf-overhead-hash-witness-v1"),
            hash("einf-overhead-hash-witness-v2"),
        ),
    )


def _require_stable_python_runtime(expected: OverheadPythonRuntimeDict) -> None:
    if _python_runtime_metadata() != expected:
        raise RuntimeError(
            "Python runtime settings changed during overhead measurement"
        )


def _dependency_versions(backend: BackendName) -> dict[str, str]:
    dependencies = {
        "numpy": version_or_missing("numpy"),
        "array_api_compat": version_or_missing("array-api-compat"),
        "opt_einsum": version_or_missing("opt_einsum"),
    }
    if backend == "torch":
        dependencies["torch"] = version_or_missing("torch")
    return dependencies


def _environment_metadata(
    backend: BackendName,
    *,
    python_runtime: OverheadPythonRuntimeDict,
) -> OverheadEnvironmentDict:
    """Return the Python runtime and dependencies used by this capture."""
    dependencies = _dependency_versions(backend)
    environment = OverheadEnvironmentDict(
        python=python_runtime,
        numpy=dependencies["numpy"],
        array_api_compat=dependencies["array_api_compat"],
        opt_einsum=dependencies["opt_einsum"],
    )
    if backend == "torch":
        environment["torch"] = dependencies["torch"]
    return environment


def _to_markdown(
    *,
    result: tuple[ScenarioResult, ...],
    backend: BackendName,
) -> str:
    dependencies = _dependency_versions(backend)
    host = _host_metadata()
    lines = [
        "# Overhead Decomposition (einf)",
        "",
        "Per-stage call-path decomposition for representative operations:",
        (
            "- rearrange_flatten / rearrange_split / repeat / reduce / contract / "
            "einop_contract / einop_contract_split"
        ),
        "",
        "Stages are measured as exclusive wrapped-call time (ms/call):",
        "- `input_shape`: TensorLike shape extraction and validation",
        "- `solve`: dim solving",
        "- `normalize_context`: runtime context build + pack expansion",
        "- `backend_checks`: backend/profile checks",
        "- `plan_select`: symbolic-plan scoring and selection",
        "- `step_specialize`: symbolic step specialization + runtime step cache",
        "- `runner_resolve`: runner cache lookup/build and step-chain compilation orchestration",
        "- `fusion`: runtime-step fusion discovery",
        "- `step_run`: symbolic plan execution orchestration",
        "- `primitive`: primitive runtime step execution",
        "- `kernel`: einsum/reducer kernel calls",
        "- `residual`: `instrumented __call__ - covered stage sum`",
        "",
        "## Repro",
        "",
        "```bash",
        "PYTHONHASHSEED=0 python -m benchmarks.profile.overhead_breakdown \\",
        f"  --backend {backend} \\",
        "  --receipt artifacts/bench/raw/overhead-breakdown.json",
        "```",
        "",
        "## Environment",
        "",
        f"- Backend: `{backend}`",
        f"- Host: `{host['system']} {host['release']} {host['machine']}`",
        f"- CPU: `{host['cpu_model']}` ({host['logical_cpu_count']} logical CPUs)",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{dependencies['numpy']}`",
        f"- array-api-compat: `{dependencies['array_api_compat']}`",
        f"- opt_einsum: `{dependencies['opt_einsum']}`",
        *([f"- torch: `{dependencies['torch']}`"] if backend == "torch" else []),
        "",
        "## Results",
        "",
    ]
    for scenario in result:
        lines.append(_scenario_markdown(scenario))

    lines.extend(
        [
            "## Notes",
            "",
            "- `__call__ ms (unpatched)` is measured in a baseline run without monkeypatch wrappers.",
            "- `__call__ ms (instrumented)` is measured in the same monkeypatched run as stage values.",
            "- Stage values are exclusive time per wrapped function group in a monkeypatched run.",
            "- Stage values should be compared against `__call__ ms (instrumented)`.",
            "- `residual ms` is the uncovered remainder after subtracting wrapped stage values from instrumented call time.",
            "- Small negative residual can appear from timer noise or overlap bugs and should stay near zero.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile overhead decomposition for core einf ops.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260215,
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional canonical JSON receipt path; Markdown is written to stdout.",
    )
    parser.add_argument(
        "--expect-einf-source-root",
        type=Path,
        default=None,
        help="Fail unless the imported einf package belongs to this checkout root.",
    )
    args = parser.parse_args()
    backend: BackendName = args.backend
    if backend == "torch" and torch is None:
        raise RuntimeError("torch backend selected but torch is not installed")
    if args.expect_einf_source_root is not None:
        require_einf_source_root(args.expect_einf_source_root)
    resolved_stage_targets = _resolved_stage_target_names()
    receipt_harness_source: str | None = None
    receipt_subject_source: dict[str, str | bool | None] | None = None
    receipt_execution_resources: OverheadExecutionResourcesDict | None = None
    receipt_python_runtime: OverheadPythonRuntimeDict | None = None
    if args.receipt is not None:
        receipt_harness_source = _harness_source_sha256()
        receipt_subject_source = _subject_source_metadata()
        receipt_execution_resources = _execution_resources_metadata(
            backend,
            cpu_allocation=_cpu_allocation_metadata(),
        )
        receipt_python_runtime = _python_runtime_metadata()

    scenario_specs = (
        ("fixed-medium", "fixed", "medium"),
        ("fixed-large", "fixed", "large"),
        ("dynamic-medium", "dynamic", "medium"),
        ("dynamic-large", "dynamic", "large"),
    )
    results = tuple(
        _profile_scenario(
            scenario=scenario,
            mode=mode,
            scale=scale,
            seed=args.seed + offset * 1009,
            backend=backend,
        )
        for offset, (scenario, mode, scale) in enumerate(scenario_specs)
    )

    markdown = _to_markdown(
        result=results,
        backend=backend,
    )
    if args.receipt is not None:
        if (
            receipt_harness_source is None
            or receipt_subject_source is None
            or receipt_execution_resources is None
            or receipt_python_runtime is None
        ):
            raise RuntimeError("receipt experiment identity was not captured")
        subject_content_sha256 = receipt_subject_source["content_sha256"]
        if not isinstance(subject_content_sha256, str):
            raise RuntimeError("receipt subject content digest is unavailable")
        _require_stable_receipt_sources(
            harness_source_sha256=receipt_harness_source,
            subject_content_sha256=subject_content_sha256,
        )
        _require_stable_execution_resources(
            receipt_execution_resources,
            backend=backend,
        )
        _require_stable_python_runtime(receipt_python_runtime)
        receipt_payload = _to_json(
            results,
            backend=backend,
            seed=args.seed,
            resolved_stage_targets=resolved_stage_targets,
            harness_source_sha256=receipt_harness_source,
            subject_source=receipt_subject_source,
            execution_resources=receipt_execution_resources,
            python_runtime=receipt_python_runtime,
        )
        publish_receipt(args.receipt, receipt_payload)
    print(markdown)
    if args.receipt is not None:
        print(f"Wrote receipt: {args.receipt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
