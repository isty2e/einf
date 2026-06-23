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
import importlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from benchmarks.shared import version_or_missing
from einf import TensorLike, ax, axes, contract, einop, rearrange, reduce, repeat

try:
    import torch
except Exception:
    torch = None

Array = NDArray[np.float32]
BackendName = Literal["numpy", "torch"]
_TensorFamily = TypeVar("_TensorFamily", bound=TensorLike)

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
    stage_ms, instrumented_call_ms, residual_ms_per_call = (
        _measure_stage_ms_per_call(
            invoke=instrumented_invoke,
            loops=case.loops,
            warmup=warmup,
        )
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
            build_invoke=lambda: (
                lambda op=rearrange(ax[b, h, w, d], ax[b, (h * w), d]): op(x_bhwd)
            ),
            loops=loops["rearrange_flatten"],
        ),
        OverheadCase(
            name="rearrange_split",
            call_repr="rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(h=h, w=w)(x)",
            build_invoke=lambda: (
                lambda op=rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(
                    h=sizes.h,
                    w=sizes.w,
                ): op(x_bflatd)
            ),
            loops=loops["rearrange_split"],
        ),
        OverheadCase(
            name="repeat",
            call_repr="repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r)(x)",
            build_invoke=lambda: (
                lambda op=repeat(ax[b, d], ax[b, d, r]).with_sizes(r=sizes.r): op(
                    x_bd
                )
            ),
            loops=loops["repeat"],
        ),
        OverheadCase(
            name="reduce",
            call_repr="reduce(ax[b, h, w, d], ax[b, d])(x)",
            build_invoke=lambda: (
                lambda op=reduce(ax[b, h, w, d], ax[b, d]): op(x_bhwd)
            ),
            loops=loops["reduce"],
        ),
        OverheadCase(
            name="contract",
            call_repr="contract((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: (
                lambda op=contract((ax[b, n, d], ax[d, j]), ax[b, n, j]): op(
                    x_bnd, w_dj
                )
            ),
            loops=loops["contract"],
        ),
        OverheadCase(
            name="einop_contract",
            call_repr="einop((ax[b, n, d], ax[d, j]), ax[b, n, j])(lhs, rhs)",
            build_invoke=lambda: (
                lambda op=einop((ax[b, n, d], ax[d, j]), ax[b, n, j]): op(
                    x_bnd, w_dj
                )
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
            build_invoke=lambda: (
                lambda op=einop(
                    (ax[b, ((h1 + h2) * r), n], ax[n, d]),
                    (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
                ).with_sizes(h1=h1_size, h2=h2_size, r=sizes.r): op(
                    x_split_contract, w_nd
                )
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
                    lambda tensor=batch.tensor,
                    op=rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(
                        h=batch.h,
                        w=batch.w,
                    ): op(tensor)
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
) -> dict[str, object]:
    return {
        "meta": {
            "backend": backend,
            "python": platform.python_version(),
            "numpy": version_or_missing("numpy"),
            "torch": version_or_missing("torch"),
            "einops": version_or_missing("einops"),
            "einx": version_or_missing("einx"),
            "stages": list(STAGES),
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


def _to_markdown(
    *,
    result: tuple[ScenarioResult, ...],
    backend: BackendName,
    raw_output: Path,
) -> str:
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
        f"Raw JSON artifact: `{raw_output}`",
        "",
        "## Repro",
        "",
        "```bash",
        "python -m benchmarks.profile.overhead_breakdown \\",
        f"  --backend {backend} \\",
        "  --output docs/benchmarks/2026-02-18-overhead-breakdown.md \\",
        f"  --raw-output {raw_output}",
        "```",
        "",
        "## Environment",
        "",
        f"- Backend: `{backend}`",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{version_or_missing('numpy')}`",
        f"- torch: `{version_or_missing('torch')}`",
        f"- einops: `{version_or_missing('einops')}`",
        f"- einx: `{version_or_missing('einx')}`",
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
        "--output",
        type=Path,
        default=Path("docs/benchmarks/2026-02-18-overhead-breakdown.md"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("docs/benchmarks/raw/2026-02-18-overhead-breakdown.json"),
    )
    args = parser.parse_args()
    backend: BackendName = args.backend
    if backend == "torch" and torch is None:
        raise RuntimeError("torch backend selected but torch is not installed")

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

    raw_payload = _to_json(results, backend=backend)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw_payload, indent=2) + "\n")

    markdown = _to_markdown(
        result=results,
        backend=backend,
        raw_output=args.raw_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n")
    print(markdown)
    print(f"\nWrote raw artifact: {args.raw_output}")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
