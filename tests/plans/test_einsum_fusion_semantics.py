from collections.abc import Callable

import numpy as np
import pytest

import einf.steps.einsum.step as einsum_step_module
from einf import ErrorCode, ExecutionError, ax, axes
from einf.axis import AxisSide
from einf.backend import BACKEND_RESOLVER
from einf.plans.fusion import discover_step_fusion
from einf.steps.axis_slice import (
    AxisSliceRuntimeStep,
    build_axis_slice_symbolic_program,
)
from einf.steps.einsum import EinsumRuntimeProgram, EinsumRuntimeStep
from einf.steps.einsum.step import EinsumEquationExecutor
from einf.tensor_types import TensorLike

_EINSUM_EQUATION = "bnd,dj->bnj"
_FALLBACK_TIERS = ("cached", "matmul", "module", "namespace", "opt")
_EXPECTED_FALLBACK_EVENTS = {
    "cached": ("cached",),
    "matmul": ("matmul",),
    "module": ("cached", "module"),
    "namespace": ("cached", "module", "namespace"),
    "opt": ("cached", "module", "namespace", "opt"),
}


def test_runtime_program_rejects_unresolved_native_matmul_admission() -> None:
    with pytest.raises(
        ValueError,
        match="native matmul admissions must belong to the runtime equations",
    ):
        EinsumRuntimeProgram(
            equations=("ij,jk->ik",),
            chain_order=(),
            carrier_index=None,
            native_matmul_equations=frozenset({"ik,kj->ij"}),
        )


def test_runtime_program_rejects_invalid_native_matmul_admission() -> None:
    with pytest.raises(
        ValueError,
        match="native matmul admissions must be semantically matmul-shaped",
    ):
        EinsumRuntimeProgram(
            equations=("ij,jkl->ikl",),
            chain_order=(),
            carrier_index=None,
            native_matmul_equations=frozenset({"ij,jkl->ikl"}),
        )


@pytest.mark.parametrize("malformed_equation", ("i.,.j->ij", "ij,jk- >ik"))
def test_runtime_program_rejects_malformed_native_matmul_admission(
    malformed_equation: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="native matmul admissions must be semantically matmul-shaped",
    ):
        EinsumRuntimeProgram(
            equations=(malformed_equation,),
            chain_order=(),
            carrier_index=None,
            native_matmul_equations=frozenset({malformed_equation}),
        )


def test_runtime_step_does_not_route_malformed_equation_to_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_equation = "i.,.j->ij"
    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    native_matmul = np.matmul
    matmul_calls: list[None] = []

    def record_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        matmul_calls.append(None)
        return native_matmul(lhs, rhs)

    monkeypatch.setattr(np, "matmul", record_matmul)
    runtime_step = EinsumRuntimeStep(
        name="einsum",
        input_arity=2,
        output_arity=1,
        program=EinsumRuntimeProgram(
            equations=(malformed_equation,),
            chain_order=(),
            carrier_index=None,
            native_matmul_equations=frozenset(),
        ),
        backend_profile=profile,
    )

    with pytest.raises(ExecutionError) as error:
        runtime_step.run((left, right))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED
    assert matmul_calls == []


def test_runtime_step_does_not_route_split_arrow_equation_to_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_arrow_equation = "ij,jk- >ik"
    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    native_matmul = np.matmul
    matmul_calls: list[None] = []

    def record_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        matmul_calls.append(None)
        return native_matmul(lhs, rhs)

    monkeypatch.setattr(np, "matmul", record_matmul)
    runtime_step = EinsumRuntimeStep(
        name="einsum",
        input_arity=2,
        output_arity=1,
        program=EinsumRuntimeProgram(
            equations=(split_arrow_equation,),
            chain_order=(),
            carrier_index=None,
            native_matmul_equations=frozenset(),
        ),
        backend_profile=profile,
    )

    with pytest.raises(ExecutionError) as error:
        runtime_step.run((left, right))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS
    assert matmul_calls == []


def _build_runtime_steps(
    *,
    executor: EinsumEquationExecutor,
    allow_native_matmul: bool,
    chain_mode: bool = False,
) -> tuple[EinsumRuntimeStep, AxisSliceRuntimeStep]:
    b, h, w, _d, j = axes("b", "h", "w", "d", "j")
    einsum_step = EinsumRuntimeStep(
        name="einsum",
        input_arity=2,
        output_arity=1,
        program=EinsumRuntimeProgram(
            equations=(_EINSUM_EQUATION,),
            chain_order=(1,) if chain_mode else (),
            carrier_index=0 if chain_mode else None,
            native_matmul_equations=(
                frozenset({_EINSUM_EQUATION}) if allow_native_matmul else frozenset()
            ),
        ),
        backend_profile=executor.profile,
        executor=executor,
    )
    slice_program = build_axis_slice_symbolic_program(
        AxisSide.from_spec(ax[b, (h + w), j], side_name="lhs"),
        AxisSide.from_spec((ax[b, h, j], ax[b, w, j]), side_name="rhs"),
    )
    slice_step = AxisSliceRuntimeStep(
        name="axis_slice",
        input_arity=1,
        output_arity=2,
        program=slice_program,
        explicit_sizes={"h": 2, "w": 1},
        precomputed_split_sizes=(2, 1),
    )
    return einsum_step, slice_step


def _run_unfused(
    *,
    einsum_step: EinsumRuntimeStep,
    slice_step: AxisSliceRuntimeStep,
    tensors: tuple[TensorLike, TensorLike],
) -> tuple[TensorLike, ...]:
    intermediate = einsum_step.run_binary(*tensors)
    return slice_step.run((intermediate,))


def _run_fused(
    *,
    einsum_step: EinsumRuntimeStep,
    slice_step: AxisSliceRuntimeStep,
    tensors: tuple[TensorLike, TensorLike],
) -> tuple[TensorLike, ...]:
    fusion = discover_step_fusion((einsum_step, slice_step))
    assert fusion is not None
    assert fusion.name == "einsum_axis_slice"
    return fusion.tuple_runner(tensors)


@pytest.mark.parametrize("successful_tier", _FALLBACK_TIERS)
def test_einsum_axis_slice_fusion_preserves_fallback_order(
    successful_tier: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    expected = np.einsum(_EINSUM_EQUATION, left, right)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    events: list[str] = []

    def tier_result(tier: str) -> np.ndarray:
        events.append(tier)
        if tier != successful_tier:
            raise RuntimeError(f"{tier} unavailable")
        return expected

    def cached_contract_expression(
        _equation: str,
        _operand_shapes: tuple[tuple[int, ...], ...],
    ) -> Callable[..., TensorLike]:
        if successful_tier != "cached":
            events.append("cached")
            raise RuntimeError("cached unavailable")

        def run_cached(*_operands: TensorLike) -> TensorLike:
            return tier_result("cached")

        return run_cached

    def module_einsum(
        _equation: str,
        *_operands: TensorLike,
    ) -> TensorLike:
        return tier_result("module")

    def namespace_einsum(
        _equation: str,
        *_operands: TensorLike,
    ) -> TensorLike:
        return tier_result("namespace")

    def module_matmul(_lhs: TensorLike, _rhs: TensorLike) -> TensorLike:
        return tier_result("matmul")

    def opt_contract(
        _equation: str,
        *_operands: TensorLike,
        optimize: str,
    ) -> TensorLike:
        assert optimize == "auto"
        return tier_result("opt")

    monkeypatch.setattr(
        einsum_step_module,
        "_cached_contract_expression",
        cached_contract_expression,
    )
    monkeypatch.setattr(einsum_step_module.opt_einsum, "contract", opt_contract)

    executor = EinsumEquationExecutor(
        profile=profile,
        namespace_einsum=namespace_einsum,
        native_module_einsum=module_einsum,
        native_module_matmul=module_matmul,
    )
    steps = _build_runtime_steps(
        executor=executor,
        allow_native_matmul=successful_tier == "matmul",
    )
    tensors = (left, right)

    unfused_outputs = _run_unfused(
        einsum_step=steps[0],
        slice_step=steps[1],
        tensors=tensors,
    )
    unfused_events = tuple(events)
    assert unfused_events == _EXPECTED_FALLBACK_EVENTS[successful_tier]
    events.clear()
    fused_outputs = _run_fused(
        einsum_step=steps[0],
        slice_step=steps[1],
        tensors=tensors,
    )

    assert tuple(events) == unfused_events
    for fused_output, unfused_output in zip(
        fused_outputs,
        unfused_outputs,
        strict=True,
    ):
        np.testing.assert_array_equal(fused_output, unfused_output)


@pytest.mark.parametrize("chain_mode", [False, True])
def test_einsum_axis_slice_fusion_preserves_error_mapping(
    chain_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    events: list[str] = []

    def fail_cached_expression(
        _equation: str,
        _operand_shapes: tuple[tuple[int, ...], ...],
    ) -> Callable[..., TensorLike]:
        events.append("cached")
        raise RuntimeError("backend failure")

    def fail_module_einsum(
        _equation: str,
        *_operands: TensorLike,
    ) -> TensorLike:
        events.append("module")
        raise RuntimeError("backend failure")

    def fail_namespace_einsum(
        _equation: str,
        *_operands: TensorLike,
    ) -> TensorLike:
        events.append("namespace")
        raise RuntimeError("backend failure")

    def fail_opt_contract(
        _equation: str,
        *_operands: TensorLike,
        optimize: str,
    ) -> TensorLike:
        assert optimize == "auto"
        events.append("opt")
        raise RuntimeError("backend failure")

    monkeypatch.setattr(
        einsum_step_module,
        "_cached_contract_expression",
        fail_cached_expression,
    )
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        fail_opt_contract,
    )

    executor = EinsumEquationExecutor(
        profile=profile,
        namespace_einsum=fail_namespace_einsum,
        native_module_einsum=fail_module_einsum,
        native_module_matmul=None,
    )
    steps = _build_runtime_steps(
        executor=executor,
        allow_native_matmul=False,
        chain_mode=chain_mode,
    )
    tensors = (left, right)

    with pytest.raises(ExecutionError) as unfused_error:
        _run_unfused(
            einsum_step=steps[0],
            slice_step=steps[1],
            tensors=tensors,
        )
    unfused_events = tuple(events)
    assert unfused_events == ("cached", "module", "namespace", "opt")
    events.clear()
    with pytest.raises(ExecutionError) as fused_error:
        _run_fused(
            einsum_step=steps[0],
            slice_step=steps[1],
            tensors=tensors,
        )

    assert tuple(events) == unfused_events
    assert fused_error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED
    assert fused_error.value.message == unfused_error.value.message
    assert fused_error.value.help == unfused_error.value.help
    assert fused_error.value.related == unfused_error.value.related
    assert fused_error.value.data == unfused_error.value.data


def test_einsum_axis_slice_fusion_preserves_einsum_output_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    invalid_intermediate = np.zeros((2, 2, 4))

    def cached_contract_expression(
        _equation: str,
        _operand_shapes: tuple[tuple[int, ...], ...],
    ) -> Callable[..., TensorLike]:
        def run_cached(*_operands: TensorLike) -> TensorLike:
            return invalid_intermediate

        return run_cached

    monkeypatch.setattr(
        einsum_step_module,
        "_cached_contract_expression",
        cached_contract_expression,
    )
    executor = EinsumEquationExecutor(profile=profile)
    steps = _build_runtime_steps(
        executor=executor,
        allow_native_matmul=False,
    )
    tensors = (left, right)

    with pytest.raises(ExecutionError) as unfused_error:
        _run_unfused(
            einsum_step=steps[0],
            slice_step=steps[1],
            tensors=tensors,
        )
    with pytest.raises(ExecutionError) as fused_error:
        _run_fused(
            einsum_step=steps[0],
            slice_step=steps[1],
            tensors=tensors,
        )

    assert fused_error.value.code == ErrorCode.INCONSISTENT_DIMS
    assert fused_error.value.data["operation"] == "einsum"
    assert fused_error.value.message == unfused_error.value.message
    assert fused_error.value.help == unfused_error.value.help
    assert fused_error.value.related == unfused_error.value.related
    assert fused_error.value.data == unfused_error.value.data


def test_single_einsum_step_is_not_registered_as_pass_through_fusion() -> None:
    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    profile = BACKEND_RESOLVER.resolve(left, right, op_name="contract")
    executor = EinsumEquationExecutor(profile=profile)
    einsum_step, _ = _build_runtime_steps(
        executor=executor,
        allow_native_matmul=False,
    )

    assert discover_step_fusion((einsum_step,)) is None
