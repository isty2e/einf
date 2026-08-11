from dataclasses import replace

import numpy as np
import pytest

import einf.plans.fusion.rules.structural as structural_module
from einf.axis import AxisTerms
from einf.backend import ArrayNamespace, get_backend_array_ops
from einf.backend.runtime import BackendArrayOps
from einf.diagnostics import ErrorCode, ExecutionError, ValidationError
from einf.plans.fusion import (
    discover_step_fusion,
    discover_step_fusions,
)
from einf.steps.expand import (
    ExpandRuntimeStep,
    build_expand_symbolic_program,
    compile_expand_target_shape_evaluator,
)
from einf.steps.permute import (
    PermuteRuntimeStep,
    build_permute_symbolic_program,
)
from einf.steps.reshape import (
    ReshapeRuntimeStep,
    build_reshape_symbolic_program,
)
from einf.steps.reshape.compile import compile_reshape_target_shape_evaluator
from einf.steps.reshape.constants import ZERO_COPY_REQUIRED_RESHAPE_MODE
from einf.tensor_types import TensorLike


def _build_permute_expand_steps(
    backend_ops: BackendArrayOps,
) -> tuple[PermuteRuntimeStep, ExpandRuntimeStep]:
    permute_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    expand_program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )
    target_shape_evaluator = compile_expand_target_shape_evaluator(
        plan=expand_program,
        explicit_sizes={},
    )
    expand_step = ExpandRuntimeStep(
        name="expand",
        input_arity=1,
        output_arity=1,
        program=expand_program,
        target_shape_evaluator=target_shape_evaluator,
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    return permute_step, expand_step


def _build_portable_permute_expand_steps(
    namespace: ArrayNamespace,
) -> tuple[PermuteRuntimeStep, ExpandRuntimeStep]:
    permute_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0)),
        runtime_backend_ops=None,
        runtime_xp=namespace,
    )
    expand_program = build_expand_symbolic_program(
        AxisTerms.from_spec((3, 2)),
        AxisTerms.from_spec((3, 2, 1)),
    )
    target_shape_evaluator = compile_expand_target_shape_evaluator(
        plan=expand_program,
        explicit_sizes={},
    )
    expand_step = ExpandRuntimeStep(
        name="expand",
        input_arity=1,
        output_arity=1,
        program=expand_program,
        target_shape_evaluator=target_shape_evaluator,
        runtime_backend_ops=None,
        runtime_xp=namespace,
    )
    return permute_step, expand_step


def _build_single_permutation_expand_steps(
    backend_ops: BackendArrayOps,
) -> tuple[PermuteRuntimeStep, ExpandRuntimeStep]:
    permute_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    expand_program = build_expand_symbolic_program(
        AxisTerms.from_spec((3, 2)),
        AxisTerms.from_spec((3, 2, 1)),
    )
    expand_step = ExpandRuntimeStep(
        name="expand",
        input_arity=1,
        output_arity=1,
        program=expand_program,
        target_shape_evaluator=compile_expand_target_shape_evaluator(
            plan=expand_program,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    return permute_step, expand_step


def _build_reshape_fusion_steps(
    backend_ops: BackendArrayOps,
) -> tuple[ReshapeRuntimeStep, ReshapeRuntimeStep]:
    first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
    )
    first_compiled = first_program.compiled
    assert first_compiled is not None
    first_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=first_program.zero_copy_mode,
    )

    second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    second_compiled = second_program.compiled
    assert second_compiled is not None
    second_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=second_program.zero_copy_mode,
    )
    return first_step, second_step


def test_discover_fuses_permute_then_permute() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    first_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 2, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    second_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0, 2)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )

    fusion = discover_step_fusion((first_step, second_step))
    assert fusion is not None
    assert fusion.name == "permute_permute"

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    fused_output = fusion.tuple_runner((tensor,))
    sequential_output = (second_step.run_unary(first_step.run_unary(tensor)),)
    expected = (np.transpose(tensor, (2, 1, 0)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], expected[0])
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_discover_fuses_permute_then_expand() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    permute_step, expand_step = _build_permute_expand_steps(backend_ops)

    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None
    assert fusion.name == "permute_expand"

    tensor = np.arange(6).reshape(3, 2)
    fused_output = fusion.tuple_runner((tensor,))
    sequential_output = (expand_step.run_unary(permute_step.run_unary(tensor)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_native_permute_expand_fusion_skips_intermediate_shape_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    permute_step, expand_step = _build_permute_expand_steps(backend_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    def reject_intermediate_validation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact native fusion must not validate intermediates")

    monkeypatch.setattr(
        structural_module,
        "validate_runtime_output_shape",
        reject_intermediate_validation,
    )

    output = fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert output[0].shape == (3, 2, 1)


def test_permute_expand_fusion_projects_trust_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    permute_step, expand_step = _build_permute_expand_steps(backend_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    def fail_trust_resolution(**_kwargs: object) -> bool:
        raise RuntimeError("adapter trust resolution failed")

    monkeypatch.setattr(
        structural_module,
        "is_trusted_backend_array_ops",
        fail_trust_resolution,
    )

    with pytest.raises(
        ExecutionError, match="adapter trust resolution failed"
    ) as error:
        fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value
    assert error.value.data["operation"] == "expand"


def test_discover_fuses_reshape_then_reshape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    first_step, second_step = _build_reshape_fusion_steps(backend_ops)

    fusion = discover_step_fusion((first_step, second_step))
    assert fusion is not None
    assert fusion.name == "reshape_reshape"

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    fused_output = fusion.tuple_runner((tensor,))
    sequential_output = (second_step.run_unary(first_step.run_unary(tensor)),)
    expected = (np.reshape(tensor, (2, 12)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], expected[0])
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_permute_fusion_rejects_distinct_backend_bindings() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    first_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 2, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    second_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0, 2)),
        runtime_backend_ops=replace(backend_ops),
        runtime_xp=None,
    )

    assert discover_step_fusion((first_step, second_step)) is None


def test_permute_expand_fusion_rejects_distinct_backend_bindings() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    first_step, second_step = _build_permute_expand_steps(backend_ops)
    second_step = replace(
        second_step,
        runtime_backend_ops=replace(backend_ops),
    )

    assert discover_step_fusion((first_step, second_step)) is None


def test_reshape_fusion_rejects_distinct_backend_bindings() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    first_step, second_step = _build_reshape_fusion_steps(backend_ops)
    second_step = replace(
        second_step,
        runtime_backend_ops=replace(backend_ops),
    )

    assert discover_step_fusion((first_step, second_step)) is None


def test_fusion_rejects_distinct_namespace_bindings() -> None:
    class Namespace:
        __name__ = "custom.numpy"

        @staticmethod
        def permute_dims(
            tensor: np.ndarray,
            axes: tuple[int, ...],
        ) -> np.ndarray:
            return np.transpose(tensor, axes)

        @staticmethod
        def expand_dims(tensor: np.ndarray, *, axis: int) -> np.ndarray:
            return np.expand_dims(tensor, axis=axis)

        @staticmethod
        def broadcast_to(
            tensor: np.ndarray,
            shape: tuple[int, ...],
        ) -> np.ndarray:
            return np.broadcast_to(tensor, shape)

    first_step, second_step = _build_portable_permute_expand_steps(
        Namespace(),  # type: ignore[arg-type]
    )
    second_step = replace(
        second_step,
        runtime_xp=Namespace(),  # type: ignore[arg-type]
    )

    assert discover_step_fusion((first_step, second_step)) is None


def test_permute_expand_fusion_falls_back_on_primitive_route_miss() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    class BroadcastOnceMissing:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            tensor: TensorLike,
            shape: tuple[int, ...],
        ) -> TensorLike:
            self.calls += 1
            if self.calls == 1:
                raise AttributeError("fused broadcast route unavailable")
            return backend_ops.broadcast_to(tensor, shape)

    broadcast_once_missing = BroadcastOnceMissing()
    route_ops = replace(backend_ops, broadcast_to=broadcast_once_missing)
    permute_step, expand_step = _build_permute_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    output = fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert output[0].shape == (3, 2, 1)
    assert broadcast_once_missing.calls == 2


def test_permute_expand_fusion_falls_back_after_wrong_route_output_shape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    class FirstBroadcastIsWrong:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            tensor: TensorLike,
            shape: tuple[int, ...],
        ) -> TensorLike:
            self.calls += 1
            if self.calls == 1:
                return np.zeros((99,))
            return backend_ops.broadcast_to(tensor, shape)

    first_broadcast_is_wrong = FirstBroadcastIsWrong()
    route_ops = replace(backend_ops, broadcast_to=first_broadcast_is_wrong)
    permute_step, expand_step = _build_permute_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    output = fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert output[0].shape == (3, 2, 1)
    assert first_broadcast_is_wrong.calls == 2


def test_permute_expand_fusion_preserves_structured_failure() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    original = ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="fused permute invariant failed",
    )

    def fail_broadcast(
        _tensor: TensorLike,
        _shape: tuple[int, ...],
    ) -> TensorLike:
        raise original

    route_ops = replace(backend_ops, broadcast_to=fail_broadcast)  # type: ignore[arg-type]
    permute_step, expand_step = _build_permute_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    with pytest.raises(ValidationError) as error:
        fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert error.value is original


def test_permute_expand_fusion_projects_unexpected_backend_failure() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    def fail_broadcast(
        _tensor: TensorLike,
        _shape: tuple[int, ...],
    ) -> TensorLike:
        raise IndexError("fused broadcast failed")

    route_ops = replace(backend_ops, broadcast_to=fail_broadcast)  # type: ignore[arg-type]
    permute_step, expand_step = _build_permute_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    with pytest.raises(ExecutionError) as sequential_error:
        expand_step.run_unary(permute_step.run_unary(np.arange(6).reshape(3, 2)))
    with pytest.raises(ExecutionError) as fused_error:
        fusion.tuple_runner((np.arange(6).reshape(3, 2),))

    assert fused_error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value
    assert fused_error.value.message == sequential_error.value.message
    assert fused_error.value.help == sequential_error.value.help
    assert fused_error.value.related == sequential_error.value.related
    assert fused_error.value.data == sequential_error.value.data


def test_custom_permute_expand_fusion_rejects_wrong_permute_shape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    route_ops = replace(
        backend_ops,
        permute=lambda tensor, _axes: tensor[:1, :2],
    )
    permute_step, expand_step = _build_single_permutation_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        fusion.tuple_runner((np.arange(6).reshape(2, 3),))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "permute"


def test_custom_permute_expand_fusion_preserves_permute_error_owner() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    def fail_permute(
        _tensor: TensorLike,
        _axes: tuple[int, ...],
    ) -> TensorLike:
        raise IndexError("custom permute failed")

    route_ops = replace(backend_ops, permute=fail_permute)
    permute_step, expand_step = _build_single_permutation_expand_steps(route_ops)
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None
    tensor = np.arange(6).reshape(2, 3)

    with pytest.raises(ExecutionError) as sequential_error:
        permute_step.run_unary(tensor)
    with pytest.raises(ExecutionError) as fused_error:
        fusion.tuple_runner((tensor,))

    assert fused_error.value.message == sequential_error.value.message
    assert fused_error.value.help == sequential_error.value.help
    assert fused_error.value.related == sequential_error.value.related
    assert fused_error.value.data == sequential_error.value.data


def test_portable_permute_expand_fusion_rejects_wrong_intermediate_shape() -> None:
    class WrongPermuteNamespace:
        __name__ = "custom.wrong_permute"

        @staticmethod
        def permute_dims(
            tensor: np.ndarray,
            _axes: tuple[int, ...],
        ) -> np.ndarray:
            return tensor[:1, :2]

        @staticmethod
        def expand_dims(tensor: np.ndarray, *, axis: int) -> np.ndarray:
            return np.expand_dims(tensor, axis=axis)

        @staticmethod
        def broadcast_to(
            tensor: np.ndarray,
            shape: tuple[int, ...],
        ) -> np.ndarray:
            return np.broadcast_to(tensor, shape)

    permute_step, expand_step = _build_portable_permute_expand_steps(
        WrongPermuteNamespace(),  # type: ignore[arg-type]
    )
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        fusion.tuple_runner((np.arange(6).reshape(2, 3),))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "permute"


def test_portable_permute_expand_fusion_rejects_wrong_inserted_axis_shape() -> None:
    class WrongExpandDimsNamespace:
        __name__ = "custom.wrong_expand_dims"

        @staticmethod
        def permute_dims(
            tensor: np.ndarray,
            axes: tuple[int, ...],
        ) -> np.ndarray:
            return np.transpose(tensor, axes)

        @staticmethod
        def expand_dims(tensor: np.ndarray, *, axis: int) -> np.ndarray:
            _ = axis
            return tensor[:1, :, None]

        @staticmethod
        def broadcast_to(
            tensor: np.ndarray,
            shape: tuple[int, ...],
        ) -> np.ndarray:
            return np.broadcast_to(tensor, shape)

    permute_step, expand_step = _build_portable_permute_expand_steps(
        WrongExpandDimsNamespace(),  # type: ignore[arg-type]
    )
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        fusion.tuple_runner((np.arange(6).reshape(2, 3),))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "expand"


def test_portable_permute_expand_fusion_preserves_permute_error_owner() -> None:
    class FailingPermuteNamespace:
        __name__ = "custom.failing_permute"

        @staticmethod
        def permute_dims(
            _tensor: np.ndarray,
            _axes: tuple[int, ...],
        ) -> np.ndarray:
            raise IndexError("permute boom")

        @staticmethod
        def expand_dims(tensor: np.ndarray, *, axis: int) -> np.ndarray:
            return np.expand_dims(tensor, axis=axis)

        @staticmethod
        def broadcast_to(
            tensor: np.ndarray,
            shape: tuple[int, ...],
        ) -> np.ndarray:
            return np.broadcast_to(tensor, shape)

    permute_step, expand_step = _build_portable_permute_expand_steps(
        FailingPermuteNamespace(),  # type: ignore[arg-type]
    )
    fusion = discover_step_fusion((permute_step, expand_step))
    assert fusion is not None
    tensor = np.arange(6).reshape(2, 3)

    with pytest.raises(ExecutionError) as sequential_error:
        permute_step.run_unary(tensor)
    with pytest.raises(ExecutionError) as fused_error:
        fusion.tuple_runner((tensor,))

    assert fused_error.value.message == sequential_error.value.message
    assert fused_error.value.help == sequential_error.value.help
    assert fused_error.value.related == sequential_error.value.related
    assert fused_error.value.data == sequential_error.value.data


def test_reshape_fusion_falls_back_after_wrong_route_output_shape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    class FirstReshapeIsWrong:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            tensor: TensorLike,
            shape: tuple[int, ...],
        ) -> TensorLike:
            self.calls += 1
            if self.calls == 1:
                return tensor
            return backend_ops.reshape(tensor, shape)

    first_reshape_is_wrong = FirstReshapeIsWrong()
    route_ops = replace(backend_ops, reshape=first_reshape_is_wrong)
    first_step, second_step = _build_reshape_fusion_steps(route_ops)
    fusion = discover_step_fusion((first_step, second_step))
    assert fusion is not None

    output = fusion.tuple_runner((np.arange(24).reshape(2, 3, 4),))

    assert output[0].shape == (2, 12)
    assert first_reshape_is_wrong.calls == 3


def test_reshape_fusion_preserves_structured_failure() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    original = ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="fused reshape invariant failed",
    )

    def fail_reshape(
        _tensor: TensorLike,
        _shape: tuple[int, ...],
    ) -> TensorLike:
        raise original

    route_ops = replace(backend_ops, reshape=fail_reshape)  # type: ignore[arg-type]
    first_step, second_step = _build_reshape_fusion_steps(route_ops)
    fusion = discover_step_fusion((first_step, second_step))
    assert fusion is not None

    with pytest.raises(ValidationError) as error:
        fusion.tuple_runner((np.arange(24).reshape(2, 3, 4),))

    assert error.value is original


def test_discover_does_not_fuse_reshape_chain_when_zero_copy_required() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
        zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
    )
    first_compiled = first_program.compiled
    assert first_compiled is not None
    first_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=first_program.zero_copy_mode,
    )

    second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    second_compiled = second_program.compiled
    assert second_compiled is not None
    second_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=second_program.zero_copy_mode,
    )

    fusion = discover_step_fusion((first_step, second_step))
    assert fusion is None


def test_discover_finds_multiple_non_overlapping_fusions() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    permute_first = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 2, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    permute_second = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0, 2)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )

    reshape_first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
    )
    reshape_first_compiled = reshape_first_program.compiled
    assert reshape_first_compiled is not None
    reshape_first = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=reshape_first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=reshape_first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=reshape_first_program.zero_copy_mode,
    )

    reshape_second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    reshape_second_compiled = reshape_second_program.compiled
    assert reshape_second_compiled is not None
    reshape_second = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=reshape_second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=reshape_second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=reshape_second_program.zero_copy_mode,
    )

    fusions = discover_step_fusions(
        (
            permute_first,
            permute_second,
            reshape_first,
            reshape_second,
        )
    )
    assert len(fusions) == 2
    assert (fusions[0].name, fusions[0].start, fusions[0].stop) == (
        "permute_permute",
        0,
        2,
    )
    assert (fusions[1].name, fusions[1].start, fusions[1].stop) == (
        "reshape_reshape",
        2,
        4,
    )
