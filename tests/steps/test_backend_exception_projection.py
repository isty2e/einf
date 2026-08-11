from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import einf.steps.einsum.native as native_module
import einf.steps.einsum.step as einsum_step_module
import einf.steps.reshape.runtime as reshape_runtime_module
import einf.steps.runtime as steps_runtime_module
from einf.axis import AxisTerms
from einf.backend import BackendProfile, get_backend_array_ops
from einf.backend.memory_alias import numpy_shares_storage, torch_shares_storage
from einf.backend.runtime import BackendArrayOps, is_trusted_backend_array_ops
from einf.diagnostics import ErrorCode, ExecutionError, ValidationError
from einf.reduction.schema import ReducerName
from einf.steps.axis_slice.step import AxisSliceRuntimeStep
from einf.steps.base import RuntimeSpecializationContext
from einf.steps.concat import ConcatRuntimeStep
from einf.steps.einsum.native import try_native_contract_einsum
from einf.steps.einsum.step import EinsumEquationExecutor
from einf.steps.expand import build_expand_symbolic_program
from einf.steps.expand.runtime import run_expand_program
from einf.steps.expand.step import ExpandRuntimeStep
from einf.steps.permute import PermuteRuntimeStep
from einf.steps.reduce.runtime import CallableReducerInvoker, ReducerRuntimeContext
from einf.steps.reduce.step import (
    DirectMethodReduceRuntimeProgram,
    NamespaceReduceRuntimeProgram,
)
from einf.steps.reshape.constants import ZERO_COPY_REQUIRED_RESHAPE_MODE
from einf.steps.reshape.runtime import run_reshape_program
from einf.steps.runtime import bind_runtime_backend
from einf.tensor_types import TensorLike, is_trusted_tensor_type


class _PermuteProgram:
    def __init__(
        self,
        *,
        has_non_identity_permutation: bool,
        permutation: tuple[int, ...],
    ) -> None:
        self.has_non_identity_permutation = has_non_identity_permutation
        self.permutation = permutation


def _raise_backend(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("raw backend primitive failure")


def _raising_backend_ops(permute) -> BackendArrayOps:
    return BackendArrayOps(
        backend_family="numpy",
        reshape=lambda tensor, shape: tensor,
        permute=permute,
        expand_dims=lambda tensor, axis: tensor,
        broadcast_to=lambda tensor, shape: tensor,
        concat=lambda tensors, axis: tensors[0],
        reducers={},
    )


def test_permute_fast_path_normalizes_backend_exception() -> None:
    step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=_PermuteProgram(  # type: ignore[arg-type]
            has_non_identity_permutation=True,
            permutation=(1, 0),
        ),
        runtime_backend_ops=_raising_backend_ops(_raise_backend),
        runtime_xp=np,  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionError) as error:
        step.run_unary(np.zeros((2, 3)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_permute_rejects_wrong_backend_output_shape() -> None:
    step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=_PermuteProgram(  # type: ignore[arg-type]
            has_non_identity_permutation=True,
            permutation=(1, 0),
        ),
        runtime_backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
        runtime_xp=None,
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        step.run_unary(np.zeros((2, 3)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_expand_compiled_fast_path_normalizes_unexpected_exception() -> None:
    def raise_index_error(_tensor: object) -> None:
        raise IndexError("raw index error")

    step = ExpandRuntimeStep(
        name="expand",
        input_arity=1,
        output_arity=1,
        program=object(),  # type: ignore[arg-type]
        compiled_unary_runner=raise_index_error,
    )

    with pytest.raises(ExecutionError) as error:
        step.run_unary(np.zeros((2, 3)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_expand_rejects_wrong_backend_output_shape() -> None:
    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((2, 3, 1)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        run_expand_program(
            plan=program,
            tensor=np.zeros((2, 3)),
            target_shape=(2, 3, 1),
            backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
            xp=None,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_expand_rejects_wrong_custom_adapter_permute_shape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    route_ops = replace(
        backend_ops,
        permute=lambda tensor, _axes: tensor[:1, :2],
    )
    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        run_expand_program(
            plan=program,
            tensor=np.arange(6).reshape(2, 3),
            target_shape=(3, 2, 1),
            backend_ops=route_ops,
            xp=None,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "permute"


def test_backend_array_ops_trust_rejects_native_tensor_subclass() -> None:
    class ArraySubclass(np.ndarray):
        pass

    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    tensor = np.arange(6).reshape(2, 3).view(ArraySubclass)

    assert not is_trusted_backend_array_ops(
        backend_ops=backend_ops,
        tensor=cast(TensorLike, tensor),
    )


def test_direct_expand_projects_permute_failure_to_permute_owner() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    route_ops = replace(
        backend_ops,
        permute=lambda _tensor, _axes: (_ for _ in ()).throw(
            IndexError("permute failed")
        ),
    )
    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )

    with pytest.raises(ExecutionError, match="reindex execution") as error:
        run_expand_program(
            plan=program,
            tensor=np.arange(6).reshape(2, 3),
            target_shape=(3, 2, 1),
            backend_ops=route_ops,
            xp=None,
        )

    assert error.value.data["operation"] == "rearrange"


def test_direct_expand_projects_expand_dims_failure() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None
    route_ops = replace(
        backend_ops,
        expand_dims=lambda _tensor, _axis: (_ for _ in ()).throw(
            IndexError("expand dims failed")
        ),
    )
    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((2, 3, 1)),
    )

    with pytest.raises(ExecutionError, match="expand runtime failed") as error:
        run_expand_program(
            plan=program,
            tensor=np.arange(6).reshape(2, 3),
            target_shape=(2, 3, 1),
            backend_ops=route_ops,
            xp=None,
        )

    assert error.value.data["operation"] == "expand"


def test_portable_expand_rejects_wrong_permute_intermediate_shape() -> None:
    class WrongPermuteNamespace:
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

    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        run_expand_program(
            plan=program,
            tensor=np.arange(6).reshape(2, 3),
            target_shape=(3, 2, 1),
            backend_ops=None,
            xp=WrongPermuteNamespace(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "permute"


def test_portable_expand_rejects_wrong_inserted_axis_shape() -> None:
    class WrongExpandDimsNamespace:
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

    program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        run_expand_program(
            plan=program,
            tensor=np.arange(6).reshape(2, 3),
            target_shape=(3, 2, 1),
            backend_ops=None,
            xp=WrongExpandDimsNamespace(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data["operation"] == "expand"


def test_concat_rejects_wrong_backend_output_shape() -> None:
    class ConcatProgram:
        concat_axis = 0

    step = ConcatRuntimeStep(
        name="concat",
        input_arity=2,
        output_arity=1,
        program=ConcatProgram(),  # type: ignore[arg-type]
        runtime_backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
        runtime_xp=None,
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        step.run((np.zeros((2, 3)), np.zeros((4, 3))))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_axis_slice_rejects_wrong_backend_output_shape() -> None:
    class SlicingTensor:
        shape = (2, 3)

        def __getitem__(self, key: object) -> "SlicingTensor":
            _ = key
            return self

    class AxisSliceProgram:
        split_axis = 0
        strict_view = False
        signature = None

    step = AxisSliceRuntimeStep(
        name="axis_slice",
        input_arity=1,
        output_arity=2,
        program=AxisSliceProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        precomputed_split_sizes=(1, 1),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        step.run((SlicingTensor(),))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reducer_rejects_shape_only_output() -> None:
    class ShapeOnly:
        shape = (2, 3)

    ctx = ReducerRuntimeContext(xp=np)  # type: ignore[arg-type]

    with pytest.raises(ExecutionError, match="must be tensor-like") as error:
        ctx.coerce_output(ShapeOnly())  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_direct_method_reducer_rejects_wrong_output_shape() -> None:
    class WrongShapeTensor:
        shape = (2, 3)

        def sum(self, *, axis: object) -> "WrongShapeTensor":
            _ = axis
            return self

        def __getitem__(self, key: object) -> "WrongShapeTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version
            return np

    program = DirectMethodReduceRuntimeProgram(
        reducer=ReducerName("sum"),
        axes=(1,),
        runtime_context=ReducerRuntimeContext(xp=np),  # type: ignore[arg-type]
        direct_method_name="sum",
        direct_axis_keyword="axis",
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        program.run_unary(WrongShapeTensor())

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_namespace_reducer_rejects_wrong_output_shape() -> None:
    def wrong_shape_reducer(tensor: object, *, axis: object) -> object:
        _ = tensor, axis
        return tensor

    class WrongShapeTensor:
        shape = (2, 3)

        def __getitem__(self, key: object) -> "WrongShapeTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version
            return np

    program = NamespaceReduceRuntimeProgram(
        reducer=ReducerName("sum"),
        axes=(1,),
        runtime_context=ReducerRuntimeContext(xp=np),  # type: ignore[arg-type]
        reducer_fn=wrong_shape_reducer,  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        program.run_unary(WrongShapeTensor())

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_direct_method_reducer_routes_output_through_coercion() -> None:
    class NoGetitemOutput:
        shape = (2,)

    class DirectTensor:
        shape = (2, 3)

        def sum(self, *, axis: object) -> NoGetitemOutput:
            _ = axis
            return NoGetitemOutput()

        def __getitem__(self, key: object) -> "DirectTensor":
            _ = key
            return self

    program = DirectMethodReduceRuntimeProgram(
        reducer=ReducerName("sum"),
        axes=(1,),
        runtime_context=ReducerRuntimeContext(xp=np),  # type: ignore[arg-type]
        direct_method_name="sum",
        direct_axis_keyword="axis",
    )

    with pytest.raises(ExecutionError, match="must be tensor-like") as error:
        program.run_unary(DirectTensor())

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_reshape_step_projects_final_fallback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from einf.steps.reshape import step as reshape_step_module
    from einf.steps.reshape.step import ReshapeRuntimeStep

    class FakeCompiled:
        pass

    class FakeProgram:
        compiled = FakeCompiled()
        reject_not_a_view = False
        signature = None

    def raise_backend(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("raw final reshape failure")

    monkeypatch.setattr(
        reshape_step_module,
        "try_run_reshape_program",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        reshape_step_module,
        "run_reshape_program",
        raise_backend,
    )
    monkeypatch.setattr(
        reshape_step_module,
        "build_runtime_execution_context",
        lambda **_kwargs: SimpleNamespace(axis_sizes={}, rhs_terms=((),)),
    )
    step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=FakeProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        runtime_backend_ops=None,
        runtime_xp=np,  # type: ignore[arg-type]
        zero_copy_mode="allow_copy",
    )

    with pytest.raises(ExecutionError, match="reshape backend failed") as error:
        step.run_unary(np.zeros((1,)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_axis_slice_projects_strict_view_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from einf.backend.dispatch import BACKEND_RESOLVER
    from einf.steps.axis_slice.step import AxisSliceRuntimeStep

    class SlicingTensor:
        def __init__(self, shape: tuple[int, ...] = (2, 3)) -> None:
            self.shape = shape

        def __getitem__(self, key: object) -> "SlicingTensor":
            _ = key
            return SlicingTensor((1, 3))

    def raise_numel(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("raw view validation failure")

    from einf.steps.axis_slice import validation as axis_slice_validation_module

    monkeypatch.setattr(
        axis_slice_validation_module,
        "_shares_memory",
        raise_numel,
    )

    class FakeProgram:
        split_axis = 0
        strict_view = True
        signature = None

    step = AxisSliceRuntimeStep(
        name="axis_slice",
        input_arity=1,
        output_arity=1,
        program=FakeProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        backend_profile=BACKEND_RESOLVER.resolve(
            np.zeros((2, 3)),
            op_name="view",
        ),
        precomputed_split_sizes=(1, 1),
    )

    with pytest.raises(
        ExecutionError,
        match="axis_slice backend slicing failed",
    ) as error:
        step.run((SlicingTensor(),))  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_compiled_runner_validation_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from einf.steps.reshape import step as reshape_step_module
    from einf.steps.reshape.step import ReshapeRuntimeStep

    class FakeProgram:
        reject_not_a_view = False
        compiled = object()
        signature = None

    def raise_validation(_tensor: object) -> None:
        raise ValidationError(
            code="inconsistent_dims", message="compiled invariant broken"
        )

    monkeypatch.setattr(
        reshape_step_module,
        "try_run_reshape_program",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        reshape_step_module,
        "build_runtime_execution_context",
        lambda **_kwargs: SimpleNamespace(axis_sizes={}, rhs_terms=((),)),
    )
    monkeypatch.setattr(
        reshape_step_module,
        "run_reshape_program",
        lambda **_kwargs: np.zeros((1,)),
    )
    step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=FakeProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        compiled_unary_runner=raise_validation,
    )

    with pytest.raises(ValidationError, match="compiled invariant broken"):
        step.run_unary(np.zeros((1,)))


def test_reshape_compiled_runner_projects_non_tuple_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from einf.steps.reshape import step as reshape_step_module
    from einf.steps.reshape.step import ReshapeRuntimeStep

    class FakeProgram:
        reject_not_a_view = False
        compiled = object()
        signature = None

    def raise_index_error(_tensor: object) -> None:
        raise IndexError("raw index error")

    monkeypatch.setattr(
        reshape_step_module,
        "try_run_reshape_program",
        lambda **_kwargs: None,
    )
    step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=FakeProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        compiled_unary_runner=raise_index_error,
    )

    with pytest.raises(ExecutionError, match="reshape backend failed") as error:
        step.run_unary(np.zeros((1,)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_try_run_reshape_program_normalizes_backend_attribute_error() -> None:
    from einf.steps.reshape.runtime import try_run_reshape_program

    class FakeProgram:
        def __init__(self) -> None:
            self.lhs_axis_names = ["b", "n"]
            self.lhs_literal_checks: list[object] = []
            self.lhs_axis_equal_checks: list[object] = []
            self.axis_names = {"b", "n"}
            self.rhs_fast_shape_eval_fns = None
            self.rhs_shape_eval_fns = [
                lambda shape, sizes: shape[1],
                lambda shape, sizes: shape[0],
            ]

    class MissingReshapeNamespace:
        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise AttributeError("raw attr from backend")

    result = try_run_reshape_program(
        tensor=np.zeros((2, 3)),
        explicit_sizes={},
        program=FakeProgram(),  # type: ignore[arg-type]
        backend_ops=None,
        xp=MissingReshapeNamespace(),  # type: ignore[arg-type]
    )

    assert result is None


def test_reducer_output_coercion_projects_backend_failure() -> None:
    from einf.steps.reduce.runtime import ReducerRuntimeContext

    class BadXP:
        def asarray(self, value: object) -> object:
            _ = value
            raise RuntimeError("asarray backend died")

    ctx = ReducerRuntimeContext(xp=BadXP())  # type: ignore[arg-type]

    with pytest.raises(ExecutionError, match="coercion failed") as error:
        ctx.coerce_output(42)

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_compiled_runner_execution_error_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from einf.diagnostics import ExecutionError
    from einf.steps.reshape import step as reshape_step_module
    from einf.steps.reshape.step import ReshapeRuntimeStep

    class FakeProgram:
        reject_not_a_view = False
        compiled = object()
        signature = None

    def raise_execution(_tensor: object) -> None:
        raise ExecutionError(
            code="op_output_protocol_violation",
            message="execution broken",
        )

    monkeypatch.setattr(
        reshape_step_module,
        "try_run_reshape_program",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        reshape_step_module,
        "build_runtime_execution_context",
        lambda **_kwargs: SimpleNamespace(axis_sizes={}, rhs_terms=((),)),
    )
    monkeypatch.setattr(
        reshape_step_module,
        "run_reshape_program",
        lambda **_kwargs: np.zeros((1,)),
    )
    step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=FakeProgram(),  # type: ignore[arg-type]
        explicit_sizes={},
        compiled_unary_runner=raise_execution,
    )

    with pytest.raises(ExecutionError, match="execution broken"):
        step.run_unary(np.zeros((1,)))


def test_reducer_output_ownership_rejects_foreign_namespace() -> None:
    from einf.steps.reduce.runtime import ReducerRuntimeContext

    class ForeignTensor:
        shape = (2,)

        def __getitem__(self, key: object) -> "ForeignTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version

            class ForeignNamespace:
                __name__ = "foreign.backend"

            return ForeignNamespace()

    ctx = ReducerRuntimeContext(xp=np)  # type: ignore[arg-type]

    with pytest.raises(
        ExecutionError,
        match="different backend namespace",
    ) as error:
        ctx.coerce_output(ForeignTensor())

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_reducer_preserves_structured_errors_from_all_boundaries() -> None:
    reducer_name = ReducerName("sum")
    tensor = np.zeros((2, 3))
    execution_error = ExecutionError(
        code="op_output_protocol_violation",
        message="backend reducer execution failed",
    )

    class BadBackendOps:
        def reduce(self, **_kwargs: object) -> object:
            raise execution_error

    backend_context = ReducerRuntimeContext(
        xp=np,  # type: ignore[arg-type]
        backend_ops=BadBackendOps(),  # type: ignore[arg-type]
    )
    with pytest.raises(ExecutionError) as backend_failure:
        backend_context.apply_string_reducer(
            reducer_name=reducer_name,
            reducer_fn=lambda _tensor, **_kwargs: tensor,
            tensor=tensor,
            axes=(1,),
        )
    assert backend_failure.value is execution_error

    namespace_error = ValidationError(
        code="inconsistent_dims",
        message="namespace reducer invariant failed",
    )
    namespace_context = ReducerRuntimeContext(xp=np)  # type: ignore[arg-type]
    with pytest.raises(ValidationError) as namespace_failure:
        namespace_context.apply_string_reducer(
            reducer_name=reducer_name,
            reducer_fn=lambda _tensor, **_kwargs: (_ for _ in ()).throw(
                namespace_error
            ),
            tensor=tensor,
            axes=(1,),
        )
    assert namespace_failure.value is namespace_error

    callable_error = ValidationError(
        code="inconsistent_dims",
        message="callable reducer invariant failed",
    )
    invoker = CallableReducerInvoker(
        reducer=lambda _tensor, **_kwargs: (_ for _ in ()).throw(callable_error),
        call_mode="axis_keyword",
    )
    with pytest.raises(ValidationError) as callable_failure:
        invoker.invoke(tensor=tensor, axes=(1,), context=namespace_context)
    assert callable_failure.value is callable_error


def test_final_einsum_route_preserves_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ValidationError(
        code="inconsistent_dims",
        message="einsum invariant failed",
    )
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    profile = BackendProfile(
        namespace=np,
    )
    executor = EinsumEquationExecutor(profile=profile)

    with pytest.raises(ValidationError) as error:
        executor.run(
            equation="i->i",
            operands=(np.zeros((2,)),),
            chain_mode=False,
            allow_native_matmul=False,
        )

    assert error.value is original


def test_final_einsum_route_projects_unexpected_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IndexError("final einsum route failed")
        ),
    )
    executor = EinsumEquationExecutor(
        profile=BackendProfile(
            namespace=np,
        )
    )

    with pytest.raises(ExecutionError, match="final einsum route failed") as error:
        executor.run(
            equation="i->i",
            operands=(np.zeros((2,)),),
            chain_mode=False,
            allow_native_matmul=False,
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_portable_einsum_rejects_wrong_semantic_output_shape() -> None:
    class PortableNamespace:
        __name__ = "array_api_compat.jax.numpy"

        @staticmethod
        def einsum(*_args: object) -> np.ndarray:
            return np.zeros((99,))

    executor = einsum_step_module._build_einsum_executor(
        BackendProfile(
            namespace=PortableNamespace(),
        )
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="ij,jk->ik",
            operands=(np.zeros((2, 3)), np.zeros((3, 4))),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_numpy_family_einsum_validates_namespace_fallback_output() -> None:
    class NumpyFamilyTensor:
        shape: tuple[int, ...]

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __getitem__(self, key: object) -> "NumpyFamilyTensor":
            _ = key
            return self

    class NumpyFamilyNamespace:
        __name__ = "numpy.custom"

        @staticmethod
        def einsum(*_args: object) -> np.ndarray:
            return np.zeros((99,))

    executor = einsum_step_module._build_einsum_executor(
        BackendProfile(
            namespace=NumpyFamilyNamespace(),
        )
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="ij,jk->ik",
            operands=(NumpyFamilyTensor((2, 3)), NumpyFamilyTensor((3, 4))),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_exact_native_einsum_skips_semantic_shape_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_shape_derivation(**_kwargs: object) -> None:
        raise AssertionError("native einsum must not derive semantic output shape")

    monkeypatch.setattr(
        einsum_step_module,
        "einsum_output_shape",
        fail_shape_derivation,
    )
    executor = einsum_step_module._build_einsum_executor(
        BackendProfile(
            namespace=np,
        )
    )

    output = executor.run(
        equation="ij,jk->ik",
        operands=(np.zeros((2, 3)), np.zeros((3, 4))),
        chain_mode=False,
        allow_native_matmul=True,
    )

    assert output.shape == (2, 4)


def test_injected_native_einsum_callable_validates_output_shape() -> None:
    executor = EinsumEquationExecutor(
        profile=BackendProfile(
            namespace=np,
        ),
        native_module_matmul=lambda _lhs, _rhs: np.zeros((99,)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="ij,jk->ik",
            operands=(np.zeros((2, 3)), np.zeros((3, 4))),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_injected_native_module_einsum_validates_output_shape() -> None:
    executor = EinsumEquationExecutor(
        profile=BackendProfile(
            namespace=np,
        ),
        native_module_einsum=lambda *_args: np.zeros((99,)),
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="i->i",
            operands=(np.zeros((2,)),),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_torch_override_mode_disables_native_validation_bypass() -> None:
    torch = pytest.importorskip("torch")

    wrong_output = torch.zeros((99,))

    class WrongMatmulMode(torch.overrides.TorchFunctionMode):
        def __torch_function__(
            self,
            function: Callable[..., object],
            types: tuple[type[object], ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            _ = self, types
            if function is torch.matmul:
                return wrong_output
            return function(*args, **(kwargs or {}))

    backend_ops = get_backend_array_ops("torch")
    assert backend_ops is not None
    executor = einsum_step_module._build_einsum_executor(
        BackendProfile(
            namespace=torch,
        )
    )
    operands = (torch.zeros((2, 3)), torch.zeros((3, 4)))

    with (
        WrongMatmulMode(),
        pytest.raises(
            ExecutionError,
            match="output shape does not match",
        ) as error,
    ):
        assert not is_trusted_backend_array_ops(
            backend_ops=backend_ops,
            tensor=operands[0],
        )
        executor.run(
            equation="ij,jk->ik",
            operands=operands,
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_torch_dispatch_mode_disables_native_validation_bypass() -> None:
    torch = pytest.importorskip("torch")
    backend_ops = get_backend_array_ops("torch")
    assert backend_ops is not None
    tensor = torch.zeros((2, 3))

    with torch.utils._python_dispatch.TorchDispatchMode():
        assert not is_trusted_backend_array_ops(
            backend_ops=backend_ops,
            tensor=tensor,
        )


def test_namespace_einsum_fallback_runs_once_and_validates_opt_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace_calls: list[None] = []

    def unavailable_namespace(*_args: object) -> TensorLike:
        namespace_calls.append(None)
        raise RuntimeError("namespace route unavailable")

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        lambda *_args, **_kwargs: np.zeros((99,)),
    )
    executor = EinsumEquationExecutor(
        profile=BackendProfile(
            namespace=np,
        ),
        namespace_einsum=unavailable_namespace,
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="ij,jk->ik",
            operands=(np.zeros((2, 3)), np.zeros((3, 4))),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert len(namespace_calls) == 1


def test_final_opt_einsum_route_validates_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        lambda *_args, **_kwargs: np.zeros((99,)),
    )
    executor = EinsumEquationExecutor(
        profile=BackendProfile(
            namespace=np,
        )
    )

    with pytest.raises(ExecutionError, match="output shape does not match") as error:
        executor.run(
            equation="ij,jk->ik",
            operands=(np.zeros((2, 3)), np.zeros((3, 4))),
            chain_mode=False,
            allow_native_matmul=True,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_native_einsum_projects_unexpected_fallback_error() -> None:
    class BadNamespace:
        def einsum(self, *_args: object) -> object:
            raise IndexError("native namespace index failure")

    with pytest.raises(ExecutionError, match="native einsum failed") as error:
        try_native_contract_einsum(
            equation="i->i",
            tensors=(np.zeros((2,)),),
            namespace=BadNamespace(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_native_einsum_projects_capability_lookup_failure() -> None:
    class BadNamespace:
        __name__ = "custom.bad_lookup"

        def __getattribute__(self, name: str) -> object:
            if name == "einsum":
                raise IndexError("einsum lookup failed")
            return object.__getattribute__(self, name)

    with pytest.raises(ExecutionError, match="einsum lookup failed") as error:
        try_native_contract_einsum(
            equation="i->i",
            tensors=(np.zeros((2,)),),
            namespace=BadNamespace(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_namespace_reshape_projects_unexpected_error() -> None:
    class BadNamespace:
        __name__ = "custom.reshape"

        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise IndexError("namespace reshape index failure")

    with pytest.raises(ExecutionError, match="reshape backend failed") as error:
        run_reshape_program(
            tensor=np.zeros((2, 3)),
            target_shape=(3, 2),
            backend_ops=None,
            xp=BadNamespace(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_reshape_rejects_wrong_backend_output_shape() -> None:
    with pytest.raises(ExecutionError, match="requested output shape") as error:
        run_reshape_program(
            tensor=np.zeros((2, 3)),
            target_shape=(3, 2),
            backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
            xp=None,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_native_einsum_projects_backend_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_family: str) -> object:
        raise RuntimeError("backend initialization failed")

    class NamespaceWithoutEinsum:
        def __init__(self) -> None:
            self.__name__ = "torch.array_api"

    monkeypatch.setattr(native_module, "load_backend_module", fail_load)

    with pytest.raises(
        ExecutionError,
        match="backend initialization failed",
    ) as error:
        try_native_contract_einsum(
            equation="i->i",
            tensors=(np.zeros((2,)),),
            namespace=NamespaceWithoutEinsum(),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_native_einsum_does_not_load_torch_for_foreign_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_family: str) -> object:
        raise AssertionError("torch route must not be selected")

    class ForeignNamespace:
        def __init__(self) -> None:
            self.__name__ = "custom.foreign"

    monkeypatch.setattr(native_module, "load_backend_module", fail_load)

    assert (
        try_native_contract_einsum(
            equation="i->i",
            tensors=(np.zeros((2,)),),
            namespace=ForeignNamespace(),  # type: ignore[arg-type]
        )
        is None
    )


def test_scalar_reducer_ownership_rejects_foreign_coerced_output() -> None:
    class ForeignTensor:
        shape = ()

        def __getitem__(self, key: object) -> "ForeignTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version

            class ForeignNamespace:
                __name__ = "foreign.scalar"

            return ForeignNamespace()

    class PrimaryNamespace:
        __name__ = "primary.scalar"

        def asarray(self, value: object) -> ForeignTensor:
            _ = value
            return ForeignTensor()

    with pytest.raises(
        ExecutionError,
        match="different backend namespace",
    ) as error:
        ReducerRuntimeContext(xp=PrimaryNamespace()).coerce_output(1)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_strict_numpy_reshape_projects_primitive_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadModule:
        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise IndexError("strict module failure")

    class BadNamespace:
        __name__ = "numpy"

        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("namespace fallback failure")

    class BadBackendOps:
        backend_family = "numpy"

        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("backend primitive failure")

    monkeypatch.setattr(
        reshape_runtime_module,
        "load_backend_module",
        lambda _family: BadModule(),
    )

    with pytest.raises(ExecutionError, match="strict module failure") as error:
        run_reshape_program(
            tensor=np.zeros((2, 3)),
            target_shape=(3, 2),
            backend_ops=BadBackendOps(),  # type: ignore[arg-type]
            xp=BadNamespace(),  # type: ignore[arg-type]
            zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_strict_numpy_reshape_preserves_no_route_as_not_a_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingReshapeRoute:
        __name__ = "numpy"

        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("reshape route unavailable")

    monkeypatch.setattr(
        reshape_runtime_module,
        "load_backend_module",
        lambda _family: MissingReshapeRoute(),
    )

    with pytest.raises(ValidationError, match="not a view") as error:
        run_reshape_program(
            tensor=np.zeros((2, 3)),
            target_shape=(3, 2),
            backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
            xp=MissingReshapeRoute(),  # type: ignore[arg-type]
            zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
        )

    assert error.value.code == ErrorCode.NOT_A_VIEW.value


def test_strict_numpy_reshape_does_not_use_order_changing_namespace_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingStrictRoute:
        def reshape(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("strict route unavailable")

    class NamespaceFallback:
        __name__ = "numpy"

        def __init__(self) -> None:
            self.calls = 0

        def reshape(self, tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            self.calls += 1
            return np.reshape(tensor, shape, order="C")

    namespace = NamespaceFallback()
    monkeypatch.setattr(
        reshape_runtime_module,
        "load_backend_module",
        lambda _family: MissingStrictRoute(),
    )

    tensor = np.asfortranarray(np.arange(8).reshape(2, 4))
    with pytest.raises(ValidationError, match="not a view"):
        run_reshape_program(
            tensor=tensor,
            target_shape=(2, 2, 2),
            backend_ops=_raising_backend_ops(lambda value, _axes: value),
            xp=namespace,  # type: ignore[arg-type]
            zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
        )

    assert namespace.calls == 0


def test_backend_specialization_projects_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = BackendProfile(
        namespace=np,
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=profile,
    )
    monkeypatch.setattr(
        steps_runtime_module,
        "get_backend_array_ops",
        lambda _family: (_ for _ in ()).throw(RuntimeError("initialization failed")),
    )

    with pytest.raises(ExecutionError, match="initialization failed") as error:
        bind_runtime_backend(
            context,
            operation="reshape",
            required_namespace_methods=("reshape",),
            bind_namespace_when_backend_ops_available=True,
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_backend_specialization_projects_capability_inspection_failure() -> None:
    class BrokenNamespace:
        __name__ = "custom.broken"

        def __getattribute__(self, name: str) -> object:
            if name == "reshape":
                raise RuntimeError("capability inspection failed")
            return super().__getattribute__(name)

    profile = BackendProfile(
        namespace=BrokenNamespace(),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=profile,
    )

    with pytest.raises(ExecutionError, match="capability inspection failed") as error:
        bind_runtime_backend(
            context,
            operation="reshape",
            required_namespace_methods=("reshape",),
            bind_namespace_when_backend_ops_available=True,
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_einsum_specialization_projects_capability_inspection_failure() -> None:
    class BrokenNamespace:
        __name__ = "custom.broken"

        def __getattribute__(self, name: str) -> object:
            if name == "einsum":
                raise RuntimeError("einsum capability inspection failed")
            return super().__getattribute__(name)

    profile = BackendProfile(
        namespace=BrokenNamespace(),
    )

    with pytest.raises(ExecutionError, match="capability inspection failed") as error:
        einsum_step_module._build_einsum_executor(profile)

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_numpy_alias_capability_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteNumpyModule:
        pass

    monkeypatch.setattr(
        "einf.backend.memory_alias.load_backend_module",
        lambda _family: IncompleteNumpyModule(),
    )

    with pytest.raises(ExecutionError, match="shares_memory") as error:
        numpy_shares_storage(lhs=np.zeros((2,)), rhs=np.zeros((2,)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_numpy_alias_initialization_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "einf.backend.memory_alias.load_backend_module",
        lambda _family: (_ for _ in ()).throw(RuntimeError("numpy init failed")),
    )

    with pytest.raises(ExecutionError, match="numpy init failed") as error:
        numpy_shares_storage(lhs=np.zeros((2,)), rhs=np.zeros((2,)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_numpy_alias_runtime_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenNumpyModule:
        ndarray = np.ndarray

        @staticmethod
        def shares_memory(_lhs: object, _rhs: object) -> bool:
            raise OSError("numpy alias runtime failed")

    monkeypatch.setattr(
        "einf.backend.memory_alias.load_backend_module",
        lambda _family: BrokenNumpyModule(),
    )

    with pytest.raises(ExecutionError, match="alias runtime failed") as error:
        numpy_shares_storage(lhs=np.zeros((2,)), rhs=np.zeros((2,)))

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_numpy_alias_proof_rejects_subclass_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LyingArray(np.ndarray):
        def __array_function__(
            self,
            func: object,
            types: object,
            args: object,
            kwargs: object,
        ) -> object:
            _ = types, args, kwargs
            if func is np.shares_memory:
                return True
            return NotImplemented

    lhs = np.arange(6).view(LyingArray)
    rhs = lhs.copy()
    monkeypatch.setattr(
        "einf.backend.memory_alias.load_backend_module",
        lambda _family: (_ for _ in ()).throw(
            AssertionError("untrusted alias proof must not load NumPy")
        ),
    )

    assert not np.shares_memory(lhs.view(np.ndarray), rhs.view(np.ndarray))
    assert (
        numpy_shares_storage(
            lhs=cast(TensorLike, lhs),
            rhs=cast(TensorLike, rhs),
        )
        is None
    )


def test_numpy_alias_proof_rejects_hostile_metaclass_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeTypeSpoof(type):
        def __hash__(cls) -> int:
            return hash(np.ndarray)

        def __eq__(cls, other: object) -> bool:
            return other is np.ndarray

    class SpoofTensor(metaclass=NativeTypeSpoof):
        shape = (6,)

        def __getitem__(self, key: object) -> "SpoofTensor":
            _ = key
            return self

    assert is_trusted_tensor_type(np.ndarray)
    monkeypatch.setattr(
        "einf.backend.memory_alias.load_backend_module",
        lambda _family: (_ for _ in ()).throw(
            AssertionError("untrusted alias proof must not load NumPy")
        ),
    )

    assert (
        numpy_shares_storage(
            lhs=cast(TensorLike, SpoofTensor()),
            rhs=cast(TensorLike, SpoofTensor()),
        )
        is None
    )


def test_torch_alias_proof_rejects_untrusted_none_storage() -> None:
    class CustomTorchTensor:
        __module__ = "torch.custom"
        shape = (0,)

        def untyped_storage(self) -> None:
            return None

        def __getitem__(self, _key: object) -> "CustomTorchTensor":
            return self

    lhs = CustomTorchTensor()
    rhs = CustomTorchTensor()

    assert torch_shares_storage(lhs=lhs, rhs=rhs) is None


def test_torch_alias_proof_ignores_instance_storage_shadow() -> None:
    torch = pytest.importorskip("torch")
    lhs = torch.arange(6)
    rhs = lhs.clone()
    view = lhs.view(2, 3)

    class FakeStorage:
        def __init__(self, pointer: int) -> None:
            self.pointer = pointer

        def data_ptr(self) -> int:
            return self.pointer

    fake_storage = FakeStorage(1)
    lhs.untyped_storage = lambda: fake_storage
    rhs.untyped_storage = lambda: fake_storage
    view.untyped_storage = lambda: FakeStorage(2)

    assert torch_shares_storage(lhs=lhs, rhs=rhs) is False
    assert torch_shares_storage(lhs=lhs, rhs=view) is True


@pytest.mark.parametrize("raw_error", (OSError("io failed"), IndexError("bad index")))
def test_named_reducer_unexpected_fault_is_execution_error(
    raw_error: Exception,
) -> None:
    class BadBackendOps:
        def reduce(self, **_kwargs: object) -> object:
            raise raw_error

    context = ReducerRuntimeContext(
        xp=np,  # type: ignore[arg-type]
        backend_ops=BadBackendOps(),  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionError) as error:
        context.apply_string_reducer(
            reducer_name=ReducerName("max"),
            reducer_fn=np.max,
            tensor=np.zeros((2, 3)),
            axes=(1,),
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_strict_numpy_reshape_does_not_fallback_on_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_family: str) -> object:
        raise RuntimeError("numpy initialization failed")

    monkeypatch.setattr(reshape_runtime_module, "load_backend_module", fail_load)

    with pytest.raises(
        ExecutionError,
        match="numpy initialization failed",
    ) as error:
        run_reshape_program(
            tensor=np.zeros((2, 3)),
            target_shape=(3, 2),
            backend_ops=_raising_backend_ops(lambda tensor, _axes: tensor),
            xp=np,  # type: ignore[arg-type]
            zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
        )

    assert error.value.code == ErrorCode.BACKEND_EXECUTION_FAILED.value


def test_reducer_same_id_foreign_namespace_is_rejected() -> None:
    class Namespace:
        __name__ = "custom.same"

    class ForeignTensor:
        shape = ()

        def __getitem__(self, key: object) -> "ForeignTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version
            return Namespace()

    context_namespace = Namespace()
    context = ReducerRuntimeContext(xp=context_namespace)  # type: ignore[arg-type]

    with pytest.raises(
        ExecutionError,
        match="different backend namespace",
    ) as error:
        context.coerce_output(ForeignTensor())

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_reducer_shape_access_failure_is_structured() -> None:
    class MalformedTensor:
        @property
        def shape(self) -> tuple[int, ...]:
            raise LookupError("shape lookup failed")

    context = ReducerRuntimeContext(xp=np)  # type: ignore[arg-type]

    with pytest.raises(ExecutionError, match="must be tensor-like") as error:
        context.coerce_output(MalformedTensor())  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_reducer_namespace_error_identity_is_preserved() -> None:
    original = ValidationError(
        code="inconsistent_dims",
        message="namespace hook failed",
    )

    class BadTensor:
        shape = ()

        def __getitem__(self, key: object) -> "BadTensor":
            _ = key
            return self

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version
            raise original

    context = ReducerRuntimeContext(xp=np)  # type: ignore[arg-type]

    with pytest.raises(ValidationError) as error:
        context.coerce_output(BadTensor())
    assert error.value is original


def test_backend_dispatch_preserves_ingress_tensor_error() -> None:
    from einf.backend import BACKEND_RESOLVER

    original = ExecutionError(
        code="op_output_protocol_violation",
        message="namespace resolution failed",
    )

    class BadTensor:
        shape = ()

        def __array_namespace__(self, api_version: object = None) -> object:
            _ = api_version
            raise original

    with pytest.raises(ExecutionError) as error:
        BACKEND_RESOLVER.resolve(BadTensor(), op_name="rearrange")
    assert error.value is original
