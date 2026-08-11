from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from einf.axis import Axis, ScalarAxisTerms
from einf.backend import ArrayNamespace
from einf.diagnostics import ErrorCode, ExecutionError, ValidationError
from einf.reduction.callable import CallableReducerBinding
from einf.reduction.schema import ReducerName
from einf.steps.reduce.build import ReduceAxesResolver
from einf.steps.reduce.runtime import REDUCER_COMPILER, ReducerRuntimeContext


def _runtime_context() -> ReducerRuntimeContext:
    return ReducerRuntimeContext(xp=cast(ArrayNamespace, np))


def test_reduce_axis_resolution_diagnostic_reports_reduce_operation() -> None:
    current_axis = Axis("current_axis")
    missing_axis = Axis("missing_axis")

    with pytest.raises(ValidationError) as error:
        ReduceAxesResolver.resolve(
            current_terms=ScalarAxisTerms((current_axis,)),
            reduce_terms=ScalarAxisTerms((missing_axis,)),
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data == {"operation": "reduce"}


def test_reduce_output_type_diagnostic_reports_reduce_operation() -> None:
    with pytest.raises(ExecutionError) as error:
        _runtime_context().raise_output_type_error()

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value
    assert error.value.data == {"operation": "reduce"}


def test_string_reducer_failure_reports_operation_and_reducer() -> None:
    error = _runtime_context().string_reducer_error(
        reducer_name=ReducerName.MAX,
        tensor=np.zeros((2, 0)),
        axes=(1,),
        error=ValueError("empty domain"),
    )

    assert error.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.data == {"operation": "reduce", "reducer": "max"}


def test_custom_reducer_failure_reports_reduce_operation() -> None:
    error = _runtime_context().custom_reducer_error(error=ValueError("invalid domain"))

    assert error.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.data == {"operation": "reduce"}


def test_runtime_reducer_signature_diagnostic_reports_reduce_operation() -> None:
    with pytest.raises(ValidationError) as error:
        _runtime_context().raise_unsupported_reducer_signature()

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data == {"operation": "reduce"}


def test_unavailable_string_reducer_reports_operation_and_reducer() -> None:
    missing_namespace = cast(ArrayNamespace, SimpleNamespace())

    with pytest.raises(ValidationError) as error:
        REDUCER_COMPILER.compile(
            reducer=ReducerName.SUM,
            axes=(0,),
            xp=missing_namespace,
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data == {"operation": "reduce", "reducer": "sum"}


def test_compiled_reducer_signature_diagnostic_reports_reduce_operation() -> None:
    def unsupported_reducer(
        tensor: NDArray[np.float64],
        axis: tuple[int, ...],
        extra: bool,
    ) -> NDArray[np.float64]:
        _ = axis
        _ = extra
        return tensor

    with pytest.raises(ValidationError) as error:
        REDUCER_COMPILER.compile(
            reducer=CallableReducerBinding(unsupported_reducer),
            axes=(0,),
            xp=cast(ArrayNamespace, np),
        )

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert error.value.data == {"operation": "reduce"}
