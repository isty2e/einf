from dataclasses import dataclass
from typing import cast

import pytest

import einf.output_normalization as output_normalization_module
from einf import ErrorCode, ExecutionError
from einf.output_normalization import (
    RuntimeOutputContract,
    normalize_outputs,
    normalize_runtime_outputs,
)
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class DummyTensor:
    name: str
    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> "DummyTensor":
        _ = key
        return self


@dataclass(frozen=True, slots=True)
class BadShapeTensor:
    shape: list[int]

    def __getitem__(self, key: object) -> "BadShapeTensor":
        _ = key
        return self


@dataclass(frozen=True, slots=True)
class VariableShapeTensor:
    shape: tuple[int, ...] | list[int]

    def __getitem__(self, key: object) -> "VariableShapeTensor":
        _ = key
        return self


@dataclass(frozen=True, slots=True)
class TrustedShapeTensor:
    __module__ = "numpy"

    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> "TrustedShapeTensor":
        _ = key
        return self


def test_call_contract_unary_singleton_list_output_is_unwrapped() -> None:
    out = DummyTensor(name="out", shape=(3,))

    result = normalize_outputs(
        op_name="view",
        expected_output_arity=1,
        raw_outputs=[out],
    )
    assert result is out


def test_call_contract_multi_output_list_is_tuple_normalized() -> None:
    out1 = DummyTensor(name="out1", shape=(3,))
    out2 = DummyTensor(name="out2", shape=(3,))

    result = normalize_outputs(
        op_name="rearrange",
        expected_output_arity=2,
        raw_outputs=[out1, out2],
    )
    assert result == (out1, out2)


def test_call_contract_multi_output_arity_mismatch_is_execution_error() -> None:
    out = DummyTensor(name="out", shape=(3,))

    with pytest.raises(ExecutionError) as error:
        _ = normalize_outputs(
            op_name="rearrange",
            expected_output_arity=2,
            raw_outputs=[out],
        )

    captured = error.value
    assert captured.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value
    assert captured.external_code == "OP_OUTPUT_PROTOCOL_VIOLATION"
    assert captured.data == {"operation": "rearrange", "expected": 2, "got": 1}


def test_call_contract_bad_shape_type_is_execution_error() -> None:
    with pytest.raises(ExecutionError) as error:
        _ = normalize_outputs(
            op_name="view",
            expected_output_arity=1,
            raw_outputs=BadShapeTensor(shape=[3]),  # type: ignore[arg-type]
        )

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_runtime_output_contract_caches_validated_trusted_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = RuntimeOutputContract(op_name="rearrange", expected_output_arity=1)
    original_validate = output_normalization_module._validate_output_tensors
    validation_calls = 0

    def counting_validate(*, op_name: str, outputs: tuple[TensorLike, ...]) -> None:
        nonlocal validation_calls
        validation_calls += 1
        original_validate(op_name=op_name, outputs=outputs)

    monkeypatch.setattr(
        output_normalization_module,
        "_validate_output_tensors",
        counting_validate,
    )

    _ = contract.normalize(TrustedShapeTensor(shape=(2, 3)))
    _ = contract.normalize(TrustedShapeTensor(shape=(4, 5)))

    assert validation_calls == 1


def test_runtime_output_contract_revalidates_untrusted_type() -> None:
    contract = RuntimeOutputContract(op_name="rearrange", expected_output_arity=1)

    first = contract.normalize(
        cast(TensorLike, VariableShapeTensor(shape=(3,))),
    )

    assert isinstance(first, VariableShapeTensor)
    with pytest.raises(ExecutionError) as error:
        _ = contract.normalize(
            cast(TensorLike, VariableShapeTensor(shape=[3])),
        )

    assert error.value.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value


def test_call_contract_runtime_unary_tuple_is_unwrapped() -> None:
    out = DummyTensor(name="out", shape=(2, 3))

    result = normalize_runtime_outputs(
        op_name="view",
        expected_output_arity=1,
        raw_outputs=(out,),
    )
    assert result is out


def test_call_contract_runtime_multi_output_tuple_is_preserved() -> None:
    out1 = DummyTensor(name="out1", shape=(3,))
    out2 = DummyTensor(name="out2", shape=(5,))

    result = normalize_runtime_outputs(
        op_name="rearrange",
        expected_output_arity=2,
        raw_outputs=(out1, out2),
    )
    assert result == (out1, out2)


def test_call_contract_runtime_multi_output_arity_mismatch_is_execution_error() -> None:
    out = DummyTensor(name="out", shape=(3,))

    with pytest.raises(ExecutionError) as error:
        _ = normalize_runtime_outputs(
            op_name="rearrange",
            expected_output_arity=2,
            raw_outputs=(out,),
        )

    captured = error.value
    assert captured.code == ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION.value
    assert captured.external_code == "OP_OUTPUT_PROTOCOL_VIOLATION"
    assert captured.data == {"operation": "rearrange", "expected": 2, "got": 1}
