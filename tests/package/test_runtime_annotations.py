from collections.abc import Callable
from typing import get_args, get_origin, get_type_hints

import pytest

from einf import ReducibleTensorOp, TensorLike, TensorOp, ax, axes, rearrange, reduce
from einf.operations.tensor_op import TensorOp as RuntimeTensorOp


def _annotated_unary(op: TensorOp[[TensorLike], TensorLike]) -> TensorLike:
    raise NotImplementedError


def _annotated_split(
    op: TensorOp[[TensorLike], tuple[TensorLike, TensorLike]],
) -> tuple[TensorLike, TensorLike]:
    raise NotImplementedError


def _annotated_reducer(op: ReducibleTensorOp[[TensorLike], TensorLike]) -> TensorLike:
    raise NotImplementedError


@pytest.mark.parametrize(
    ("function", "result"),
    [
        (_annotated_unary, TensorLike),
        (_annotated_split, tuple[TensorLike, TensorLike]),
        (_annotated_reducer, TensorLike),
    ],
)
def test_runtime_annotations_resolve_to_the_existing_operation_class(
    function: Callable[..., object], result: object
) -> None:
    hints = get_type_hints(function)
    annotation = hints["op"]
    assert get_origin(annotation) is RuntimeTensorOp
    assert get_args(annotation) == ((TensorLike,), result)


def test_reducer_annotation_is_not_a_runtime_capability_predicate() -> None:
    b, n = axes("b", "n")
    plain = rearrange(ax[b, n], ax[n, b])
    reducible = reduce(ax[b, n], ax[b])

    assert type(plain) is type(reducible) is RuntimeTensorOp
    assert TensorOp is ReducibleTensorOp is RuntimeTensorOp
    assert not plain.supports_reducer
    assert reducible.supports_reducer


def test_annotated_operations_keep_execution_and_configuration_behavior() -> None:
    np = pytest.importorskip("numpy")
    b, n = axes("b", "n")
    tensor = np.arange(6).reshape(2, 3)
    transpose = rearrange(ax[b, n], ax[n, b])
    mean = reduce(ax[b, n], ax[b]).reduce_by("mean").with_sizes(b=2)

    np.testing.assert_array_equal(transpose(tensor), tensor.T)
    np.testing.assert_array_equal(mean(tensor), tensor.mean(axis=1))
    assert rearrange(ax[b, n], ax[n, b]) is transpose

    def call_with_keyword(op: Callable[..., object]) -> object:
        return op(tensor=tensor)

    with pytest.raises(TypeError):
        call_with_keyword(transpose)
