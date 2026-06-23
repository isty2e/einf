from dataclasses import dataclass

import numpy as np
import pytest
from array_api_compat import numpy as array_api_numpy

from einf import (
    ErrorCode,
    ValidationError,
    ax,
    axes,
    contract,
    rearrange,
    reduce,
    repeat,
    view,
)
from einf.reduction.schema import ReducerPhase


@dataclass(frozen=True, slots=True)
class InvalidShapeTensor:
    shape: tuple[int | bool, ...]


@dataclass(frozen=True, slots=True)
class InputShapeListTensor:
    shape: list[int]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy


@dataclass(frozen=True, slots=True)
class InputShapeBoolTensor:
    shape: tuple[int | bool, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy


@dataclass(frozen=True, slots=True)
class InputWithoutShapeTensor:
    marker: str = "no-shape"

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy


@dataclass(frozen=True, slots=True)
class InputShapeRaisesTensor:
    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return array_api_numpy

    @property
    def shape(self) -> tuple[int, ...]:
        raise RuntimeError("input shape exploded")


@dataclass(frozen=True, slots=True)
class CustomFamilyTensor:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> type:
        _ = api_version
        return _CustomNamespace

    def __getitem__(self, key: object) -> "CustomFamilyTensor":
        _ = key
        return self


class _CustomNamespace:
    __name__ = "custom.family"


@dataclass(frozen=True, slots=True)
class NumpyLinalgFamilyTensor:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpy.linalg"

        return Namespace()

    def __getitem__(self, key: object) -> "NumpyLinalgFamilyTensor":
        _ = key
        return self


@dataclass(frozen=True, slots=True)
class NumpyRandomFamilyTensor:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpy.random"

        return Namespace()

    def __getitem__(self, key: object) -> "NumpyRandomFamilyTensor":
        _ = key
        return self


def test_tensorop_underflow_arity_raises_validation_error() -> None:
    b, c = axes("b", "c")
    x = np.zeros((2,), dtype=np.float32)
    op = contract((ax[b], ax[c]), ax[b])

    with pytest.raises(ValidationError) as error:
        _ = getattr(op, "__call__")(x)

    assert error.value.code == ErrorCode.OP_ARITY_MISMATCH.value
    assert error.value.external_code == "OP_ARITY_MISMATCH"


def test_tensorop_signature_property_reuses_cached_instance() -> None:
    b, n, d = axes("b", "n", "d")
    op = rearrange(ax[b, n, d], ax[b, n, d])

    first = op.signature
    second = op.signature

    assert first is second
    assert first.inputs == op.lhs
    assert first.outputs == op.rhs


def test_tensorop_base_constructor_reuses_cached_instance() -> None:
    b, n, d = axes("b", "n", "d")
    first = reduce(ax[b, n, d], ax[b, d])
    second = reduce(ax[b, n, d], ax[b, d])

    assert first is second


def test_tensorop_with_sizes_reuses_cached_configured_instance() -> None:
    b, c, r = axes("b", "c", "r")
    base = repeat(ax[b, c], ax[b, c, r])
    first = base.with_sizes(r=4)
    second = base.with_sizes(r=4)

    assert first is second


def test_tensorop_with_sizes_cache_key_is_canonical_over_kwarg_order() -> None:
    a, b, c = axes("a", "b", "c")
    base = rearrange(ax[a, b, c], ax[(a * b), c])
    first = base.with_sizes(a=2, b=3)
    second = base.with_sizes(b=3, a=2)

    assert first is second


def test_tensorop_reduce_by_reuses_cached_configured_instance() -> None:
    b, h, d = axes("b", "h", "d")
    base = reduce(ax[b, h, d], ax[b])
    first = base.reduce_by((ax[h], "sum"), (ax[d], "prod"))
    second = base.reduce_by((ax[h], "sum"), (ax[d], "prod"))

    assert first is second


def test_tensorop_sizes_property_returns_detached_copy() -> None:
    b, n, d = axes("b", "n", "d")
    op = reduce(ax[b, n, d], ax[b, d]).with_sizes(n=3)
    detached = op.sizes
    detached["n"] = 100

    assert op.sizes["n"] == 3


def test_tensorop_reduce_uses_sum_by_default_without_reduce_by() -> None:
    b, c = axes("b", "c")
    op = reduce(ax[b, c], ax[b])

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    result = op(x)

    np.testing.assert_allclose(result, np.sum(x, axis=1))


def test_tensorop_reduce_supports_ordered_phase_reducers() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "prod"))

    assert op.reducer_plan == (
        ReducerPhase(axes=(h,), reducer="sum"),
        ReducerPhase(axes=(d,), reducer="prod"),
    )


def test_tensorop_reduce_rejects_dict_reducer_plan() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    with pytest.raises(TypeError, match="dict reducer plans are not supported"):
        _ = op.reduce_by({ax[h]: "sum", ax[d]: "prod"})  # type: ignore[arg-type]


def test_tensorop_reduce_rejects_phase_plan_that_does_not_partition() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op.reduce_by((ax[h], "sum"))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_reduce_by_rejects_string_reducer_with_extra_phase_args() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[b])

    with pytest.raises(
        TypeError,
        match="phase reducer arguments must start with a phase tuple",
    ):
        _ = op.reduce_by("sum", (ax[h], "sum"))  # type: ignore[arg-type]


def test_tensorop_reduce_by_rejects_non_string_non_callable_reducer() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[b])

    with pytest.raises(TypeError, match="reducer must be a string or callable"):
        _ = op.reduce_by(123)  # type: ignore[arg-type]


def test_tensorop_reduce_rejects_duplicate_phase_coverage() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op.reduce_by((ax[h], "sum"), (ax[h], "prod"), (ax[d], "sum"))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_reduce_default_reducer_rejects_ambiguous_partial_multiplicity() -> (
    None
):
    (b,) = axes("b")
    x = np.arange(9, dtype=np.float32).reshape(3, 3)

    op = reduce(ax[b, b], ax[b])
    with pytest.raises(ValidationError) as error:
        _ = op(x)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_tensorop_reduce_explicit_single_reducer_allows_duplicate_keep_multiplicity() -> (
    None
):
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[b, b])

    configured = op.reduce_by("sum")

    assert configured.reducer_plan == (ReducerPhase(axes=(d,), reducer="sum"),)


def test_tensorop_overflow_arity_raises_validation_error() -> None:
    (b,) = axes("b")
    x = np.zeros((3,), dtype=np.float32)
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = getattr(op, "__call__")(x, x)

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value
    assert error.value.external_code == "MULTI_INPUT_NOT_ALLOWED"


def test_tensorop_inflate_overflow_arity_raises_multi_input_not_allowed() -> None:
    b, r = axes("b", "r")
    x = np.zeros((3,), dtype=np.float32)
    op = repeat(ax[b], ax[b, r]).with_sizes(r=2)

    with pytest.raises(ValidationError) as error:
        _ = getattr(op, "__call__")(x, x)

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value
    assert error.value.external_code == "MULTI_INPUT_NOT_ALLOWED"


def test_rearrange_constructor_normalizes_empty_axis_list_spelling() -> None:
    op_single = getattr(rearrange, "__call__")((), ())
    op_tuple = getattr(rearrange, "__call__")(((),), ((),))

    assert op_single.lhs == ((),)
    assert op_single.rhs == ((),)
    assert op_tuple.lhs == ((),)
    assert op_tuple.rhs == ((),)


def test_tensorop_call_rejects_mixed_tensor_families_before_execution() -> None:
    (b,) = axes("b")
    x_numpy = np.zeros((3,), dtype=np.float32)
    x_custom = CustomFamilyTensor(shape=(3,))
    op = contract((ax[b], ax[b]), ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x_numpy, x_custom)  # type: ignore[arg-type,arg-type]

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_MIXED_FAMILY.value


def test_tensorop_call_known_family_aliases_fail_when_runtime_namespace_differs() -> (
    None
):
    (b,) = axes("b")
    x_linalg = NumpyLinalgFamilyTensor(shape=(3,))
    x_random = NumpyRandomFamilyTensor(shape=(3,))
    op = rearrange((ax[b], ax[b]), (ax[b], ax[b]))

    with pytest.raises(ValidationError) as error:
        _ = op(x_linalg, x_random)

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_tensorop_contract_rejects_missing_einsum_extension() -> None:
    (b,) = axes("b")
    x_custom = CustomFamilyTensor(shape=(3,))

    op = contract((ax[b], ax[b]), ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x_custom, x_custom)

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value


def test_tensorop_view_rejects_unsupported_input() -> None:
    (b,) = axes("b")
    x = InvalidShapeTensor(shape=(3,))  # type: ignore[arg-type]
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_tensorop_call_rejects_input_without_shape_attribute_with_validation_error() -> (
    None
):
    (b,) = axes("b")
    x = InputWithoutShapeTensor()
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_input_shape_property_failure_with_validation_error() -> (
    None
):
    (b,) = axes("b")
    x = InputShapeRaisesTensor()
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_non_tuple_input_shape_with_validation_error() -> None:
    (b,) = axes("b")
    x = InputShapeListTensor(shape=[3])
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_call_rejects_bool_input_shape_entries_with_validation_error() -> None:
    (b,) = axes("b")
    x = InputShapeBoolTensor(shape=(True,))  # type: ignore[arg-type]
    op = view(ax[b], ax[b])

    with pytest.raises(ValidationError) as error:
        _ = op(x)  # type: ignore[arg-type]

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_tensorop_reduce_by_on_non_reducer_op_raises_attribute_error() -> None:
    (b,) = axes("b")
    op = view(ax[b], ax[b])

    with pytest.raises(AttributeError, match="does not support .reduce_by"):
        _ = getattr(op, "reduce_by")("sum")


def test_tensorop_does_not_expose_with_executor() -> None:
    (b,) = axes("b")
    op = view(ax[b], ax[b])

    assert not hasattr(op, "with_executor")
