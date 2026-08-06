import numpy as np
import pytest
from numpy.typing import NDArray

from einf import ErrorCode, ExecutionError, ValidationError, ax, axes, reduce, repeat
from einf.reduction.schema import CanonicalReducer
from einf.steps.expand import step as expand_step_module
from einf.steps.reduce import build as reduce_build_module
from einf.steps.reduce import step as reduce_step_module
from einf.tensor_types import TensorLike

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _explode_native_contract_einsum(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native contract einsum should not be called in this path")


def test_inflate_rejects_multi_input_lhs_with_diagnostic_code() -> None:
    (b,) = axes("b")

    with pytest.raises(ValidationError) as error:
        _ = repeat.__call__((ax[b], ax[b]), ax[b])

    assert error.value.code == ErrorCode.MULTI_INPUT_NOT_ALLOWED.value


def test_inflate_appends_axis_and_broadcasts_values() -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_repeat_appends_axis_and_broadcasts_values_with_torch() -> None:
    assert torch is not None
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = torch.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, tensor.unsqueeze(2).expand(2, 3, 4))


def test_repeat_uses_symbolic_fastpath_without_reindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, c, r = axes("b", "c", "r")
    op = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _explode_native_contract_einsum,
    )

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


def test_inflate_rank_zero_input_to_vector() -> None:
    op = repeat(ax[()], ax[3])

    tensor = np.asarray(7)
    result = op(tensor)

    expected = np.asarray([7, 7, 7])
    np.testing.assert_array_equal(result, expected)


def test_inflate_evaluable_expression_repeat_treats_expr_as_atomic() -> None:
    a, b, c = axes("a", "b", "c")
    op = repeat(ax[(a + a)], ax[b, (a + (a * c)), (c * c)]).with_sizes(a=2, b=1, c=2)

    result = op(np.zeros((4,)))

    assert result.shape == (1, 6, 4)


def test_inflate_singleton_axes_resolve_with_labeled_axis_preference() -> None:
    (n,) = axes("n")
    op = repeat(ax[n], ax[1, n, 1]).with_sizes(n=1)

    result = op(np.arange(1))

    expected = np.array([[[0]]])
    np.testing.assert_array_equal(result, expected)


def test_inflate_labeled_preference_falls_back_when_only_nonpreserving_path_is_viable() -> (
    None
):
    (a,) = axes("a")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a=2)

    result = op(np.array([5, 7, 9]))

    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_repeat_uses_solver_fast_path_for_expression_lhs_with_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (a,) = axes("a_solver_ws")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a_solver_ws=2)
    called = {"value": False}
    original_solve_expand_program_from_input_shape = (
        expand_step_module.solve_expand_program_from_input_shape
    )

    def _spy_solve_expand_program_from_input_shape(*args, **kwargs):
        called["value"] = True
        return original_solve_expand_program_from_input_shape(*args, **kwargs)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _spy_solve_expand_program_from_input_shape,
    )

    result = op(np.array([5, 7, 9]))
    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert called["value"]


def test_repeat_uses_solver_fast_path_without_explicit_sizes_when_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (a,) = axes("a_solver_wo")
    op = repeat(ax[(1 + a)], ax[(1 + a), a])
    called = {"value": False}
    original_solve_expand_program_from_input_shape = (
        expand_step_module.solve_expand_program_from_input_shape
    )

    def _spy_solve_expand_program_from_input_shape(*args, **kwargs):
        called["value"] = True
        return original_solve_expand_program_from_input_shape(*args, **kwargs)

    monkeypatch.setattr(
        expand_step_module,
        "solve_expand_program_from_input_shape",
        _spy_solve_expand_program_from_input_shape,
    )

    result = op(np.array([5, 7, 9]))
    expected = np.array(
        [
            [5, 5],
            [7, 7],
            [9, 9],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert called["value"]


def test_repeat_expression_solver_reports_ambiguity_when_non_unique() -> None:
    a, b = axes("a", "b")
    op = repeat(ax[(a + b)], ax[(a + b), a])

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_repeat_missing_explicit_axis_reports_validation_error() -> None:
    a, b = axes("a", "b")
    op = repeat(ax[a], ax[a, b])

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_repeat_expression_solver_reports_inconsistent_when_constraints_conflict() -> (
    None
):
    (a,) = axes("a")
    op = repeat(ax[(1 + a)], ax[(1 + a), a]).with_sizes(a=5)

    with pytest.raises(ValidationError) as error:
        _ = op(np.array([5, 7, 9]))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_default_sum_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_default_sum_uses_compiled_phase_without_reindex() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_default_sum_uses_static_runner_without_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])

    def explode_context_normalization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reduce runtime context normalization should be called")

    monkeypatch.setattr(
        reduce_step_module,
        "build_runtime_execution_context",
        explode_context_normalization,
    )

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)
    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_reuses_cached_compiled_runtime_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b])
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(op(tensor), expected)
    monkeypatch.setattr(
        reduce_build_module,
        "_compile_reduce_runtime_phase",
        _explode_native_contract_einsum,
    )
    np.testing.assert_array_equal(op(tensor), expected)


def test_reduce_compile_invariant_rejects_mismatched_output_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[d, b]).reduce_by(
        lambda x, *, axis: np.sum(x, axis=axis)
    )
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    reduce_build_module._REDUCE_RUNTIME_CACHE_ENTRIES.clear()
    reduce_build_module._REDUCE_RUNTIME_CACHE_ORDER.clear()
    compile_reduce_runtime_phase = reduce_build_module._compile_reduce_runtime_phase

    def compile_with_wrong_output_terms(
        *,
        lhs_terms: reduce_build_module.ScalarAxisTerms,
        reduce_axes: reduce_build_module.AxisTerms,
        reducer: CanonicalReducer,
        pack_sizes: dict[str, tuple[int, ...]],
        axis_sizes: dict[str, int],
        tensor: TensorLike,
        xp: reduce_build_module.ArrayNamespace,
    ) -> tuple[
        tuple[int, ...],
        reduce_build_module.CompiledReducer,
        reduce_build_module.ScalarAxisTerms,
    ]:
        axes, compiled_reducer, output_terms = compile_reduce_runtime_phase(
            lhs_terms=lhs_terms,
            reduce_axes=reduce_axes,
            reducer=reducer,
            pack_sizes=pack_sizes,
            axis_sizes=axis_sizes,
            tensor=tensor,
            xp=xp,
        )
        wrong_terms = reduce_build_module.ScalarAxisTerms(tuple(reversed(output_terms)))
        return axes, compiled_reducer, wrong_terms

    monkeypatch.setattr(
        reduce_build_module,
        "_compile_reduce_runtime_phase",
        compile_with_wrong_output_terms,
    )

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "reduce lowering invariant violated" in error.value.message


def test_reduce_duplicate_axis_partial_multiplicity_is_ambiguous() -> None:
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[b])

    tensor = np.arange(12).reshape(2, 2, 3)
    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.AMBIGUOUS_DIMS.value


def test_reduce_duplicate_axis_full_reduction_executes() -> None:
    b, d = axes("b", "d")
    op = reduce(ax[b, b, d], ax[()])

    tensor = np.arange(12).reshape(2, 2, 3)
    result = op(tensor)

    expected = np.sum(tensor, axis=(0, 1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_reorders_remaining_axes_to_rhs_order() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[d, b])

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.transpose(np.sum(tensor, axis=1), (1, 0))
    np.testing.assert_array_equal(result, expected)


def test_reduce_ordered_phase_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "prod"))

    tensor = np.arange(1, (2 * 3 * 4) + 1).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.prod(np.sum(tensor, axis=1), axis=1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_accepts_one_axis() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b, d]).reduce_by("prod")
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    torch.testing.assert_close(result, tensor.prod(dim=1))


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_accepts_multiple_axes() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by("prod")
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    expected = tensor.prod(dim=2).prod(dim=1)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_reduce_prod_torch_preserves_nonconsecutive_axis_positions() -> None:
    assert torch is not None
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[h]).reduce_by((ax[b, d], "prod"))
    tensor = torch.linspace(0.5, 1.5, steps=2 * 3 * 4, dtype=torch.float64).reshape(
        2, 3, 4
    )

    result = op(tensor)

    expected = tensor.prod(dim=2).prod(dim=0)
    torch.testing.assert_close(result, expected)


def test_reduce_ordered_phase_preserves_declared_axis_order() -> None:
    b, h, d = axes("b", "h", "d")
    seen_axes: list[tuple[int, ...]] = []

    def reducer(tensor: NDArray[np.int64], *, axis: tuple[int, ...]):
        seen_axes.append(axis)
        return np.sum(tensor, axis=axis)

    op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[d, h], reducer))

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    assert seen_axes == [(2, 1)]
    expected = np.sum(tensor, axis=(2, 1))
    np.testing.assert_array_equal(result, expected)


def test_reduce_callable_reducer_executes() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(tensor: NDArray[np.int64], axis: tuple[int, ...]):
        return np.max(tensor, axis=axis)

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.max(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_builtin_numpy_sum_callable_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by(np.sum)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_uninspectable_callable_executes() -> None:
    b, h, d = axes("b", "h", "d")
    op = reduce(ax[b, h, d], ax[b]).reduce_by(np.add.reduce)

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)

    expected = np.sum(tensor, axis=(1, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_uninspectable_callable_typeerror_is_not_swallowed() -> None:
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            /,
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = tensor
            _ = axis
            raise TypeError("user boom")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="user boom"):
        _ = op(tensor)


def test_reduce_uninspectable_callable_typeerror_with_binding_like_message_is_not_swallowed() -> (
    None
):
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            /,
            *,
            axis: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = tensor
            _ = axis
            raise TypeError("missing 1 required positional argument: 'x'")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _ = op(tensor)


def test_reduce_uninspectable_function_typeerror_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> NDArray[np.int64]:
        _ = tensor
        _ = axis
        raise TypeError("missing 1 required positional argument: 'x'")

    monkeypatch.setattr(reducer, "__signature__", object(), raising=False)
    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _ = op(tensor)


def test_reduce_uninspectable_callable_unsupported_signature_is_normalized() -> None:
    b, h, d = axes("b", "h", "d")

    class Reducer:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature")

        def __call__(
            self,
            tensor: NDArray[np.int64],
            axis: tuple[int, ...],
            axes: tuple[int, ...],
        ) -> NDArray[np.int64]:
            _ = axis
            _ = axes
            return tensor

    op = reduce(ax[b, h, d], ax[b]).reduce_by(Reducer())
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "reducer signature is unsupported" in str(error.value)


def test_reduce_callable_scalar_output_is_coerced_to_rank_zero_tensor() -> None:
    b, h = axes("b", "h")

    def reducer(tensor: np.ndarray) -> float:
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[()]).reduce_by(reducer)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    assert result.shape == ()
    assert np.asarray(result).item() == float(np.sum(tensor))


def test_reduce_scalar_custom_output_does_not_silently_broadcast() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> float:
        _ = axis
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)
    tensor = np.arange(6).reshape(2, 3)

    with pytest.raises(ExecutionError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_scalar_custom_output_is_validated_without_reindex() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.int64],
        *,
        axis: tuple[int, ...],
    ) -> float:
        _ = axis
        return float(np.sum(tensor))

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)

    with pytest.raises(ExecutionError) as error:
        _ = op(np.arange(6).reshape(2, 3))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_reduce_callable_typeerror_is_not_swallowed_by_fallback_dispatch() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64], *, axis: tuple[int, ...]
    ) -> NDArray[np.int64]:
        _ = tensor
        _ = axis
        raise TypeError("user boom")

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(TypeError, match="user boom"):
        _ = op(tensor)


def test_reduce_callable_valueerror_is_normalized_to_validation_error() -> None:
    b, h = axes("b", "h")

    def reducer(
        tensor: NDArray[np.float64], *, axis: tuple[int, ...]
    ) -> NDArray[np.float64]:
        _ = tensor
        _ = axis
        raise ValueError("domain exploded")

    op = reduce(ax[b, h], ax[b]).reduce_by(reducer)
    tensor = np.zeros((2, 0), dtype=np.float64)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "custom reducer failed" in str(error.value)


def test_reduce_string_reducer_empty_domain_is_normalized_to_validation_error() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[()]).reduce_by("max")

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 0)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "backend reducer 'max' failed" in str(error.value)


def test_reduce_builtin_numpy_max_callable_empty_domain_is_normalized() -> None:
    b, h = axes("b", "h")
    op = reduce(ax[b, h], ax[()]).reduce_by(np.max)

    with pytest.raises(ValidationError) as error:
        _ = op(np.zeros((2, 0)))

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "custom reducer failed" in str(error.value)


def test_reduce_callable_with_unsupported_signature_raises_validation_error() -> None:
    b, h, d = axes("b", "h", "d")

    def reducer(
        tensor: NDArray[np.int64],
        axis: tuple[int, ...],
        axes: tuple[int, ...],
    ) -> NDArray[np.int64]:
        _ = axis
        _ = axes
        return tensor

    op = reduce(ax[b, h, d], ax[b]).reduce_by(reducer)
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
    assert "reducer signature is unsupported" in str(error.value)
