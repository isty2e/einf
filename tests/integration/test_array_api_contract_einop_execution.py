from collections.abc import Callable

import numpy as np
import pytest

import einf.plans.abstract as abstract_plan_module
import einf.plans.runners as runner_module
import einf.steps.einsum as einsum_step_module
import einf.steps.einsum.step as einsum_step_impl
import einf.steps.permute as permute_step_module
from einf import (
    ErrorCode,
    TensorOp,
    ValidationError,
    ax,
    axes,
    contract,
    einop,
    packs,
    reduce,
    repeat,
)
from einf.lowering import einop as einop_plan_module
from einf.steps.expand import step as expand_step_module
from einf.tensor_types import TensorLike

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _explode_opt_einsum_contract(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "opt_einsum.contract should not be called on torch native contract path"
    )


def _explode_native_contract_einsum(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native contract einsum should not be called in this path")


def _explode_build_tuple_runner(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "build_tuple_runner should not be called for one-step single-output plans"
    )


def _single_tensor_output(
    result: TensorLike | tuple[TensorLike, ...],
    /,
) -> TensorLike:
    if isinstance(result, tuple):
        raise AssertionError("expected one tensor output")
    return result


def test_contract_matrix_multiply_executes_with_numpy() -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    result = op(left, right)
    assert not isinstance(result, tuple)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def test_contract_matrix_multiply_numpy_prefers_native_matmul_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    result = _single_tensor_output(op(left, right))
    assert isinstance(result, np.ndarray)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "build_op",
    (
        lambda i, k, j: contract((ax[i, k], ax[k, j]), ax[i, j]),
        lambda i, k, j: einop((ax[i, k], ax[k, j]), ax[i, j]),
    ),
)
def test_atomic_contract_equivalent_numpy_ops_prefer_native_matmul_path(
    monkeypatch: pytest.MonkeyPatch,
    build_op: Callable[..., TensorOp],
) -> None:
    i, k, j = axes("i", "k", "j")
    op = build_op(i, k, j)

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(3 * 4).reshape(3, 4)
    result = _single_tensor_output(op(left, right))
    assert isinstance(result, np.ndarray)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_contract_matrix_multiply_torch_uses_native_einsum_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    assert torch is not None
    left = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    right = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    result = _single_tensor_output(op(left, right))
    assert isinstance(result, torch.Tensor)

    expected = left @ right
    assert torch.equal(result, expected)


@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize(
    "build_op",
    (
        lambda i, k, j: contract((ax[i, k], ax[k, j]), ax[i, j]),
        lambda i, k, j: einop((ax[i, k], ax[k, j]), ax[i, j]),
    ),
)
def test_atomic_contract_equivalent_torch_ops_skip_opt_einsum_contract(
    monkeypatch: pytest.MonkeyPatch,
    build_op: Callable[..., TensorOp],
) -> None:
    i, k, j = axes("i", "k", "j")
    op = build_op(i, k, j)

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_opt_einsum_contract,
    )

    assert torch is not None
    left = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    right = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    result = _single_tensor_output(op(left, right))
    assert isinstance(result, torch.Tensor)

    expected = left @ right
    assert torch.equal(result, expected)


def test_contract_three_inputs_reuses_cached_contract_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, j, k, out = axes("i", "j", "k", "out")
    op = contract((ax[i, j], ax[j, k], ax[k, out]), ax[i, out])
    compiled_calls = {"value": 0}

    def _spy_contract_expression(
        equation: str,
        *operand_shapes: tuple[int, ...],
        optimize: str,
    ) -> Callable[..., np.ndarray]:
        compiled_calls["value"] += 1
        _ = operand_shapes
        _ = optimize

        def _run(*operands: np.ndarray) -> np.ndarray:
            return np.einsum(equation, *operands, optimize=True)

        return _run

    einsum_step_impl._cached_contract_expression.cache_clear()

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract_expression",
        _spy_contract_expression,
    )
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_native_contract_einsum,
    )

    left = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    middle = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    right = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
    first = op(left, middle, right)
    second = op(left, middle, right)

    expected = left @ middle @ right
    assert compiled_calls["value"] == 1
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)


def test_contract_expression_cache_is_shared_across_nary_contract_and_einop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, j, k, out = axes("i", "j", "k", "out")
    contract_op = contract((ax[i, j], ax[j, k], ax[k, out]), ax[i, out])
    einop_op = einop((ax[i, j], ax[j, k], ax[k, out]), ax[i, out])
    compiled_calls = {"value": 0}

    def _spy_contract_expression(
        equation: str,
        *operand_shapes: tuple[int, ...],
        optimize: str,
    ) -> Callable[..., np.ndarray]:
        compiled_calls["value"] += 1
        _ = operand_shapes
        _ = optimize

        def _run(*operands: np.ndarray) -> np.ndarray:
            return np.einsum(equation, *operands, optimize=True)

        return _run

    einsum_step_impl._cached_contract_expression.cache_clear()

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract_expression",
        _spy_contract_expression,
    )
    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_native_contract_einsum,
    )

    left = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    middle = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    right = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)

    contract_result = _single_tensor_output(contract_op(left, middle, right))
    einop_result = _single_tensor_output(einop_op(left, middle, right))
    assert isinstance(contract_result, np.ndarray)
    assert isinstance(einop_result, np.ndarray)

    expected = left @ middle @ right
    assert compiled_calls["value"] == 1
    np.testing.assert_allclose(contract_result, expected)
    np.testing.assert_allclose(einop_result, expected)
    assert einsum_step_impl._cached_contract_expression.cache_info().maxsize == 2048


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_einop_chain_two_input_torch_uses_native_contract_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    monkeypatch.setattr(
        einsum_step_module.opt_einsum,
        "contract",
        _explode_native_contract_einsum,
    )

    assert torch is not None
    lhs = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
    rhs = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    out_left, out_right = op(lhs, rhs)

    expected = torch.einsum("bmn,nd->bmd", lhs, rhs)
    assert torch.equal(out_left, expected[:, :6, :])
    assert torch.equal(out_right, expected[:, 6:, :])


def test_einop_concat_matches_multi_input_rearrange_concat() -> None:
    a, b, c = axes("a", "b", "c")
    op = einop((ax[a, b], ax[c, b]), ax[(a + c), b])

    left = np.arange(2 * 3).reshape(2, 3)
    right = np.arange(4 * 3).reshape(4, 3) + 100
    result = op(left, right)

    expected = np.concatenate((left, right), axis=0)
    np.testing.assert_array_equal(result, expected)


def test_einop_factorized_contract_executes_with_numpy() -> None:
    b, h, d, j = axes("b", "h", "d", "j")
    op = einop((ax[b, (h * d)], ax[(h * d), j]), ax[b, j]).with_sizes(h=2, d=3)

    left = np.arange(2 * 6).reshape(2, 6)
    right = np.arange(6 * 4).reshape(6, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def test_einop_atomic_contract_skips_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    op = einop((ax[b, n, d], ax[d, j]), ax[b, n, j])

    monkeypatch.setattr(
        permute_step_module,
        "build_runtime_execution_context",
        _explode_native_contract_einsum,
    )

    left = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    right = np.arange(4 * 5).reshape(4, 5)
    result = op(left, right)

    expected = np.einsum("bnd,dj->bnj", left, right, optimize=True)
    np.testing.assert_array_equal(result, expected)


def test_einop_factorized_contract_with_pack_executes_with_numpy() -> None:
    h, d, j = axes("h", "d", "j")
    (tail,) = packs("tail")
    op = einop((ax[tail, (h * d)], ax[(h * d), j]), ax[tail, j]).with_sizes(h=2, d=3)

    left = np.arange(2 * 5 * 6).reshape(2, 5, 6)
    right = np.arange(6 * 4).reshape(6, 4)
    result = op(left, right)

    expected = left @ right
    np.testing.assert_array_equal(result, expected)


def test_einop_repeat_like_path_does_not_use_tensorop_contract_fastpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, c, r = axes("b", "c", "r")
    op = einop(ax[b, c], ax[b, c, r]).with_sizes(r=4)
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

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    assert not called["value"]
    np.testing.assert_array_equal(result, expected)


def test_einop_contract_then_split_outputs_executes_with_numpy() -> None:
    b, h, w, d, j = axes("b", "h", "w", "d", "j")
    op = einop((ax[b, (h + w), d], ax[d, j]), (ax[b, h, j], ax[b, w, j])).with_sizes(
        h=2, w=1
    )

    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    right = np.arange(5 * 4).reshape(5, 4)
    out_left, out_right = op(left, right)

    intermediate = np.einsum("bnd,dj->bnj", left, right)
    expected_left = intermediate[:, :2, :]
    expected_right = intermediate[:, 2:, :]
    np.testing.assert_array_equal(out_left, expected_left)
    np.testing.assert_array_equal(out_right, expected_right)


def test_einop_three_input_contract_then_split_outputs_executes_with_numpy() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    left = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    middle = np.arange(5 * 4).reshape(5, 4)
    right = np.arange(4 * 6).reshape(4, 6)
    out_left, out_right = op(left, middle, right)

    intermediate = np.einsum("bnd,dj,jk->bnk", left, middle, right)
    expected_left = intermediate[:, :2, :]
    expected_right = intermediate[:, 2:, :]
    np.testing.assert_array_equal(out_left, expected_left)
    np.testing.assert_array_equal(out_right, expected_right)


def test_einop_cached_chain_split_skips_nested_stage_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    monkeypatch.setattr(
        einop_plan_module,
        "build_einop_execution_plan",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_einop_cached_chain_split_skips_tensor_op_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_einop_cached_chain_split_uses_direct_slice_tail_fastpath() -> None:
    b, h1, h2, r, n, d = axes("b", "h1", "h2", "r", "n", "d")
    op = einop(
        (ax[b, ((h1 + h2) * r), n], ax[n, d]),
        (ax[b, (h1 * r), d], ax[b, (h2 * r), d]),
    ).with_sizes(h1=2, h2=1, r=3)

    first_lhs = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    first_rhs = np.arange(4 * 5).reshape(4, 5)
    _ = op(first_lhs, first_rhs)

    second_lhs = np.arange(3 * 9 * 4).reshape(3, 9, 4)
    second_rhs = np.arange(4 * 5).reshape(4, 5)
    out_left, out_right = op(second_lhs, second_rhs)

    intermediate = np.einsum("bmn,nd->bmd", second_lhs, second_rhs)
    np.testing.assert_array_equal(out_left, intermediate[:, :6, :])
    np.testing.assert_array_equal(out_right, intermediate[:, 6:, :])


def test_shape_free_single_runner_cache_is_arity_agnostic_for_ternary_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d, j, k = axes("b", "n", "d", "j", "k")
    op = contract((ax[b, n, d], ax[d, j], ax[j, k]), ax[b, n, k])

    first_lhs = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    first_mid = np.arange(4 * 5).reshape(4, 5)
    first_rhs = np.arange(5 * 6).reshape(5, 6)
    np.testing.assert_array_equal(
        op(first_lhs, first_mid, first_rhs),
        np.einsum("bnd,dj,jk->bnk", first_lhs, first_mid, first_rhs),
    )

    monkeypatch.setattr(
        abstract_plan_module.AbstractPlan,
        "resolve_single_output_runner",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 3 * 4).reshape(3, 3, 4)
    second_mid = np.arange(4 * 5).reshape(4, 5)
    second_rhs = np.arange(5 * 6).reshape(5, 6)
    np.testing.assert_array_equal(
        op(second_lhs, second_mid, second_rhs),
        np.einsum("bnd,dj,jk->bnk", second_lhs, second_mid, second_rhs),
    )


def test_shape_free_tuple_runner_cache_is_arity_agnostic_for_ternary_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    first_lhs = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    first_mid = np.arange(5 * 4).reshape(5, 4)
    first_rhs = np.arange(4 * 6).reshape(4, 6)
    first_out_left, first_out_right = op(first_lhs, first_mid, first_rhs)
    first_intermediate = np.einsum("bnd,dj,jk->bnk", first_lhs, first_mid, first_rhs)
    np.testing.assert_array_equal(first_out_left, first_intermediate[:, :2, :])
    np.testing.assert_array_equal(first_out_right, first_intermediate[:, 2:, :])

    monkeypatch.setattr(
        abstract_plan_module.AbstractPlan,
        "resolve_tuple_runner",
        _explode_native_contract_einsum,
    )

    second_lhs = np.arange(3 * 3 * 5).reshape(3, 3, 5)
    second_mid = np.arange(5 * 4).reshape(5, 4)
    second_rhs = np.arange(4 * 6).reshape(4, 6)
    second_out_left, second_out_right = op(second_lhs, second_mid, second_rhs)
    second_intermediate = np.einsum(
        "bnd,dj,jk->bnk", second_lhs, second_mid, second_rhs
    )
    np.testing.assert_array_equal(second_out_left, second_intermediate[:, :2, :])
    np.testing.assert_array_equal(second_out_right, second_intermediate[:, 2:, :])


@pytest.mark.parametrize("case_name", ["reduce", "repeat", "contract"])
def test_single_output_runner_bypasses_tuple_chain_compile_for_one_step_plans(
    case_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, h, w, d, r, j = axes("b", "h", "w", "d", "r", "j")

    if case_name == "reduce":
        op = reduce(ax[b, h, w, d], ax[b, d])
        tensor = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        tensors = (tensor,)
        expected = np.sum(tensor, axis=(1, 2))
    elif case_name == "repeat":
        op = repeat(ax[b, d], ax[b, d, r]).with_sizes(r=4)
        tensor = np.arange(2 * 5).reshape(2, 5)
        tensors = (tensor,)
        expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 5, 4))
    else:
        op = contract((ax[b, h, d], ax[d, j]), ax[b, h, j])
        lhs = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        rhs = np.arange(4 * 6).reshape(4, 6)
        tensors = (lhs, rhs)
        expected = np.einsum("bhd,dj->bhj", lhs, rhs)

    monkeypatch.setattr(
        runner_module.StepChainRunnerKernel,
        "build_tuple_runner",
        _explode_build_tuple_runner,
    )

    result = op(*tensors)
    assert not isinstance(result, tuple)
    np.testing.assert_array_equal(result, expected)


def test_einop_inflate_like_broadcast_matches_inflate() -> None:
    b, c, r = axes("b", "c", "r")
    op = einop(ax[b, c], ax[b, c, r]).with_sizes(r=4)

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.broadcast_to(np.expand_dims(tensor, axis=2), (2, 3, 4))
    np.testing.assert_array_equal(result, expected)


def test_einop_reduce_default_sum_matches_reduce() -> None:
    b, c = axes("b", "c")
    op = einop(ax[b, c], ax[b])

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    expected = np.sum(tensor, axis=1)
    np.testing.assert_array_equal(result, expected)


def test_einop_reduce_by_callable_matches_reduce_plus_inflate_pipeline() -> None:
    b, c, r = axes("b", "c", "r")
    op = (
        einop(ax[b, c], ax[b, r])
        .with_sizes(r=2)
        .reduce_by(lambda x, *, axis: np.sum(x, axis=axis))
    )

    tensor = np.arange(2 * 3).reshape(2, 3)
    result = op(tensor)

    reduced = np.sum(tensor, axis=1)
    expected = np.broadcast_to(np.expand_dims(reduced, axis=1), (2, 2))
    np.testing.assert_array_equal(result, expected)


def test_contract_dot_product_to_scalar_executes_with_numpy() -> None:
    i = axes("i")[0]
    op = contract((ax[i], ax[i]), ax[()])

    left = np.arange(5)
    right = np.arange(5)
    result = op(left, right)

    expected = np.asarray(np.dot(left, right))
    assert result.shape == ()
    np.testing.assert_array_equal(result, expected)


def test_contract_rank_zero_identity_executes_with_numpy() -> None:
    op = contract(ax[()], ax[()])

    tensor = np.asarray(7)
    result = op(tensor)

    assert result.shape == ()
    np.testing.assert_array_equal(result, tensor)


def test_contract_trace_and_diagonal_match_numpy() -> None:
    i = axes("i")[0]
    op_trace = contract(ax[i, i], ax[()])
    op_diag = contract(ax[i, i], ax[i])

    tensor = np.arange(16).reshape(4, 4)
    trace_result = op_trace(tensor)
    diag_result = op_diag(tensor)

    trace_expected = np.einsum("ii->", tensor)
    diag_expected = np.einsum("ii->i", tensor)
    np.testing.assert_array_equal(trace_result, trace_expected)
    np.testing.assert_array_equal(diag_result, diag_expected)


def test_contract_three_input_chain_executes_with_numpy() -> None:
    a, b, c, d = axes("a", "b", "c", "d")
    op = contract((ax[a, b], ax[b, c], ax[c, d]), ax[a, d])

    x = np.arange(2 * 3).reshape(2, 3)
    y = np.arange(3 * 4).reshape(3, 4)
    z = np.arange(4 * 5).reshape(4, 5)
    result = op(x, y, z)

    expected = x @ y @ z
    np.testing.assert_array_equal(result, expected)


def test_contract_rejects_non_atomic_axis_expression_at_constructor() -> None:
    b, h, w, c = axes("b", "h", "w", "c")

    with pytest.raises(ValidationError) as error:
        _ = contract((ax[b, (h * w)], ax[(h * w), c]), ax[b, c])

    assert error.value.code == ErrorCode.CONTRACT_NON_ATOMIC_AXIS.value


def test_contract_rejects_output_axis_missing_from_inputs() -> None:
    i, j = axes("i", "j")
    op = contract(ax[i], ax[j]).with_sizes(j=3)
    tensor = np.arange(3)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value


def test_contract_rejects_duplicate_output_axis_names() -> None:
    i = axes("i")[0]
    op = contract(ax[i], ax[i, i])
    tensor = np.arange(3)

    with pytest.raises(ValidationError) as error:
        _ = op(tensor)

    assert error.value.code == ErrorCode.INCONSISTENT_DIMS.value
