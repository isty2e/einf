from collections.abc import Iterator

import numpy as np
import pytest

import einf.steps.permute as permute_step_module
import einf.steps.reshape as reshape_step_module
import einf.steps.reshape.compile as reshape_compile_module
from einf import ax, axes, rearrange
from einf.axis import AxisTerms
from einf.lowering.builders import rearrange as rearrange_builder_module


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    reshape_step_module.build_reshape_compiled_program.cache_clear()
    yield
    reshape_step_module.build_reshape_compiled_program.cache_clear()


def _disable_symbolic_unary_plan(
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
) -> None:
    _ = (lhs_terms, rhs_terms)
    return None


def _disable_symbolic_fastpath_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rearrange_builder_module,
        "build_reshape_compiled_program",
        _disable_symbolic_unary_plan,
    )
    monkeypatch.setattr(
        reshape_step_module,
        "build_reshape_compiled_program",
        _disable_symbolic_unary_plan,
    )


def _explode_compile_shape_node(**_kwargs: object) -> None:
    raise AssertionError("shape-node compile should not run on cached unary plan")


def test_rearrange_symbolic_unary_plan_cache_reuses_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d = axes("b", "n", "d")
    op = rearrange(ax[b, n, d], ax[b, (n * d)])

    first = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    first_result = op(first)
    np.testing.assert_array_equal(first_result, first.reshape(2, 12))

    monkeypatch.setattr(
        reshape_compile_module,
        "compile_shape_node",
        _explode_compile_shape_node,
    )

    second = np.arange(5 * 2 * 7).reshape(5, 2, 7)
    second_result = op(second)
    np.testing.assert_array_equal(second_result, second.reshape(5, 14))


def test_rearrange_non_unary_keeps_non_unary_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, m, d = axes("b", "n", "m", "d")
    op = rearrange((ax[b, n, d], ax[b, m, d]), ax[b, (n + m), d]).with_sizes(n=2, m=3)

    _disable_symbolic_fastpath_build(monkeypatch)

    left = np.arange(1 * 2 * 4).reshape(1, 2, 4)
    right = np.arange(100, 100 + (1 * 3 * 4)).reshape(1, 3, 4)
    result = op(left, right)

    expected = np.concatenate((left, right), axis=1)
    np.testing.assert_array_equal(result, expected)


def _explode_context_normalization(**_kwargs: object) -> None:
    raise AssertionError("context normalization should not run")


def test_rearrange_symbolic_unary_fast_path_skips_context_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b, n, d = axes("b", "n", "d")
    op = rearrange(ax[b, n, d], ax[b, d, n])

    monkeypatch.setattr(
        permute_step_module,
        "build_runtime_execution_context",
        _explode_context_normalization,
    )

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = op(tensor)
    np.testing.assert_array_equal(result, np.transpose(tensor, (0, 2, 1)))
