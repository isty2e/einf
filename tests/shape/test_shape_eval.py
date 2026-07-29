from collections.abc import Callable

from einf import axes
from einf.shape import (
    compile_fixed_rank_shape_evaluator,
    compile_shape_eval_fn,
    compile_shape_node,
    shape_node_axis_names,
)


def _make_fast_eval(offset: int) -> Callable[[tuple[int, ...], dict[str, int]], int]:
    """Build one deterministic fast evaluator for tests."""

    def eval_fn(input_shape: tuple[int, ...], explicit_sizes: dict[str, int]) -> int:
        return input_shape[0] + explicit_sizes["k"] + offset

    return eval_fn


def test_compile_fixed_rank_shape_evaluator_supports_arities_1_to_4() -> None:
    for arity in range(1, 5):
        fast_shape_eval_fns = tuple(_make_fast_eval(index) for index in range(arity))
        evaluator = compile_fixed_rank_shape_evaluator(
            fast_shape_eval_fns=fast_shape_eval_fns,
            lhs_rank=1,
            explicit_sizes={"k": 10},
        )

        assert evaluator is not None
        assert evaluator((3,)) == tuple(13 + index for index in range(arity))
        assert evaluator((3, 4)) is None


def test_compile_fixed_rank_shape_evaluator_rejects_unsupported_arity() -> None:
    fast_shape_eval_fns = tuple(_make_fast_eval(index) for index in range(5))
    evaluator = compile_fixed_rank_shape_evaluator(
        fast_shape_eval_fns=fast_shape_eval_fns,
        lhs_rank=1,
        explicit_sizes={"k": 10},
    )

    assert evaluator is None


def test_compile_shape_node_collects_unresolved_axis_names() -> None:
    a, b = axes("a", "b")
    node = compile_shape_node(term=(a + b) * 2, axis_index_by_name={"a": 0})

    assert node is not None
    assert shape_node_axis_names(node) == {"b"}


def test_compile_shape_node_supports_direct_eval() -> None:
    a, b = axes("a", "b")
    node = compile_shape_node(term=a * b + 1, axis_index_by_name={"a": 0})

    assert node is not None
    eval_fn = compile_shape_eval_fn(node)
    assert eval_fn((3,), {"b": 4}) == 13
    assert eval_fn((3,), {}) is None
