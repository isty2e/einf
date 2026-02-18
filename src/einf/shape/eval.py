from collections.abc import Callable

from .nodes import ShapeAxisName, ShapeBinary, ShapeDimRef, ShapeLiteral, ShapeNode


def compile_fixed_rank_shape_evaluator(
    *,
    fast_shape_eval_fns: tuple[Callable[[tuple[int, ...], dict[str, int]], int], ...],
    lhs_rank: int,
    explicit_sizes: dict[str, int],
) -> Callable[[tuple[int, ...]], tuple[int, ...] | None] | None:
    """Compile fast fixed-rank target-shape evaluators for arities 1..4."""
    explicit_sizes_ref = explicit_sizes
    if len(fast_shape_eval_fns) == 1:
        shape_eval_0 = fast_shape_eval_fns[0]
        return (
            lambda input_shape, shape_eval_0=shape_eval_0, explicit_sizes_ref=explicit_sizes_ref: (
                None
                if len(input_shape) != lhs_rank
                else (shape_eval_0(input_shape, explicit_sizes_ref),)
            )
        )
    if len(fast_shape_eval_fns) == 2:
        shape_eval_0 = fast_shape_eval_fns[0]
        shape_eval_1 = fast_shape_eval_fns[1]
        return (
            lambda input_shape, shape_eval_0=shape_eval_0, shape_eval_1=shape_eval_1, explicit_sizes_ref=explicit_sizes_ref: (
                None
                if len(input_shape) != lhs_rank
                else (
                    shape_eval_0(input_shape, explicit_sizes_ref),
                    shape_eval_1(input_shape, explicit_sizes_ref),
                )
            )
        )
    if len(fast_shape_eval_fns) == 3:
        shape_eval_0 = fast_shape_eval_fns[0]
        shape_eval_1 = fast_shape_eval_fns[1]
        shape_eval_2 = fast_shape_eval_fns[2]
        return (
            lambda input_shape, shape_eval_0=shape_eval_0, shape_eval_1=shape_eval_1, shape_eval_2=shape_eval_2, explicit_sizes_ref=explicit_sizes_ref: (
                None
                if len(input_shape) != lhs_rank
                else (
                    shape_eval_0(input_shape, explicit_sizes_ref),
                    shape_eval_1(input_shape, explicit_sizes_ref),
                    shape_eval_2(input_shape, explicit_sizes_ref),
                )
            )
        )
    if len(fast_shape_eval_fns) == 4:
        shape_eval_0 = fast_shape_eval_fns[0]
        shape_eval_1 = fast_shape_eval_fns[1]
        shape_eval_2 = fast_shape_eval_fns[2]
        shape_eval_3 = fast_shape_eval_fns[3]
        return (
            lambda input_shape, shape_eval_0=shape_eval_0, shape_eval_1=shape_eval_1, shape_eval_2=shape_eval_2, shape_eval_3=shape_eval_3, explicit_sizes_ref=explicit_sizes_ref: (
                None
                if len(input_shape) != lhs_rank
                else (
                    shape_eval_0(input_shape, explicit_sizes_ref),
                    shape_eval_1(input_shape, explicit_sizes_ref),
                    shape_eval_2(input_shape, explicit_sizes_ref),
                    shape_eval_3(input_shape, explicit_sizes_ref),
                )
            )
        )
    return None


def compile_shape_eval_fn(
    node: ShapeNode,
    /,
) -> Callable[[tuple[int, ...], dict[str, int]], int | None]:
    """Compile one shape node into a safe evaluator that can return None."""
    if isinstance(node, ShapeLiteral):
        literal_value = node.value
        return lambda input_shape, explicit_sizes, literal_value=literal_value: (
            literal_value
        )
    if isinstance(node, ShapeDimRef):
        dim_index = node.index

        def eval_dim(
            input_shape: tuple[int, ...],
            explicit_sizes: dict[str, int],
        ) -> int | None:
            _ = explicit_sizes
            if dim_index < 0 or dim_index >= len(input_shape):
                return None
            return input_shape[dim_index]

        return eval_dim
    if isinstance(node, ShapeAxisName):
        axis_name = node.name
        return lambda input_shape, explicit_sizes, axis_name=axis_name: (
            explicit_sizes.get(axis_name)
        )
    if isinstance(node, ShapeBinary):
        left_eval = compile_shape_eval_fn(node.left)
        right_eval = compile_shape_eval_fn(node.right)
        if node.operator == "+":

            def eval_add(
                input_shape: tuple[int, ...],
                explicit_sizes: dict[str, int],
            ) -> int | None:
                left_value = left_eval(input_shape, explicit_sizes)
                if left_value is None:
                    return None
                right_value = right_eval(input_shape, explicit_sizes)
                if right_value is None:
                    return None
                return left_value + right_value

            return eval_add
        if node.operator == "*":

            def eval_mul(
                input_shape: tuple[int, ...],
                explicit_sizes: dict[str, int],
            ) -> int | None:
                left_value = left_eval(input_shape, explicit_sizes)
                if left_value is None:
                    return None
                right_value = right_eval(input_shape, explicit_sizes)
                if right_value is None:
                    return None
                return left_value * right_value

            return eval_mul

    def eval_unsupported(
        input_shape: tuple[int, ...],
        explicit_sizes: dict[str, int],
    ) -> int | None:
        _ = input_shape
        _ = explicit_sizes
        return None

    return eval_unsupported


def compile_fast_shape_eval_fn(
    node: ShapeNode,
    /,
) -> Callable[[tuple[int, ...], dict[str, int]], int] | None:
    """Compile one shape node into an unchecked fast evaluator."""
    if isinstance(node, ShapeLiteral):
        literal_value = node.value
        return lambda input_shape, explicit_sizes, literal_value=literal_value: (
            literal_value
        )
    if isinstance(node, ShapeDimRef):
        dim_index = node.index
        return lambda input_shape, explicit_sizes, dim_index=dim_index: input_shape[
            dim_index
        ]
    if isinstance(node, ShapeAxisName):
        axis_name = node.name
        return lambda input_shape, explicit_sizes, axis_name=axis_name: explicit_sizes[
            axis_name
        ]
    if not isinstance(node, ShapeBinary):
        return None

    if isinstance(node.left, ShapeDimRef) and isinstance(node.right, ShapeDimRef):
        left_index = node.left.index
        right_index = node.right.index
        if node.operator == "+":
            return (
                lambda input_shape, explicit_sizes, left_index=left_index, right_index=right_index: (
                    input_shape[left_index] + input_shape[right_index]
                )
            )
        if node.operator == "*":
            return (
                lambda input_shape, explicit_sizes, left_index=left_index, right_index=right_index: (
                    input_shape[left_index] * input_shape[right_index]
                )
            )
        return None

    if isinstance(node.left, ShapeDimRef) and isinstance(node.right, ShapeLiteral):
        left_index = node.left.index
        right_literal = node.right.value
        if node.operator == "+":
            return (
                lambda input_shape, explicit_sizes, left_index=left_index, right_literal=right_literal: (
                    input_shape[left_index] + right_literal
                )
            )
        if node.operator == "*":
            return (
                lambda input_shape, explicit_sizes, left_index=left_index, right_literal=right_literal: (
                    input_shape[left_index] * right_literal
                )
            )
        return None

    if isinstance(node.left, ShapeLiteral) and isinstance(node.right, ShapeDimRef):
        left_literal = node.left.value
        right_index = node.right.index
        if node.operator == "+":
            return (
                lambda input_shape, explicit_sizes, left_literal=left_literal, right_index=right_index: (
                    left_literal + input_shape[right_index]
                )
            )
        if node.operator == "*":
            return (
                lambda input_shape, explicit_sizes, left_literal=left_literal, right_index=right_index: (
                    left_literal * input_shape[right_index]
                )
            )
        return None

    left_fast_eval = compile_fast_shape_eval_fn(node.left)
    right_fast_eval = compile_fast_shape_eval_fn(node.right)
    if left_fast_eval is None or right_fast_eval is None:
        return None
    if node.operator == "+":
        return (
            lambda input_shape, explicit_sizes, left_fast_eval=left_fast_eval, right_fast_eval=right_fast_eval: (
                left_fast_eval(
                    input_shape,
                    explicit_sizes,
                )
                + right_fast_eval(
                    input_shape,
                    explicit_sizes,
                )
            )
        )
    if node.operator == "*":
        return (
            lambda input_shape, explicit_sizes, left_fast_eval=left_fast_eval, right_fast_eval=right_fast_eval: (
                left_fast_eval(
                    input_shape,
                    explicit_sizes,
                )
                * right_fast_eval(
                    input_shape,
                    explicit_sizes,
                )
            )
        )
    return None


__all__ = [
    "compile_fast_shape_eval_fn",
    "compile_fixed_rank_shape_evaluator",
    "compile_shape_eval_fn",
]
