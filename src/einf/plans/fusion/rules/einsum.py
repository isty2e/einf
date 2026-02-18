from collections.abc import Callable

import opt_einsum

from einf.steps.axis_slice import AxisSliceRuntimeStep
from einf.steps.einsum import EinsumRuntimeStep
from einf.steps.einsum.native import try_native_contract_einsum
from einf.steps.einsum.step import (
    _EinsumEquationExecutor,
    _is_binary_matmul_equation,
    _prefer_native_matmul,
)
from einf.tensor_types import TensorLike

from ..types import RuntimeSteps, TupleFusionRule, TupleRunner


def _slice_tensor_by_sizes_unchecked(
    *,
    tensor: TensorLike,
    split_axis: int,
    split_sizes: tuple[int, ...],
) -> tuple[TensorLike, ...]:
    """Slice one tensor into contiguous partitions without runtime validation."""
    split_method = getattr(tensor, "split", None)
    if callable(split_method):
        try:
            split_outputs = split_method(split_sizes, dim=split_axis)
            if isinstance(split_outputs, tuple):
                return split_outputs
            if isinstance(split_outputs, list):
                return tuple(split_outputs)
        except (TypeError, ValueError):
            pass

    input_rank = len(tensor.shape)
    prefix = (slice(None),) * split_axis
    suffix = (slice(None),) * (input_rank - split_axis - 1)
    outputs: list[TensorLike] = []
    offset = 0
    for size in split_sizes:
        next_offset = offset + size
        outputs.append(tensor[prefix + (slice(offset, next_offset),) + suffix])
        offset = next_offset
    return tuple(outputs)


def _binary_expression_key(
    lhs: TensorLike,
    rhs: TensorLike,
    /,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Build hashable binary operand shape key for contract expression cache."""
    lhs_shape = getattr(lhs, "shape", None)
    rhs_shape = getattr(rhs, "shape", None)
    if not isinstance(lhs_shape, tuple) or not isinstance(rhs_shape, tuple):
        return None
    if any(type(dim) is not int for dim in lhs_shape):
        return None
    if any(type(dim) is not int for dim in rhs_shape):
        return None
    return lhs_shape, rhs_shape


def _resolve_binary_order(
    *,
    chain_order: tuple[int, ...],
    carrier_index: int | None,
) -> tuple[bool, int, int] | None:
    """Resolve binary operand order and chain-mode flag for one einsum step."""
    if not chain_order:
        return False, 0, 1
    if len(chain_order) != 1 or not isinstance(carrier_index, int):
        return None
    next_index = chain_order[0]
    if carrier_index == 0 and next_index == 1:
        return True, 0, 1
    if carrier_index == 1 and next_index == 0:
        return True, 1, 0
    return None


def _build_native_binary_einsum_runner(
    *,
    equation: str,
    executor: _EinsumEquationExecutor,
    allow_native_matmul: bool,
) -> Callable[[TensorLike, TensorLike], TensorLike] | None:
    """Build direct native binary runner for one equation when determinable."""
    module_matmul = executor.native_module_matmul
    if (
        allow_native_matmul
        and module_matmul is not None
        and _is_binary_matmul_equation(equation)
    ):
        module_einsum = executor.native_module_einsum

        def run_matmul(lhs: TensorLike, rhs: TensorLike, /) -> TensorLike:
            if not _prefer_native_matmul(operands=(lhs, rhs)):
                if module_einsum is not None:
                    return module_einsum(equation, lhs, rhs)
            return module_matmul(lhs, rhs)

        return run_matmul

    module_einsum = executor.native_module_einsum
    if module_einsum is not None:

        def run_module_einsum(lhs: TensorLike, rhs: TensorLike, /) -> TensorLike:
            return module_einsum(equation, lhs, rhs)

        return run_module_einsum

    namespace_einsum = executor.native_namespace_einsum
    if namespace_einsum is not None:

        def run_namespace_einsum(lhs: TensorLike, rhs: TensorLike, /) -> TensorLike:
            try:
                return namespace_einsum(equation, lhs, rhs)
            except Exception:
                native_output = try_native_contract_einsum(
                    equation=equation,
                    tensors=(lhs, rhs),
                    namespace=executor.profile.namespace,
                )
                if native_output is None:
                    raise
                return native_output

        return run_namespace_einsum

    return None


def build_einsum_binary_tuple_runner(window: RuntimeSteps, /) -> TupleRunner | None:
    """Build fused tuple runner for one binary einsum runtime step."""
    if len(window) != 1:
        return None
    (runtime_step,) = window
    if not isinstance(runtime_step, EinsumRuntimeStep):
        return None
    if runtime_step.input_arity != 2 or runtime_step.output_arity != 1:
        return None

    equations = runtime_step.program.equations
    if len(equations) != 1:
        return None
    resolved_order = _resolve_binary_order(
        chain_order=runtime_step.program.chain_order,
        carrier_index=runtime_step.program.carrier_index,
    )
    if resolved_order is None:
        return None
    chain_mode, lhs_index, rhs_index = resolved_order
    executor = runtime_step._resolve_executor()
    equation = equations[0]
    allow_native_matmul = runtime_step.program.allow_native_matmul
    native_binary_runner = _build_native_binary_einsum_runner(
        equation=equation,
        executor=executor,
        allow_native_matmul=allow_native_matmul,
    )

    if native_binary_runner is not None:

        def run_binary(
            runtime_tensors: tuple[TensorLike, ...], /
        ) -> tuple[TensorLike, ...]:
            lhs_tensor = runtime_tensors[lhs_index]
            rhs_tensor = runtime_tensors[rhs_index]
            return (native_binary_runner(lhs_tensor, rhs_tensor),)

        return run_binary

    expression_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        Callable[..., TensorLike],
    ] = {}

    def run_binary(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        lhs_tensor = runtime_tensors[lhs_index]
        rhs_tensor = runtime_tensors[rhs_index]
        expression_key = _binary_expression_key(lhs_tensor, rhs_tensor)
        if expression_key is None:
            return (
                executor.run(
                    equation,
                    (lhs_tensor, rhs_tensor),
                    chain_mode,
                    False,
                ),
            )
        expression = expression_cache.get(expression_key)
        if expression is None:
            expression = opt_einsum.contract_expression(
                equation,
                expression_key[0],
                expression_key[1],
                optimize="auto",
            )
            expression_cache[expression_key] = expression
        return (expression(lhs_tensor, rhs_tensor),)

    return run_binary


def build_einsum_axis_slice_tuple_runner(
    window: RuntimeSteps,
    /,
) -> TupleRunner | None:
    """Build fused tuple runner for one einsum->axis_slice runtime-step pair."""
    if len(window) != 2:
        return None
    first_step = window[0]
    second_step = window[1]
    if not isinstance(first_step, EinsumRuntimeStep):
        return None
    if not isinstance(second_step, AxisSliceRuntimeStep):
        return None
    if first_step.input_arity != 2 or first_step.output_arity != 1:
        return None
    if second_step.input_arity != 1:
        return None

    equations = first_step.program.equations
    if len(equations) != 1:
        return None
    resolved_order = _resolve_binary_order(
        chain_order=first_step.program.chain_order,
        carrier_index=first_step.program.carrier_index,
    )
    if resolved_order is None:
        return None
    use_chain_mode, lhs_index, rhs_index = resolved_order

    split_axis = second_step.program.split_axis
    split_sizes = second_step.precomputed_split_sizes
    if split_axis < 0 or split_sizes is None:
        return None

    executor = first_step._resolve_executor()
    equation = equations[0]
    allow_native_matmul = first_step.program.allow_native_matmul
    native_binary_runner = _build_native_binary_einsum_runner(
        equation=equation,
        executor=executor,
        allow_native_matmul=allow_native_matmul,
    )
    if native_binary_runner is not None:

        def run_fused_binary_split(
            runtime_tensors: tuple[TensorLike, ...], /
        ) -> tuple[TensorLike, ...]:
            lhs_tensor = runtime_tensors[lhs_index]
            rhs_tensor = runtime_tensors[rhs_index]
            intermediate = native_binary_runner(lhs_tensor, rhs_tensor)
            return _slice_tensor_by_sizes_unchecked(
                tensor=intermediate,
                split_axis=split_axis,
                split_sizes=split_sizes,
            )

        return run_fused_binary_split

    expression_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        Callable[..., TensorLike],
    ] = {}

    def run_fused_binary_split(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        lhs_tensor = runtime_tensors[lhs_index]
        rhs_tensor = runtime_tensors[rhs_index]
        expression_key = _binary_expression_key(lhs_tensor, rhs_tensor)
        if expression_key is None:
            intermediate = executor.run(
                equation,
                (lhs_tensor, rhs_tensor),
                use_chain_mode,
                allow_native_matmul,
            )
        else:
            expression = expression_cache.get(expression_key)
            if expression is None:
                expression = opt_einsum.contract_expression(
                    equation,
                    expression_key[0],
                    expression_key[1],
                    optimize="auto",
                )
                expression_cache[expression_key] = expression
            intermediate = expression(lhs_tensor, rhs_tensor)
        return _slice_tensor_by_sizes_unchecked(
            tensor=intermediate,
            split_axis=split_axis,
            split_sizes=split_sizes,
        )

    return run_fused_binary_split


EINSUM_AXIS_SLICE_RULE = TupleFusionRule(
    name="einsum_axis_slice",
    window_size=2,
    build_runner=build_einsum_axis_slice_tuple_runner,
)

EINSUM_BINARY_RULE = TupleFusionRule(
    name="einsum_binary",
    window_size=1,
    build_runner=build_einsum_binary_tuple_runner,
)


__all__ = [
    "EINSUM_AXIS_SLICE_RULE",
    "EINSUM_BINARY_RULE",
]
