from einf.steps.expand import ExpandRuntimeStep
from einf.steps.permute import PermuteRuntimeStep, build_permute_symbolic_program
from einf.steps.reshape import ReshapeRuntimeStep
from einf.steps.reshape.constants import ZERO_COPY_ALLOWED_RESHAPE_MODE
from einf.steps.reshape.runtime import run_reshape_program
from einf.tensor_types import TensorLike

from ..types import RuntimeSteps, TupleFusionRule, TupleRunner


def _compose_permutations(
    first: tuple[int, ...],
    second: tuple[int, ...],
    /,
) -> tuple[int, ...] | None:
    """Compose two permute_dims-style permutations into one permutation."""
    if len(first) != len(second):
        return None
    combined = tuple(first[source_axis] for source_axis in second)
    rank = len(combined)
    if tuple(sorted(combined)) != tuple(range(rank)):
        return None
    return combined


def build_permute_permute_tuple_runner(window: RuntimeSteps, /) -> TupleRunner | None:
    """Build fused tuple runner for one unary permute->permute pair."""
    if len(window) != 2:
        return None
    first_step = window[0]
    second_step = window[1]
    if not isinstance(first_step, PermuteRuntimeStep):
        return None
    if not isinstance(second_step, PermuteRuntimeStep):
        return None
    if first_step.input_arity != 1 or first_step.output_arity != 1:
        return None
    if second_step.input_arity != 1 or second_step.output_arity != 1:
        return None

    fused_permutation = _compose_permutations(
        first_step.program.permutation,
        second_step.program.permutation,
    )
    if fused_permutation is None:
        return None

    runtime_backend_ops = (
        second_step.runtime_backend_ops
        if second_step.runtime_backend_ops is not None
        else first_step.runtime_backend_ops
    )
    runtime_xp = (
        second_step.runtime_xp
        if second_step.runtime_xp is not None
        else first_step.runtime_xp
    )
    fused_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program(fused_permutation),
        runtime_backend_ops=runtime_backend_ops,
        runtime_xp=runtime_xp,
    )

    def run_fused_permute(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        return (fused_step.run_unary(runtime_tensors[0]),)

    return run_fused_permute


def build_permute_expand_tuple_runner(window: RuntimeSteps, /) -> TupleRunner | None:
    """Build fused tuple runner for one unary permute->expand pair."""
    if len(window) != 2:
        return None
    first_step = window[0]
    second_step = window[1]
    if not isinstance(first_step, PermuteRuntimeStep):
        return None
    if not isinstance(second_step, ExpandRuntimeStep):
        return None
    if first_step.input_arity != 1 or first_step.output_arity != 1:
        return None
    if second_step.input_arity != 1 or second_step.output_arity != 1:
        return None

    second_program = second_step.program.compiled
    target_shape_evaluator = second_step.target_shape_evaluator
    if second_program is None or target_shape_evaluator is None:
        return None

    fused_permutation = _compose_permutations(
        first_step.program.permutation,
        second_program.permutation,
    )
    if fused_permutation is None:
        return None
    has_non_identity_permutation = fused_permutation != tuple(
        range(len(fused_permutation))
    )

    backend_ops = (
        second_step.runtime_backend_ops
        if second_step.runtime_backend_ops is not None
        else first_step.runtime_backend_ops
    )
    xp = (
        second_step.runtime_xp
        if second_step.runtime_xp is not None
        else first_step.runtime_xp
    )
    if backend_ops is None and xp is None:
        return None

    first_permutation = first_step.program.permutation
    insert_axes = second_program.insert_axes

    def run_fused_permute_expand(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        input_tensor = runtime_tensors[0]
        input_shape = input_tensor.shape
        if len(input_shape) != len(first_permutation):
            return (second_step.run_unary(first_step.run_unary(input_tensor)),)

        first_output_shape = tuple(
            input_shape[input_index] for input_index in first_permutation
        )
        target_shape = target_shape_evaluator(first_output_shape)
        if target_shape is None:
            return (second_step.run_unary(first_step.run_unary(input_tensor)),)

        transformed = input_tensor
        try:
            if backend_ops is not None:
                if has_non_identity_permutation:
                    transformed = backend_ops.permute(transformed, fused_permutation)
                for output_index in insert_axes:
                    transformed = backend_ops.expand_dims(transformed, output_index)
                return (backend_ops.broadcast_to(transformed, target_shape),)

            assert xp is not None
            if has_non_identity_permutation:
                transformed = xp.permute_dims(transformed, fused_permutation)
            for output_index in insert_axes:
                transformed = xp.expand_dims(transformed, axis=output_index)
            return (xp.broadcast_to(transformed, target_shape),)
        except (TypeError, ValueError, RuntimeError):
            return (second_step.run_unary(first_step.run_unary(input_tensor)),)

    return run_fused_permute_expand


def build_reshape_reshape_tuple_runner(window: RuntimeSteps, /) -> TupleRunner | None:
    """Build fused tuple runner for one unary reshape->reshape pair."""
    if len(window) != 2:
        return None
    first_step = window[0]
    second_step = window[1]
    if not isinstance(first_step, ReshapeRuntimeStep):
        return None
    if not isinstance(second_step, ReshapeRuntimeStep):
        return None
    if first_step.input_arity != 1 or first_step.output_arity != 1:
        return None
    if second_step.input_arity != 1 or second_step.output_arity != 1:
        return None

    first_program = first_step.program
    second_program = second_step.program
    if (
        first_program.zero_copy_mode != ZERO_COPY_ALLOWED_RESHAPE_MODE
        or second_program.zero_copy_mode != ZERO_COPY_ALLOWED_RESHAPE_MODE
    ):
        return None
    if first_program.reject_not_a_view or second_program.reject_not_a_view:
        return None

    first_shape_evaluator = first_step.target_shape_evaluator
    second_shape_evaluator = second_step.target_shape_evaluator
    if first_shape_evaluator is None or second_shape_evaluator is None:
        return None

    backend_ops = (
        second_step.runtime_backend_ops
        if second_step.runtime_backend_ops is not None
        else first_step.runtime_backend_ops
    )
    xp = (
        second_step.runtime_xp
        if second_step.runtime_xp is not None
        else first_step.runtime_xp
    )

    def run_fused_reshape(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        tensor = runtime_tensors[0]
        first_target_shape = first_shape_evaluator(tensor.shape)
        if first_target_shape is None:
            return (second_step.run_unary(first_step.run_unary(tensor)),)
        second_target_shape = second_shape_evaluator(first_target_shape)
        if second_target_shape is None:
            return (second_step.run_unary(first_step.run_unary(tensor)),)
        if tensor.shape == second_target_shape:
            return (tensor,)
        if backend_ops is None and xp is None:
            return (second_step.run_unary(first_step.run_unary(tensor)),)
        try:
            return (
                run_reshape_program(
                    tensor=tensor,
                    target_shape=second_target_shape,
                    backend_ops=backend_ops,
                    xp=xp,
                    zero_copy_mode=ZERO_COPY_ALLOWED_RESHAPE_MODE,
                ),
            )
        except (TypeError, ValueError, RuntimeError):
            return (second_step.run_unary(first_step.run_unary(tensor)),)

    return run_fused_reshape


PERMUTE_EXPAND_RULE = TupleFusionRule(
    name="permute_expand",
    window_size=2,
    build_runner=build_permute_expand_tuple_runner,
)

PERMUTE_PERMUTE_RULE = TupleFusionRule(
    name="permute_permute",
    window_size=2,
    build_runner=build_permute_permute_tuple_runner,
)

RESHAPE_RESHAPE_RULE = TupleFusionRule(
    name="reshape_reshape",
    window_size=2,
    build_runner=build_reshape_reshape_tuple_runner,
)


__all__ = [
    "PERMUTE_EXPAND_RULE",
    "PERMUTE_PERMUTE_RULE",
    "RESHAPE_RESHAPE_RULE",
]
