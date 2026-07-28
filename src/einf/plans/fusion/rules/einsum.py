from einf.steps.axis_slice import AxisSliceRuntimeStep
from einf.steps.einsum import EinsumRuntimeStep
from einf.tensor_types import TensorLike

from ..types import RuntimeStepFusionRule, RuntimeSteps, TupleRunner


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

    run_einsum = first_step.run_binary
    run_axis_slice = second_step.run

    def run_fused_binary_split(
        runtime_tensors: tuple[TensorLike, ...], /
    ) -> tuple[TensorLike, ...]:
        intermediate = run_einsum(runtime_tensors[0], runtime_tensors[1])
        return run_axis_slice((intermediate,))

    return run_fused_binary_split


EINSUM_AXIS_SLICE_RULE = RuntimeStepFusionRule(
    name="einsum_axis_slice",
    window_size=2,
    build_tuple_runner=build_einsum_axis_slice_tuple_runner,
)


__all__ = ["EINSUM_AXIS_SLICE_RULE"]
