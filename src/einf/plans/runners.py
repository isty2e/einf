from abc import ABC, abstractmethod
from dataclasses import dataclass

from einf.plans.fusion import RuntimeStepFusions, RuntimeSteps, TupleRunner
from einf.plans.routing import route_outputs
from einf.steps.base import RuntimeStep, StepProgram
from einf.tensor_types import TensorLike

from .fusion import SingleOutputRunner


class RunnerKernel(ABC):
    """Runtime-runner kernel categorized by execution source."""

    @property
    @abstractmethod
    def input_arity(self) -> int:
        """Return runner input arity."""

    @property
    @abstractmethod
    def output_arity(self) -> int:
        """Return runner output arity."""

    @abstractmethod
    def build_tuple_runner(self) -> TupleRunner:
        """Project this kernel to tuple-output execution."""

    def build_single_output_runner(self) -> SingleOutputRunner:
        """Project this kernel to single-output execution."""
        if self.output_arity != 1:
            raise ValueError(
                "runtime step chain output arity mismatch: "
                f"expected 1, got {self.output_arity}"
            )
        tuple_runner = self.build_tuple_runner()

        def run_single_output(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
            outputs = tuple_runner(runtime_tensors)
            if len(outputs) != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    f"expected 1, got {len(outputs)}"
                )
            return outputs[0]

        return run_single_output


@dataclass(frozen=True, slots=True)
class RouteRunnerKernel(RunnerKernel):
    """Runner kernel for route-only symbolic plans."""

    _input_arity: int
    output_indices: tuple[int, ...]

    @property
    def input_arity(self) -> int:
        """Return route-runner input arity."""
        return self._input_arity

    @property
    def output_arity(self) -> int:
        """Return projected output arity."""
        return len(self.output_indices)

    def build_tuple_runner(self) -> TupleRunner:
        """Build tuple-output route runner."""
        output_indices = self.output_indices

        def run_route(runtime_tensors: tuple[TensorLike, ...], /) -> tuple[TensorLike, ...]:
            return route_outputs(
                tensors=runtime_tensors,
                output_indices=output_indices,
            )

        return run_route

    def build_single_output_runner(self) -> SingleOutputRunner:
        """Build single-output route runner."""
        if len(self.output_indices) != 1:
            raise ValueError(
                "runtime step chain output arity mismatch: "
                f"expected 1, got {len(self.output_indices)}"
            )
        output_index = self.output_indices[0]

        def run_route(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
            return runtime_tensors[output_index]

        return run_route


def run_runtime_step(
    *,
    runtime_step: RuntimeStep[StepProgram],
    current: tuple[TensorLike, ...],
) -> tuple[TensorLike, ...]:
    """Run one runtime step with unary/binary dispatch shortcuts."""
    if (
        len(current) == 1
        and runtime_step.input_arity == 1
        and runtime_step.output_arity == 1
    ):
        return (runtime_step.run_unary(current[0]),)
    if (
        len(current) == 2
        and runtime_step.input_arity == 2
        and runtime_step.output_arity == 1
    ):
        return (runtime_step.run_binary(current[0], current[1]),)
    return runtime_step.run(current)


@dataclass(frozen=True, slots=True)
class StepChainRunnerKernel(RunnerKernel):
    """Runner kernel for specialized runtime-step chains."""

    _input_arity: int
    _output_arity: int
    runtime_steps: RuntimeSteps
    fusions: RuntimeStepFusions

    @property
    def input_arity(self) -> int:
        """Return step-chain input arity."""
        return self._input_arity

    @property
    def output_arity(self) -> int:
        """Return step-chain output arity."""
        return self._output_arity

    def build_tuple_runner(self) -> TupleRunner:
        """Build tuple-output runner for one runtime-step chain."""
        runtime_steps = self.runtime_steps
        if not runtime_steps:

            def run_identity(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> tuple[TensorLike, ...]:
                return runtime_tensors

            return run_identity

        if (
            len(self.fusions) == 1
            and self.fusions[0].start == 0
            and self.fusions[0].stop == len(runtime_steps)
        ):
            return self.fusions[0].tuple_runner

        if len(runtime_steps) == 1:
            runtime_step = runtime_steps[0]
            if runtime_step.input_arity == 1 and runtime_step.output_arity == 1:
                run_unary_method = runtime_step.run_unary

                def run_unary(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> tuple[TensorLike, ...]:
                    return (run_unary_method(runtime_tensors[0]),)

                return run_unary

            if runtime_step.input_arity == 2 and runtime_step.output_arity == 1:
                run_binary_method = runtime_step.run_binary

                def run_binary(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> tuple[TensorLike, ...]:
                    return (run_binary_method(runtime_tensors[0], runtime_tensors[1]),)

                return run_binary

            def run_single(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> tuple[TensorLike, ...]:
                return runtime_step.run(runtime_tensors)

            return run_single

        fusion_by_start = {fusion.start: fusion for fusion in self.fusions}

        def run_chain(
            runtime_tensors: tuple[TensorLike, ...], /
        ) -> tuple[TensorLike, ...]:
            current = runtime_tensors
            step_index = 0
            while step_index < len(runtime_steps):
                fusion = fusion_by_start.get(step_index)
                if fusion is not None:
                    if len(current) != fusion.input_arity:
                        raise ValueError(
                            "runtime step fusion input arity mismatch: "
                            f"expected {fusion.input_arity}, got {len(current)}"
                        )
                    current = fusion.tuple_runner(current)
                    step_index = fusion.stop
                    continue
                current = run_runtime_step(
                    runtime_step=runtime_steps[step_index],
                    current=current,
                )
                step_index += 1
            return current

        return run_chain

    def build_single_output_runner(self) -> SingleOutputRunner:
        """Build single-output runner for one runtime-step chain."""
        if self.output_arity != 1:
            raise ValueError(
                "runtime step chain output arity mismatch: "
                f"expected 1, got {self.output_arity}"
            )

        runtime_steps = self.runtime_steps
        if not runtime_steps:
            if self.input_arity != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    "cannot emit one output without runtime steps"
                )

            def run_identity(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
                return runtime_tensors[0]

            return run_identity

        if (
            len(self.fusions) == 1
            and self.fusions[0].start == 0
            and self.fusions[0].stop == len(runtime_steps)
            and self.fusions[0].output_arity == 1
        ):
            fused_runner = self.fusions[0].tuple_runner

            def run_fused_single_output(
                runtime_tensors: tuple[TensorLike, ...], /
            ) -> TensorLike:
                outputs = fused_runner(runtime_tensors)
                if len(outputs) != 1:
                    raise ValueError(
                        "runtime step fusion output arity mismatch: "
                        f"expected 1, got {len(outputs)}"
                    )
                return outputs[0]

            return run_fused_single_output

        if len(runtime_steps) == 1:
            runtime_step = runtime_steps[0]
            if runtime_step.input_arity == 1 and runtime_step.output_arity == 1:
                run_unary_method = runtime_step.run_unary

                def run_unary_single_output(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> TensorLike:
                    return run_unary_method(runtime_tensors[0])

                return run_unary_single_output

            if runtime_step.input_arity == 2 and runtime_step.output_arity == 1:
                run_binary_method = runtime_step.run_binary

                def run_binary_single_output(
                    runtime_tensors: tuple[TensorLike, ...], /
                ) -> TensorLike:
                    return run_binary_method(runtime_tensors[0], runtime_tensors[1])

                return run_binary_single_output

            def run_single_output(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
                outputs = runtime_step.run(runtime_tensors)
                if len(outputs) != 1:
                    raise ValueError(
                        "runtime step chain output arity mismatch: "
                        f"expected 1, got {len(outputs)}"
                    )
                return outputs[0]

            return run_single_output

        tuple_runner = self.build_tuple_runner()

        def run_chain(runtime_tensors: tuple[TensorLike, ...], /) -> TensorLike:
            outputs = tuple_runner(runtime_tensors)
            if len(outputs) != 1:
                raise ValueError(
                    "runtime step chain output arity mismatch: "
                    f"expected 1, got {len(outputs)}"
                )
            return outputs[0]

        return run_chain


__all__ = [
    "RouteRunnerKernel",
    "RunnerKernel",
    "StepChainRunnerKernel",
    "run_runtime_step",
]
