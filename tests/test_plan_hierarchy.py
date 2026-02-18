from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest

from einf import ax, axes
from einf.axis import AxisSide
from einf.backend import BACKEND_RESOLVER
from einf.lowering import DefaultLoweringProgram, StaticLoweringProgram
from einf.plans.abstract import AbstractPlan
from einf.plans.context import PlanSelectionContext
from einf.plans.scoring import SymbolicPlanScore
from einf.plans.symbolic import SymbolicPlan
from einf.steps.axis_slice import AxisSliceSymbolicStep
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    StepProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.steps.concat import ConcatSymbolicStep
from einf.steps.einsum import (
    EinsumRuntimeStep,
    EinsumSymbolicStep,
    build_einsum_symbolic_program_from_equations,
)
from einf.steps.expand import (
    ExpandRuntimeStep,
    ExpandSymbolicStep,
    build_expand_symbolic_program,
)
from einf.steps.permute import (
    AxisPermuteSymbolicStep,
    PermuteRuntimeStep,
    PermuteSymbolicStep,
    build_axis_permute_symbolic_program,
)
from einf.steps.reduce import ReduceSymbolicStep
from einf.steps.reshape import ReshapeSymbolicStep
from einf.tensor_types import TensorLike


@dataclass(frozen=True, slots=True)
class _TestProgram(StepProgram):
    token: str = "test-program"


@dataclass(frozen=True, slots=True)
class _IdentityRuntimeStep(RuntimeStep[_TestProgram]):
    name: str
    input_arity: int
    output_arity: int
    program: _TestProgram = field(default_factory=_TestProgram)

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        return tensors


@dataclass(frozen=True, slots=True)
class _IdentitySymbolicStep(SymbolicStep[_TestProgram]):
    name: str
    input_arity: int
    output_arity: int
    program: _TestProgram = field(default_factory=_TestProgram)

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        _ = context
        return _IdentityRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=0,
            allocation_count=0,
            kernel_count=1,
        )

    def specialization_depends_on_input_shapes(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _CountingSymbolicStep(SymbolicStep[_TestProgram]):
    name: str
    input_arity: int
    output_arity: int
    counter_key: str
    program: _TestProgram = field(default_factory=_TestProgram)
    calls: ClassVar[dict[str, int]] = {}

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        _ = context
        current = self.calls.get(self.counter_key, 0)
        self.calls[self.counter_key] = current + 1
        return _IdentityRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=0,
            allocation_count=0,
            kernel_count=1,
        )

    def specialization_depends_on_input_shapes(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class _ScoredSymbolicStep(SymbolicStep[_TestProgram]):
    name: str
    input_arity: int
    output_arity: int
    step_score: SymbolicStepScore
    program: _TestProgram = field(default_factory=_TestProgram)

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        _ = context
        return _IdentityRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return self.step_score

    def specialization_depends_on_input_shapes(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _CountingScoreSymbolicStep(SymbolicStep[_TestProgram]):
    name: str
    input_arity: int
    output_arity: int
    counter_key: str
    step_score: SymbolicStepScore
    program: _TestProgram = field(default_factory=_TestProgram)
    calls: ClassVar[dict[str, int]] = {}

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        _ = context
        return _IdentityRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        current = self.calls.get(self.counter_key, 0)
        self.calls[self.counter_key] = current + 1
        return self.step_score

    def specialization_depends_on_input_shapes(self) -> bool:
        return False


def _unary_side() -> tuple[AxisSide, AxisSide]:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n, d], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, d, n], side_name="rhs")
    return lhs, rhs


def test_abstract_plan_delegates_to_lowering_program() -> None:
    lhs, rhs = _unary_side()
    symbolic = SymbolicPlan(
        kind="rearrange",
        input_arity=1,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="identity",
                input_arity=1,
                output_arity=1,
            ),
        ),
    )
    lowering = StaticLoweringProgram(candidates=(symbolic,))
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=lowering,
    )

    candidates = abstract.symbolic_candidates
    assert candidates == (symbolic,)


def test_symbolic_plan_execute_specializes_all_steps() -> None:
    plan = SymbolicPlan(
        kind="identity",
        input_arity=1,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="identity",
                input_arity=1,
                output_arity=1,
            ),
        ),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )
    tensors = (np.zeros((2, 3)),)

    outputs = plan.execute(context, tensors)
    assert len(outputs) == 1
    np.testing.assert_array_equal(np.asarray(outputs[0]), np.zeros((2, 3)))


def test_symbolic_plan_execute_runs_steps_in_order() -> None:
    plan = SymbolicPlan(
        kind="identity",
        input_arity=1,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="identity",
                input_arity=1,
                output_arity=1,
            ),
        ),
    )
    tensor = np.arange(6).reshape(2, 3)
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )

    outputs = plan.execute(context, (tensor,))
    assert len(outputs) == 1
    np.testing.assert_array_equal(np.asarray(outputs[0]), tensor)


def test_plan_construction_rejects_arity_mismatch() -> None:
    try:
        _ = SymbolicPlan(
            kind="bad",
            input_arity=1,
            output_arity=1,
            steps=(
                _IdentitySymbolicStep(
                    name="bad-step",
                    input_arity=2,
                    output_arity=1,
                ),
            ),
        )
    except ValueError as error:
        assert "expects 2 inputs" in str(error)
    else:
        raise AssertionError("expected ValueError for symbolic arity mismatch")


def test_symbolic_step_subclass_requires_program_annotation() -> None:
    with pytest.raises(TypeError, match="must annotate `program`"):

        @dataclass(frozen=True, slots=True)
        class _MissingProgramSymbolicStep(SymbolicStep[StepProgram]):
            name: str
            input_arity: int
            output_arity: int

            def specialize(
                self,
                context: RuntimeSpecializationContext,
                /,
            ) -> RuntimeStep:
                _ = context
                return _IdentityRuntimeStep(
                    name=self.name,
                    input_arity=self.input_arity,
                    output_arity=self.output_arity,
                )

            def score(
                self,
                context: PlanSelectionContext,
                /,
            ) -> SymbolicStepScore:
                _ = context
                return SymbolicStepScore(
                    peak_einsum_numel=0,
                    materialize_numel=0,
                    allocation_count=0,
                    kernel_count=1,
                )


def test_runtime_step_subclass_requires_program_annotation() -> None:
    with pytest.raises(TypeError, match="must annotate `program`"):

        @dataclass(frozen=True, slots=True)
        class _MissingProgramRuntimeStep(RuntimeStep[StepProgram]):
            name: str
            input_arity: int
            output_arity: int

            def run(
                self,
                tensors: tuple[TensorLike, ...],
                /,
            ) -> tuple[TensorLike, ...]:
                return tensors


def test_default_lowering_program_emits_expected_symbolic_steps() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    lowering = DefaultLoweringProgram()

    def lower_one(op_name: str, lhs_side: AxisSide, rhs_side: AxisSide) -> SymbolicPlan:
        ir_program = lowering.ir_program(
            op_name=op_name,
            lhs=lhs_side,
            rhs=rhs_side,
            explicit_sizes_items=(),
        )
        return lowering.symbolic_candidates(
            ir_program=ir_program,
            explicit_sizes_items=(),
        )[0]

    plan = lower_one("contract", lhs, rhs)
    assert isinstance(plan.steps[0], EinsumSymbolicStep)
    assert plan.input_arity == 2
    assert plan.output_arity == 1

    unary_lhs = AxisSide.from_spec(ax[b, n, d], side_name="lhs")
    unary_rhs = AxisSide.from_spec(ax[b, d, n], side_name="rhs")
    rearrange_plan = lower_one("rearrange", unary_lhs, unary_rhs)
    assert isinstance(rearrange_plan.steps[0], PermuteSymbolicStep)

    split_lhs = AxisSide.from_spec(ax[b, (n + d), j], side_name="lhs")
    split_rhs = AxisSide.from_spec(
        (ax[b, n, j], ax[b, d, j]),
        side_name="rhs",
    )
    split_plan = lower_one("rearrange", split_lhs, split_rhs)
    assert isinstance(split_plan.steps[0], AxisSliceSymbolicStep)

    concat_lhs = AxisSide.from_spec(
        (ax[b, n, j], ax[b, d, j]),
        side_name="lhs",
    )
    concat_rhs = AxisSide.from_spec(ax[b, (n + d), j], side_name="rhs")
    concat_plan = lower_one("rearrange", concat_lhs, concat_rhs)
    assert isinstance(concat_plan.steps[0], ConcatSymbolicStep)

    view_plan = lower_one("view", unary_lhs, unary_rhs)
    assert isinstance(view_plan.steps[0], PermuteSymbolicStep)

    view_additive_plan = lower_one(
        "view",
        AxisSide.from_spec(ax[b, (n + d)], side_name="lhs"),
        AxisSide.from_spec(ax[(n + d), b], side_name="rhs"),
    )
    assert isinstance(view_additive_plan.steps[0], PermuteSymbolicStep)

    a, c = axes("a", "c")
    view_reshape_plan = lower_one(
        "view",
        AxisSide.from_spec(ax[a, n, c], side_name="lhs"),
        AxisSide.from_spec(ax[(a * n), c], side_name="rhs"),
    )
    reshape_step = view_reshape_plan.steps[0]
    assert isinstance(reshape_step, ReshapeSymbolicStep)
    assert reshape_step.program.zero_copy_mode == "require_zero_copy"

    repeat_plan = lower_one("repeat", unary_lhs, unary_rhs)
    assert isinstance(repeat_plan.steps[0], PermuteSymbolicStep)

    repeat_expand_plan = lower_one(
        "repeat",
        AxisSide.from_spec(ax[b, n], side_name="lhs"),
        AxisSide.from_spec(ax[n, b, d], side_name="rhs"),
    )
    assert len(repeat_expand_plan.steps) == 2
    assert isinstance(repeat_expand_plan.steps[0], PermuteSymbolicStep)
    assert isinstance(repeat_expand_plan.steps[1], ExpandSymbolicStep)

    reduce_plan = lower_one("reduce", unary_lhs, unary_rhs)
    reduce_step = reduce_plan.steps[0]
    assert isinstance(reduce_step, ReduceSymbolicStep)
    assert reduce_step.reducer_label() == "sum(default)"

    einop_plan = lower_one("einop", unary_lhs, unary_rhs)
    assert len(einop_plan.steps) == 1
    assert isinstance(einop_plan.steps[0], PermuteSymbolicStep)

    einop_contract_plan = lower_one(
        "einop",
        AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs"),
        AxisSide.from_spec(ax[b, n, j], side_name="rhs"),
    )
    assert isinstance(einop_contract_plan.steps[0], EinsumSymbolicStep)


def test_symbolic_specialization_builds_fastpath_runtime_steps() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    axis_permute_step = AxisPermuteSymbolicStep(
        lhs=AxisSide.from_spec(ax[b, n, d], side_name="lhs"),
        rhs=AxisSide.from_spec(ax[b, d, n], side_name="rhs"),
        program=build_axis_permute_symbolic_program(
            lhs=AxisSide.from_spec(ax[b, n, d], side_name="lhs"),
            rhs=AxisSide.from_spec(ax[b, d, n], side_name="rhs"),
            explicit_sizes_items=(),
        ),
    )
    expand_step = ExpandSymbolicStep(
        lhs=AxisSide.from_spec(ax[b, n], side_name="lhs"),
        rhs=AxisSide.from_spec(ax[b, n, d], side_name="rhs"),
        program=build_expand_symbolic_program(ax[b, n], ax[b, n, d]),
    )
    unary_context = RuntimeSpecializationContext(
        input_shapes=((2, 3, 4),),
        backend_profile=None,
    )
    expand_context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )

    permute_runtime = axis_permute_step.specialize(unary_context)
    expand_runtime = expand_step.specialize(expand_context)
    assert isinstance(permute_runtime, PermuteRuntimeStep)
    assert isinstance(expand_runtime, ExpandRuntimeStep)
    assert expand_runtime.program is not None

    einsum_step = EinsumSymbolicStep(
        program=build_einsum_symbolic_program_from_equations(
            equations=("abc,cd->abd",),
            input_arity=2,
            output_arity=1,
        )
    )
    left = np.zeros((2, 3, 4), dtype=np.float32)
    right = np.zeros((4, 5), dtype=np.float32)
    einsum_context = RuntimeSpecializationContext(
        input_shapes=((2, 3, 4), (4, 5)),
        backend_profile=BACKEND_RESOLVER.resolve(left, right, op_name="contract"),
    )
    einsum_runtime = einsum_step.specialize(einsum_context)
    assert isinstance(einsum_runtime, EinsumRuntimeStep)

    lowering = DefaultLoweringProgram()
    einop_contract_ir = lowering.ir_program(
        op_name="einop",
        lhs=AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs"),
        rhs=AxisSide.from_spec(ax[b, n, j], side_name="rhs"),
        explicit_sizes_items=(),
    )
    einop_contract_plan = lowering.symbolic_candidates(
        ir_program=einop_contract_ir,
        explicit_sizes_items=(),
    )[0]
    contract_outputs = einop_contract_plan.execute(
        einsum_context,
        (left, right),
    )
    assert len(contract_outputs) == 1

    route_abstract_plan = AbstractPlan(
        op_name="rearrange",
        lhs=AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs"),
        rhs=AxisSide.from_spec((ax[d, j], ax[b, n, d]), side_name="rhs"),
        explicit_sizes_items=(),
        lowering=DefaultLoweringProgram(),
    )
    routed_outputs = route_abstract_plan.execute(
        RuntimeSpecializationContext(
            input_shapes=((2, 3, 4), (4, 5)),
            backend_profile=None,
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((4, 5), dtype=np.float32),
        ),
    )
    assert routed_outputs[0].shape == (4, 5)
    assert routed_outputs[1].shape == (2, 3, 4)


def test_symbolic_plan_execute_caches_step_specialization_per_runtime_key() -> None:
    _CountingSymbolicStep.calls.clear()
    step = _CountingSymbolicStep(
        name="counting",
        input_arity=1,
        output_arity=1,
        counter_key="cache-hit",
    )
    plan = SymbolicPlan(
        kind="counting",
        input_arity=1,
        output_arity=1,
        steps=(step,),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )
    tensors = (np.zeros((2, 3)),)

    _ = plan.execute(context, tensors)
    _ = plan.execute(context, tensors)
    assert _CountingSymbolicStep.calls["cache-hit"] == 1

    second_context = RuntimeSpecializationContext(
        input_shapes=((2, 4),),
        backend_profile=None,
    )
    second_tensors = (np.zeros((2, 4)),)
    _ = plan.execute(second_context, second_tensors)
    assert _CountingSymbolicStep.calls["cache-hit"] == 2


def test_select_symbolic_plan_uses_input_arity() -> None:
    one = SymbolicPlan(
        kind="one",
        input_arity=1,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="one",
                input_arity=1,
                output_arity=1,
            ),
        ),
    )
    two = SymbolicPlan(
        kind="two",
        input_arity=2,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="two",
                input_arity=2,
                output_arity=1,
            ),
        ),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3), (3, 4)),
        backend_profile=None,
    )

    lhs, rhs = _unary_side()
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=StaticLoweringProgram(candidates=(one, two)),
    )
    selected = abstract.select_symbolic_plan(
        PlanSelectionContext(input_shapes=context.input_shapes, explicit_sizes={})
    )
    assert selected.kind == "two"


def test_select_symbolic_plan_prefers_lower_score() -> None:
    higher = SymbolicPlan(
        kind="higher",
        input_arity=1,
        output_arity=1,
        steps=(
            _ScoredSymbolicStep(
                name="higher-step",
                input_arity=1,
                output_arity=1,
                step_score=SymbolicStepScore(
                    peak_einsum_numel=100,
                    materialize_numel=0,
                    allocation_count=1,
                    kernel_count=1,
                ),
            ),
        ),
    )
    lower = SymbolicPlan(
        kind="lower",
        input_arity=1,
        output_arity=1,
        steps=(
            _ScoredSymbolicStep(
                name="lower-step",
                input_arity=1,
                output_arity=1,
                step_score=SymbolicStepScore(
                    peak_einsum_numel=10,
                    materialize_numel=0,
                    allocation_count=1,
                    kernel_count=1,
                ),
            ),
        ),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )
    lhs, rhs = _unary_side()
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=StaticLoweringProgram(candidates=(higher, lower)),
    )
    selected = abstract.select_symbolic_plan(
        PlanSelectionContext(input_shapes=context.input_shapes, explicit_sizes={})
    )
    assert selected.kind == "lower"
    assert selected.score(
        PlanSelectionContext(input_shapes=context.input_shapes, explicit_sizes={})
    ) == SymbolicPlanScore(
        peak_einsum_numel=10,
        pre_einsum_materialize_numel=0,
        post_einsum_materialize_numel=0,
        allocation_count=1,
        kernel_count=1,
        step_count=1,
    )


def test_select_symbolic_plan_uses_selection_cache() -> None:
    _CountingScoreSymbolicStep.calls.clear()
    higher = SymbolicPlan(
        kind="higher",
        input_arity=1,
        output_arity=1,
        steps=(
            _CountingScoreSymbolicStep(
                name="higher-step",
                input_arity=1,
                output_arity=1,
                counter_key="higher",
                step_score=SymbolicStepScore(
                    peak_einsum_numel=100,
                    materialize_numel=10,
                    allocation_count=1,
                    kernel_count=1,
                ),
            ),
        ),
    )
    lower = SymbolicPlan(
        kind="lower",
        input_arity=1,
        output_arity=1,
        steps=(
            _CountingScoreSymbolicStep(
                name="lower-step",
                input_arity=1,
                output_arity=1,
                counter_key="lower",
                step_score=SymbolicStepScore(
                    peak_einsum_numel=10,
                    materialize_numel=10,
                    allocation_count=1,
                    kernel_count=1,
                ),
            ),
        ),
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )
    lhs, rhs = _unary_side()
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(("n", 3),),
        lowering=StaticLoweringProgram(candidates=(higher, lower)),
    )

    first = abstract.select_symbolic_plan(
        PlanSelectionContext(
            input_shapes=context.input_shapes,
            explicit_sizes={"n": 3},
        )
    )
    second = abstract.select_symbolic_plan(
        PlanSelectionContext(
            input_shapes=context.input_shapes,
            explicit_sizes={"n": 3},
        )
    )
    assert first.kind == "lower"
    assert second.kind == "lower"
    assert _CountingScoreSymbolicStep.calls == {"higher": 1, "lower": 1}


def test_abstract_plan_execute_runs_symbolic_then_runtime() -> None:
    lhs, rhs = _unary_side()
    symbolic = SymbolicPlan(
        kind="identity",
        input_arity=1,
        output_arity=1,
        steps=(
            _IdentitySymbolicStep(
                name="identity",
                input_arity=1,
                output_arity=1,
            ),
        ),
    )
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=StaticLoweringProgram(candidates=(symbolic,)),
    )
    tensor = np.arange(6).reshape(2, 3)
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3),),
        backend_profile=None,
    )

    outputs = abstract.execute(context, (tensor,))
    assert len(outputs) == 1
    np.testing.assert_array_equal(np.asarray(outputs[0]), tensor)


def test_abstract_plan_execute_rearrange_runs_real_runtime_step() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n, d], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, d, n], side_name="rhs")
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=DefaultLoweringProgram(),
    )
    tensor = np.arange(24).reshape(2, 3, 4)
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3, 4),),
        backend_profile=None,
    )

    outputs = abstract.execute(context, (tensor,))
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 3)
    np.testing.assert_array_equal(
        np.asarray(outputs[0]), np.transpose(tensor, (0, 2, 1))
    )


def test_abstract_plan_execute_einop_contract_runs_real_runtime_step() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    abstract = AbstractPlan(
        op_name="einop",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=DefaultLoweringProgram(),
    )
    left = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    right = np.arange(20).reshape(4, 5).astype(np.float32)
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3, 4), (4, 5)),
        backend_profile=None,
    )

    outputs = abstract.execute(context, (left, right))
    assert len(outputs) == 1
    expected = np.einsum("bnd,dj->bnj", left, right)
    np.testing.assert_allclose(np.asarray(outputs[0]), expected)
