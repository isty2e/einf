import pytest

from einf import ax, axes
from einf.axis import AxisSide
from einf.ir import IRProgram, LoweringSignature, LoweringTraceStage
from einf.lowering import DefaultLoweringProgram, StaticLoweringProgram
from einf.plans.abstract import AbstractPlan
from einf.plans.symbolic import SymbolicPlan
from einf.signature import Signature


def _source(
    op_name: str,
    lhs: AxisSide,
    rhs: AxisSide,
    *,
    explicit_sizes_items: tuple[tuple[str, int], ...] = (),
) -> LoweringSignature:
    return LoweringSignature(
        op_name=op_name,
        signature=Signature(inputs=lhs, outputs=rhs),
        explicit_sizes_items=explicit_sizes_items,
    )


def test_default_lowering_records_transform_trace_for_contract() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    lowering = DefaultLoweringProgram()

    source = _source("contract", lhs, rhs)
    ir_program = lowering.ir_program(source)

    assert isinstance(ir_program, IRProgram)
    assert ir_program.source == source
    assert ir_program.input_arity == 2
    assert ir_program.output_arity == 1
    assert ir_program.trace == (LoweringTraceStage.TRANSFORM,)


def test_default_lowering_records_composite_trace_for_einop() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    lowering = DefaultLoweringProgram()

    ir_program = lowering.ir_program(_source("einop", lhs, rhs))

    assert isinstance(ir_program, IRProgram)
    assert ir_program.trace == (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
        LoweringTraceStage.GATHER,
    )


def test_default_lowering_records_composite_trace_for_view() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n, d], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, d, n], side_name="rhs")
    lowering = DefaultLoweringProgram()

    ir_program = lowering.ir_program(_source("view", lhs, rhs))

    assert isinstance(ir_program, IRProgram)
    assert ir_program.trace == (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
        LoweringTraceStage.GATHER,
    )


def test_abstract_plan_keeps_ir_program_from_lowering() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n, d], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, d, n], side_name="rhs")
    lowering = DefaultLoweringProgram()
    source = _source("rearrange", lhs, rhs)
    abstract = AbstractPlan(
        source=source,
        lowering=lowering,
    )

    assert isinstance(abstract.ir_program, IRProgram)
    assert abstract.source == source
    assert abstract.ir_program.op_name == "rearrange"
    assert abstract.ir_program.input_arity == 1
    assert abstract.ir_program.output_arity == 1
    assert abstract.ir_program.trace == (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
        LoweringTraceStage.GATHER,
    )


def test_default_lowering_compiles_independently_of_observability_trace() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    lowering = DefaultLoweringProgram()
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    default_ir_program = lowering.ir_program(source)
    assert default_ir_program.trace == (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
    )
    alternate_trace_program = IRProgram(
        source=source,
        trace=(LoweringTraceStage.GATHER,),
    )

    default_candidates = lowering.symbolic_candidates(
        ir_program=default_ir_program,
    )
    alternate_trace_candidates = lowering.symbolic_candidates(
        ir_program=alternate_trace_program,
    )

    assert alternate_trace_candidates == default_candidates
    assert alternate_trace_candidates[0].source == source


def test_abstract_plan_rejects_ir_from_another_source() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    foreign_ir = IRProgram(
        source=_source(
            "repeat",
            lhs,
            rhs,
            explicit_sizes_items=(("d", 3),),
        )
    )

    with pytest.raises(ValueError, match="IR source"):
        AbstractPlan(
            source=source,
            lowering=StaticLoweringProgram(candidates=(), ir=foreign_ir),
        )


def test_abstract_plan_rejects_ir_from_another_operation() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    foreign_ir = IRProgram(
        source=_source(
            "einop",
            lhs,
            rhs,
            explicit_sizes_items=(("d", 2),),
        )
    )

    with pytest.raises(ValueError, match="IR source"):
        AbstractPlan(
            source=source,
            lowering=StaticLoweringProgram(candidates=(), ir=foreign_ir),
        )


def test_abstract_plan_rejects_ir_from_another_axis_signature() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    foreign_lhs = AxisSide.from_spec(ax[n, b], side_name="lhs")
    foreign_ir = IRProgram(
        source=_source(
            "repeat",
            foreign_lhs,
            rhs,
            explicit_sizes_items=(("d", 2),),
        )
    )

    with pytest.raises(ValueError, match="IR source"):
        AbstractPlan(
            source=source,
            lowering=StaticLoweringProgram(candidates=(), ir=foreign_ir),
        )


def test_abstract_plan_rejects_candidate_with_other_explicit_sizes() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    foreign_candidate = SymbolicPlan(
        source=_source(
            "repeat",
            lhs,
            rhs,
            explicit_sizes_items=(("d", 3),),
        ),
        kind="route",
        steps=(),
    )

    with pytest.raises(ValueError, match="symbolic plan source"):
        AbstractPlan(
            source=source,
            lowering=StaticLoweringProgram(candidates=(foreign_candidate,)),
        )


def test_abstract_plan_rejects_candidate_from_another_operation() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    source = _source(
        "repeat",
        lhs,
        rhs,
        explicit_sizes_items=(("d", 2),),
    )
    foreign_candidate = SymbolicPlan(
        source=_source(
            "einop",
            lhs,
            rhs,
            explicit_sizes_items=(("d", 2),),
        ),
        kind="route",
        steps=(),
    )

    with pytest.raises(ValueError, match="symbolic plan source"):
        AbstractPlan(
            source=source,
            lowering=StaticLoweringProgram(candidates=(foreign_candidate,)),
        )
