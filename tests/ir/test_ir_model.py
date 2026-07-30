from einf import ax, axes
from einf.axis import AxisSide
from einf.ir import IRProgram, LoweringTraceStage
from einf.lowering import DefaultLoweringProgram
from einf.plans.abstract import AbstractPlan


def test_default_lowering_records_transform_trace_for_contract() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    lowering = DefaultLoweringProgram()

    ir_program = lowering.ir_program(
        op_name="contract",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
    )

    assert isinstance(ir_program, IRProgram)
    assert ir_program.input_arity == 2
    assert ir_program.output_arity == 1
    assert ir_program.trace == (LoweringTraceStage.TRANSFORM,)


def test_default_lowering_records_composite_trace_for_einop() -> None:
    b, n, d, j = axes("b", "n", "d", "j")
    lhs = AxisSide.from_spec((ax[b, n, d], ax[d, j]), side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, j], side_name="rhs")
    lowering = DefaultLoweringProgram()

    ir_program = lowering.ir_program(
        op_name="einop",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
    )

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

    ir_program = lowering.ir_program(
        op_name="view",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
    )

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
    abstract = AbstractPlan(
        op_name="rearrange",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(),
        lowering=lowering,
    )

    assert isinstance(abstract.ir_program, IRProgram)
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
    default_ir_program = lowering.ir_program(
        op_name="repeat",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(("d", 2),),
    )
    assert default_ir_program.trace == (
        LoweringTraceStage.ASSEMBLE,
        LoweringTraceStage.TRANSFORM,
        LoweringTraceStage.ROUTE,
    )
    alternate_trace_program = IRProgram(
        op_name="repeat",
        lhs=lhs,
        rhs=rhs,
        trace=(LoweringTraceStage.GATHER,),
    )

    default_candidates = lowering.symbolic_candidates(
        ir_program=default_ir_program,
        explicit_sizes_items=(("d", 2),),
    )
    alternate_trace_candidates = lowering.symbolic_candidates(
        ir_program=alternate_trace_program,
        explicit_sizes_items=(("d", 2),),
    )

    assert alternate_trace_candidates == default_candidates
