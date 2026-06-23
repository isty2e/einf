from einf import ax, axes
from einf.axis import AxisSide
from einf.ir import AssembleIR, GatherIR, IRProgram, RouteIR, TransformIR
from einf.lowering import DefaultLoweringProgram
from einf.plans.abstract import AbstractPlan


def test_default_lowering_builds_transform_ir_for_contract() -> None:
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
    assert ir_program.node_kinds() == ("transform",)
    assert isinstance(ir_program.nodes[0], TransformIR)


def test_default_lowering_builds_composite_ir_for_einop() -> None:
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
    assert ir_program.node_kinds() == (
        "assemble",
        "transform",
        "route",
        "gather",
    )
    assert isinstance(ir_program.nodes[0], AssembleIR)
    assert isinstance(ir_program.nodes[1], TransformIR)
    assert isinstance(ir_program.nodes[2], RouteIR)
    assert isinstance(ir_program.nodes[3], GatherIR)


def test_default_lowering_builds_composite_ir_for_view() -> None:
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
    assert ir_program.node_kinds() == (
        "assemble",
        "transform",
        "route",
        "gather",
    )
    assert isinstance(ir_program.nodes[0], AssembleIR)
    assert isinstance(ir_program.nodes[1], TransformIR)
    assert isinstance(ir_program.nodes[2], RouteIR)
    assert isinstance(ir_program.nodes[3], GatherIR)


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
    assert abstract.ir_program.node_kinds() == (
        "assemble",
        "transform",
        "route",
        "gather",
    )


class _InvalidRepeatIRLowering(DefaultLoweringProgram):
    """Test lowering that injects invalid repeat IR shape."""

    def ir_program(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> IRProgram:
        _ = explicit_sizes_items
        if op_name != "repeat":
            return super().ir_program(
                op_name=op_name,
                lhs=lhs,
                rhs=rhs,
                explicit_sizes_items=(),
            )
        return IRProgram(
            op_name=op_name,
            lhs=lhs,
            rhs=rhs,
            nodes=(TransformIR(op_name=op_name, lhs=lhs, rhs=rhs),),
        )


def test_default_lowering_rejects_invalid_ir_shape() -> None:
    b, n, d = axes("b", "n", "d")
    lhs = AxisSide.from_spec(ax[b, n], side_name="lhs")
    rhs = AxisSide.from_spec(ax[b, n, d], side_name="rhs")
    lowering = _InvalidRepeatIRLowering()
    ir_program = lowering.ir_program(
        op_name="repeat",
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(("d", 2),),
    )

    try:
        _ = lowering.symbolic_candidates(
            ir_program=ir_program,
            explicit_sizes_items=(("d", 2),),
        )
    except ValueError as error:
        assert "missing required node kinds" in str(error)
    else:
        raise AssertionError("expected ValueError for invalid repeat IR shape")
