from inspect import signature

import numpy as np
import pytest

from einf import ax, axes
from einf.axis import AxisSide
from einf.backend import BACKEND_RESOLVER
from einf.steps.base import RuntimeSpecializationContext, SymbolicStepScore
from einf.steps.context import PlanSelectionContext
from einf.steps.einsum import (
    ChainEinsumSymbolicProgram,
    DirectEinsumSymbolicProgram,
    EinsumSymbolicProgram,
    EinsumSymbolicStep,
    SideEinsumSymbolicProgram,
    build_einsum_symbolic_program_from_equations,
    build_einsum_symbolic_program_from_sides,
)


def test_einsum_symbolic_program_base_is_abstract() -> None:
    required_methods = {
        "_preview_equations",
        "_resolve_equations",
        "_runtime_chain",
        "_specialization_depends_on_input_shapes",
    }

    assert required_methods <= EinsumSymbolicProgram.__abstractmethods__

    with pytest.raises(TypeError, match="abstract"):
        type.__call__(EinsumSymbolicProgram)


def test_einsum_symbolic_step_arity_has_one_authority() -> None:
    parameters = signature(EinsumSymbolicStep).parameters

    assert "input_arity" not in parameters
    assert "output_arity" not in parameters


def test_direct_einsum_program_owns_only_direct_equations() -> None:
    program = DirectEinsumSymbolicProgram(
        equations=("ab,bc->ac", "ab,bc->ac"),
        allow_native_matmul=True,
    )

    assert program.input_arity == 2
    assert program.output_arity == 2
    assert program.equations == ("ab,bc->ac", "ab,bc->ac")
    assert program.allow_native_matmul is True
    assert not hasattr(program, "chain_order")
    assert not hasattr(program, "carrier_index")
    assert not hasattr(program, "lhs")
    assert not hasattr(program, "rhs")
    assert not hasattr(program, "explicit_sizes_items")


def test_direct_einsum_program_rejects_incomplete_state() -> None:
    with pytest.raises(ValueError, match="requires equations"):
        DirectEinsumSymbolicProgram(equations=())


def test_direct_einsum_program_rejects_operand_arity_mismatch() -> None:
    with pytest.raises(ValueError, match="same input arity"):
        DirectEinsumSymbolicProgram(
            equations=("ab->ab", "ab,bc->ac"),
        )


def test_chain_einsum_program_owns_complete_chain() -> None:
    program = ChainEinsumSymbolicProgram(
        equations=("ab,bc->ac", "ac,cd->ad"),
        chain_order=(1, 2),
        carrier_index=0,
        allow_native_matmul=True,
    )

    assert program.input_arity == 3
    assert program.output_arity == 1
    assert program.equations == ("ab,bc->ac", "ac,cd->ad")
    assert program.chain_order == (1, 2)
    assert program.carrier_index == 0
    assert program.allow_native_matmul is True
    assert not hasattr(program, "lhs")
    assert not hasattr(program, "rhs")
    assert not hasattr(program, "explicit_sizes_items")


@pytest.mark.parametrize(
    ("equations", "chain_order", "carrier_index", "message"),
    [
        ((), (), 0, "requires equations"),
        (("ab,bc->ac",), (), 0, "requires chain order"),
        (("ab,bc->ac",), (1, 2), 0, "one equation per chain edge"),
        (("ab->ab", "ac,cd->ad"), (1, 2), 0, "binary edge"),
        (("ab,bc->ac", "ac,cd->ad"), (1, 2), -1, "out of bounds"),
        (("ab,bc->ac", "ac,cd->ad"), (1, 2), 3, "out of bounds"),
        (("ab,bc->ac", "ac,cd->ad"), (1, 1), 0, "non-carrier"),
        (("ab,bc->ac", "ac,cd->ad"), (0, 2), 0, "non-carrier"),
    ],
)
def test_chain_einsum_program_rejects_incomplete_state(
    equations: tuple[str, ...],
    chain_order: tuple[int, ...],
    carrier_index: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChainEinsumSymbolicProgram(
            equations=equations,
            chain_order=chain_order,
            carrier_index=carrier_index,
            allow_native_matmul=False,
        )


def test_side_einsum_program_owns_only_axis_sides() -> None:
    batch, contract, output = axes("batch", "contract", "output")
    lhs = AxisSide((ax[batch, contract], ax[contract, output]))
    rhs = AxisSide((ax[batch, output],))

    program = SideEinsumSymbolicProgram(
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=(("batch", 2),),
        allow_native_matmul=True,
    )

    assert program.input_arity == 2
    assert program.output_arity == 1
    assert program.lhs == lhs
    assert program.rhs == rhs
    assert program.explicit_sizes_items == (("batch", 2),)
    assert program.allow_native_matmul is True
    assert not hasattr(program, "equations")
    assert not hasattr(program, "chain_order")
    assert not hasattr(program, "carrier_index")


def test_side_einsum_program_rejects_multiple_outputs() -> None:
    batch, contract, output = axes("batch", "contract", "output")

    with pytest.raises(ValueError, match="must be N->1"):
        SideEinsumSymbolicProgram(
            lhs=AxisSide((ax[batch, contract], ax[contract, output])),
            rhs=AxisSide((ax[batch, output], ax[batch, output])),
            explicit_sizes_items=(),
            allow_native_matmul=False,
        )


def test_equation_builder_normalizes_direct_and_chain_variants() -> None:
    direct = build_einsum_symbolic_program_from_equations(
        input_arity=2,
        output_arity=1,
        equations=("ab,bc->ac",),
    )
    chain = build_einsum_symbolic_program_from_equations(
        input_arity=3,
        output_arity=1,
        equations=("ab,bc->ac", "ac,cd->ad"),
        chain_order=(1, 2),
        carrier_index=0,
    )

    assert isinstance(direct, DirectEinsumSymbolicProgram)
    assert isinstance(chain, ChainEinsumSymbolicProgram)


@pytest.mark.parametrize(
    ("input_arity", "output_arity", "message"),
    [
        (1, 1, "input arity mismatch"),
        (2, 2, "output arity mismatch"),
    ],
)
def test_equation_builder_rejects_declared_arity_mismatch(
    input_arity: int,
    output_arity: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_einsum_symbolic_program_from_equations(
            input_arity=input_arity,
            output_arity=output_arity,
            equations=("ab,bc->ac",),
        )


@pytest.mark.parametrize(
    ("input_arity", "output_arity", "message"),
    [
        (4, 1, "input arity mismatch"),
        (3, 2, "must be N->1"),
    ],
)
def test_equation_builder_rejects_chain_declared_arity_mismatch(
    input_arity: int,
    output_arity: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_einsum_symbolic_program_from_equations(
            input_arity=input_arity,
            output_arity=output_arity,
            equations=("ab,bc->ac", "ac,cd->ad"),
            chain_order=(1, 2),
            carrier_index=0,
        )


def test_side_builder_normalizes_side_variant() -> None:
    batch, contract, output = axes("batch", "contract", "output")

    program = build_einsum_symbolic_program_from_sides(
        lhs=AxisSide((ax[batch, contract], ax[contract, output])),
        rhs=AxisSide((ax[batch, output],)),
        explicit_sizes_items=(),
    )

    assert isinstance(program, SideEinsumSymbolicProgram)


def test_equation_builder_rejects_ambiguous_chain_metadata() -> None:
    with pytest.raises(ValueError, match="direct.*carrier"):
        build_einsum_symbolic_program_from_equations(
            input_arity=2,
            output_arity=1,
            equations=("ab,bc->ac",),
            carrier_index=0,
        )

    with pytest.raises(ValueError, match="chain.*carrier"):
        build_einsum_symbolic_program_from_equations(
            input_arity=2,
            output_arity=1,
            equations=("ab,bc->ac",),
            chain_order=(1,),
        )


def test_symbolic_step_projects_direct_and_chain_variants_to_runtime() -> None:
    direct_step = EinsumSymbolicStep(
        program=DirectEinsumSymbolicProgram(
            equations=("ab,bc->ac",),
        )
    )
    chain_step = EinsumSymbolicStep(
        program=ChainEinsumSymbolicProgram(
            equations=("ab,bc->ac", "ac,cd->ad"),
            chain_order=(1, 2),
            carrier_index=0,
        )
    )
    context = RuntimeSpecializationContext(
        input_shapes=((2, 3), (3, 4), (4, 5)),
        backend_profile=None,
    )

    direct_runtime = direct_step.specialize(context)
    chain_runtime = chain_step.specialize(context)

    assert direct_step.preview_equations() == ("ab,bc->ac",)
    assert direct_step.specialization_depends_on_input_shapes() is False
    assert direct_runtime.program.chain_order == ()
    assert direct_runtime.program.carrier_index is None
    assert chain_step.preview_equations() == ("ab,bc->ac", "ac,cd->ad")
    assert chain_step.specialization_depends_on_input_shapes() is False
    assert chain_runtime.program.chain_order == (1, 2)
    assert chain_runtime.program.carrier_index == 0


def test_nonzero_carrier_chain_preserves_execution_projection_and_score() -> None:
    first = np.arange(35, dtype=np.float64).reshape(5, 7)
    carrier = np.arange(6, dtype=np.float64).reshape(2, 3)
    second = np.arange(15, dtype=np.float64).reshape(3, 5)
    tensors = (first, carrier, second)
    input_shapes = tuple(tensor.shape for tensor in tensors)
    step = EinsumSymbolicStep(
        program=ChainEinsumSymbolicProgram(
            equations=("ab,bc->ac", "ac,cd->ad"),
            chain_order=(2, 0),
            carrier_index=1,
        )
    )
    runtime = step.specialize(
        RuntimeSpecializationContext(
            input_shapes=input_shapes,
            backend_profile=BACKEND_RESOLVER.resolve(
                *tensors,
                op_name="contract",
            ),
        )
    )

    assert runtime.program.chain_order == (2, 0)
    assert runtime.program.carrier_index == 1
    (actual,) = runtime.run(tensors)
    expected = carrier @ second @ first

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected)
    assert step.score(
        PlanSelectionContext(
            input_shapes=input_shapes,
            explicit_sizes={},
        )
    ) == SymbolicStepScore(
        peak_einsum_numel=35,
        materialize_numel=0,
        allocation_count=1,
        kernel_count=2,
    )


def test_symbolic_step_preserves_side_shape_dependency() -> None:
    batch, contract, output = axes("batch", "contract", "output")
    step = EinsumSymbolicStep(
        program=SideEinsumSymbolicProgram(
            lhs=AxisSide((ax[batch, contract], ax[contract, output])),
            rhs=AxisSide((ax[batch, output],)),
            explicit_sizes_items=(),
        )
    )

    runtime = step.specialize(
        RuntimeSpecializationContext(
            input_shapes=((2, 3), (3, 4)),
            backend_profile=None,
        )
    )

    assert step.preview_equations() == ("ab,bc->ac",)
    assert step.specialization_depends_on_input_shapes() is True
    assert runtime.program.equations == ("ab,bc->ac",)
    assert runtime.program.chain_order == ()
    assert runtime.program.carrier_index is None
