import numpy as np

from einf.axis import AxisTerms
from einf.backend import get_backend_array_ops
from einf.plans.fusion import (
    discover_tuple_step_fusion,
    discover_tuple_step_fusions,
)
from einf.steps.expand import (
    ExpandRuntimeStep,
    build_expand_symbolic_program,
    compile_expand_target_shape_evaluator,
)
from einf.steps.permute import (
    PermuteRuntimeStep,
    build_permute_symbolic_program,
)
from einf.steps.reshape import (
    ReshapeRuntimeStep,
    build_reshape_symbolic_program,
)
from einf.steps.reshape.compile import compile_reshape_target_shape_evaluator
from einf.steps.reshape.constants import ZERO_COPY_REQUIRED_RESHAPE_MODE


def test_discover_fuses_permute_then_permute() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    first_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 2, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    second_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0, 2)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )

    fusion = discover_tuple_step_fusion((first_step, second_step))
    assert fusion is not None
    assert fusion.name == "permute_permute"

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    fused_output = fusion.runner((tensor,))
    sequential_output = (second_step.run_unary(first_step.run_unary(tensor)),)
    expected = (np.transpose(tensor, (2, 1, 0)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], expected[0])
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_discover_fuses_permute_then_expand() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    permute_step = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    expand_program = build_expand_symbolic_program(
        AxisTerms.from_spec((2, 3)),
        AxisTerms.from_spec((3, 2, 1)),
    )
    target_shape_evaluator = compile_expand_target_shape_evaluator(
        plan=expand_program,
        explicit_sizes={},
    )
    expand_step = ExpandRuntimeStep(
        name="expand",
        input_arity=1,
        output_arity=1,
        program=expand_program,
        target_shape_evaluator=target_shape_evaluator,
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )

    fusion = discover_tuple_step_fusion((permute_step, expand_step))
    assert fusion is not None
    assert fusion.name == "permute_expand"

    tensor = np.arange(6).reshape(3, 2)
    fused_output = fusion.runner((tensor,))
    sequential_output = (expand_step.run_unary(permute_step.run_unary(tensor)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_discover_fuses_reshape_then_reshape() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
    )
    first_compiled = first_program.compiled
    assert first_compiled is not None
    first_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=first_program.zero_copy_mode,
    )

    second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    second_compiled = second_program.compiled
    assert second_compiled is not None
    second_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=second_program.zero_copy_mode,
    )

    fusion = discover_tuple_step_fusion((first_step, second_step))
    assert fusion is not None
    assert fusion.name == "reshape_reshape"

    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    fused_output = fusion.runner((tensor,))
    sequential_output = (second_step.run_unary(first_step.run_unary(tensor)),)
    expected = (np.reshape(tensor, (2, 12)),)

    assert isinstance(fused_output[0], np.ndarray)
    assert isinstance(sequential_output[0], np.ndarray)
    np.testing.assert_array_equal(fused_output[0], expected[0])
    np.testing.assert_array_equal(fused_output[0], sequential_output[0])


def test_discover_does_not_fuse_reshape_chain_when_zero_copy_required() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
        zero_copy_mode=ZERO_COPY_REQUIRED_RESHAPE_MODE,
    )
    first_compiled = first_program.compiled
    assert first_compiled is not None
    first_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=first_program.zero_copy_mode,
    )

    second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    second_compiled = second_program.compiled
    assert second_compiled is not None
    second_step = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=second_program.zero_copy_mode,
    )

    fusion = discover_tuple_step_fusion((first_step, second_step))
    assert fusion is None


def test_discover_finds_multiple_non_overlapping_fusions() -> None:
    backend_ops = get_backend_array_ops("numpy")
    assert backend_ops is not None

    permute_first = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 2, 0)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )
    permute_second = PermuteRuntimeStep(
        name="permute",
        input_arity=1,
        output_arity=1,
        program=build_permute_symbolic_program((1, 0, 2)),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
    )

    reshape_first_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((2, 3, 4)),
        AxisTerms.from_spec((6, 4)),
    )
    reshape_first_compiled = reshape_first_program.compiled
    assert reshape_first_compiled is not None
    reshape_first = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=reshape_first_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=reshape_first_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=reshape_first_program.zero_copy_mode,
    )

    reshape_second_program = build_reshape_symbolic_program(
        AxisTerms.from_spec((6, 4)),
        AxisTerms.from_spec((2, 12)),
    )
    reshape_second_compiled = reshape_second_program.compiled
    assert reshape_second_compiled is not None
    reshape_second = ReshapeRuntimeStep(
        name="reshape",
        input_arity=1,
        output_arity=1,
        program=reshape_second_program,
        explicit_sizes={},
        target_shape_evaluator=compile_reshape_target_shape_evaluator(
            plan=reshape_second_compiled,
            explicit_sizes={},
        ),
        runtime_backend_ops=backend_ops,
        runtime_xp=None,
        zero_copy_mode=reshape_second_program.zero_copy_mode,
    )

    fusions = discover_tuple_step_fusions(
        (
            permute_first,
            permute_second,
            reshape_first,
            reshape_second,
        )
    )
    assert len(fusions) == 2
    assert (fusions[0].name, fusions[0].start, fusions[0].stop) == (
        "permute_permute",
        0,
        2,
    )
    assert (fusions[1].name, fusions[1].start, fusions[1].stop) == (
        "reshape_reshape",
        2,
        4,
    )
