from collections.abc import Callable
from dataclasses import dataclass

from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
)
from einf.backend.runtime import is_trusted_backend_array_ops
from einf.diagnostics import ErrorCode, TensorOpError, ValidationError
from einf.signature import Signature
from einf.steps.base import RuntimeSpecializationContext, RuntimeStep
from einf.steps.runtime import (
    FALLBACK_ELIGIBLE_BACKEND_ERRORS,
    runtime_output_has_shape,
)
from einf.tensor_types import TensorLike

from ..base import AxisSideSymbolicStep
from ..runtime import (
    bind_runtime_backend,
    coerce_step_outputs,
)
from .model import ExpandSymbolicProgram
from .runtime import (
    compile_expand_target_shape_evaluator,
    expand_execution_error,
    run_expand_program,
)
from .solve import solve_expand_program_from_input_shape


@dataclass(frozen=True, slots=True)
class ExpandRuntimeStep(RuntimeStep[ExpandSymbolicProgram]):
    """Runtime expand step."""

    name: str
    input_arity: int
    output_arity: int
    program: ExpandSymbolicProgram
    target_shape_evaluator: (
        Callable[[tuple[int, ...]], tuple[int, ...] | None] | None
    ) = None
    runtime_backend_ops: BackendArrayOps | None = None
    runtime_xp: ArrayNamespace | None = None
    compiled_unary_runner: Callable[[TensorLike], TensorLike | None] | None = None

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute unary expand step and return one tensor."""
        try:
            compiled_unary_runner = self.compiled_unary_runner
            if compiled_unary_runner is not None:
                try:
                    compiled_output = compiled_unary_runner(tensor)
                except TensorOpError:
                    raise
                except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
                    compiled_output = None
                if compiled_output is not None:
                    return compiled_output

            target_shape_evaluator = self.target_shape_evaluator
            if target_shape_evaluator is None:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: expand lowering invariant violated "
                        "(missing target-shape evaluator)"
                    ),
                    help="ensure lowering emits one resolvable unary expand program",
                    related=("expand lowering",),
                    data={"operation": "expand"},
                )
            target_shape = target_shape_evaluator(tensor.shape)
            if target_shape is None:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: expand lowering invariant violated "
                        "(target shape is unresolved at runtime)"
                    ),
                    help="bind required dimensions via with_sizes or one resolvable lhs shape",
                    related=("expand lowering",),
                    data={"operation": "expand"},
                )
            return run_expand_program(
                plan=self.program,
                tensor=tensor,
                target_shape=target_shape,
                backend_ops=self.runtime_backend_ops,
                xp=self.runtime_xp,
            )
        except TensorOpError:
            raise
        except Exception as error:
            raise expand_execution_error(error) from error

    def run(self, tensors: tuple[TensorLike, ...], /) -> tuple[TensorLike, ...]:
        """Execute runtime expand step."""
        if len(tensors) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=(
                    f"expand arity mismatch: expected 1 input, got {len(tensors)}"
                ),
                help="expand runtime step requires one tensor",
                related=("expand runtime",),
                data={"operation": "expand"},
            )
        return coerce_step_outputs(self.run_unary(tensors[0]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpandSymbolicStep(AxisSideSymbolicStep[ExpandSymbolicProgram]):
    """Symbolic expand step."""

    program: ExpandSymbolicProgram
    name: str = "expand"

    def __post_init__(self) -> None:
        AxisSideSymbolicStep.__post_init__(self)
        if self.input_arity != 1 or self.output_arity != 1:
            raise ValueError("expand symbolic step must be 1->1")
        if self.program.lhs_terms != self.lhs[0]:
            raise ValueError("expand symbolic step program lhs does not match step lhs")
        if self.program.rhs_terms != self.rhs[0]:
            raise ValueError("expand symbolic step program rhs does not match step rhs")

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        signature = Signature(inputs=self.lhs, outputs=self.rhs)
        explicit_sizes = signature.filter_explicit_sizes(
            dict(self.explicit_sizes_items)
        )
        program = self.program
        if program.compiled is None:
            input_shapes = context.input_shapes
            if len(input_shapes) != 1:
                raise ValidationError(
                    code=ErrorCode.OP_ARITY_MISMATCH,
                    message=(
                        "expand arity mismatch during specialization: "
                        f"expected one input shape, got {len(input_shapes)}"
                    ),
                    help="ensure expand specialization receives one unary input shape",
                    related=("expand specialization",),
                    data={"operation": "expand"},
                )
            solved = solve_expand_program_from_input_shape(
                input_shape=input_shapes[0],
                lhs_terms=self.program.lhs_terms,
                rhs_terms=self.program.rhs_terms,
                explicit_sizes=explicit_sizes,
            )
            if solved is None:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: expand lowering invariant violated "
                        "(could not specialize unresolved expand program)"
                    ),
                    help="bind required dimensions via with_sizes to make expand executable",
                    related=("expand lowering",),
                    data={"operation": "expand"},
                )
            solved_program, _ = solved
            program = solved_program

        if program.compiled is None:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: expand lowering invariant violated "
                    "(compiled expand program is missing)"
                ),
                help="ensure lowering emits one compiled unary expand program",
                related=("expand lowering",),
                data={"operation": "expand"},
            )

        program_axis_names = program.compiled.axis_names
        program_explicit_sizes = {
            axis_name: axis_size
            for axis_name, axis_size in explicit_sizes.items()
            if axis_name in program_axis_names
        }
        target_shape_evaluator = compile_expand_target_shape_evaluator(
            plan=program,
            explicit_sizes=program_explicit_sizes,
        )
        backend_binding = bind_runtime_backend(
            context,
            operation="expand",
            required_namespace_methods=("permute_dims", "expand_dims", "broadcast_to"),
            bind_namespace_when_backend_ops_available=False,
        )
        compiled_unary_runner: Callable[[TensorLike], TensorLike | None] | None = None
        compiled_program = program.compiled
        if compiled_program is not None and target_shape_evaluator is not None:
            backend_ops = backend_binding.backend_ops
            if backend_ops is not None:
                permutation = compiled_program.permutation
                has_non_identity_permutation = (
                    compiled_program.has_non_identity_permutation
                )
                insert_axes = compiled_program.insert_axes
                permute_fn = backend_ops.permute
                expand_dims_fn = backend_ops.expand_dims
                broadcast_to_fn = backend_ops.broadcast_to

                def run_compiled_with_backend_ops(
                    tensor: TensorLike,
                ) -> TensorLike | None:
                    target_shape = target_shape_evaluator(tensor.shape)
                    if target_shape is None:
                        return None
                    try:
                        trusted_route = is_trusted_backend_array_ops(
                            backend_ops=backend_ops,
                            tensor=tensor,
                        )
                    except TensorOpError:
                        raise
                    except Exception as error:
                        raise expand_execution_error(error) from error
                    if not trusted_route:
                        return run_expand_program(
                            plan=program,
                            tensor=tensor,
                            target_shape=target_shape,
                            backend_ops=backend_ops,
                            xp=None,
                        )
                    transformed = tensor
                    if has_non_identity_permutation:
                        transformed = permute_fn(transformed, permutation)
                    for output_index in insert_axes:
                        transformed = expand_dims_fn(transformed, output_index)
                    output = broadcast_to_fn(transformed, target_shape)
                    if runtime_output_has_shape(
                        output,
                        target_shape,
                        operation="expand",
                    ):
                        return output
                    return None

                compiled_unary_runner = run_compiled_with_backend_ops
            elif backend_binding.xp is not None:
                xp = backend_binding.xp

                def run_compiled_with_namespace(
                    tensor: TensorLike,
                ) -> TensorLike | None:
                    target_shape = target_shape_evaluator(tensor.shape)
                    if target_shape is None:
                        return None
                    return run_expand_program(
                        plan=program,
                        tensor=tensor,
                        target_shape=target_shape,
                        backend_ops=None,
                        xp=xp,
                    )

                compiled_unary_runner = run_compiled_with_namespace
        return ExpandRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=program,
            target_shape_evaluator=target_shape_evaluator,
            runtime_backend_ops=backend_binding.backend_ops,
            runtime_xp=backend_binding.xp,
            compiled_unary_runner=compiled_unary_runner,
        )

    def specialization_depends_on_input_shapes(self) -> bool:
        """Return whether expand specialization depends on runtime input shapes."""
        return self.program.compiled is None


__all__ = [
    "ExpandRuntimeStep",
    "ExpandSymbolicStep",
]
