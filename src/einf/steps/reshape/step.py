from collections.abc import Callable
from dataclasses import dataclass

from einf.axis import term_size
from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    BackendProfile,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import PlanSelectionContext, build_runtime_execution_context
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.tensor_types import TensorLike

from ..runtime import bind_runtime_backend
from .compile import compile_reshape_target_shape_evaluator
from .constants import (
    RESHAPE_REQUIRED_NAMESPACE_METHODS,
    ZERO_COPY_ALLOWED_RESHAPE_MODE,
    ZERO_COPY_REQUIRED_RESHAPE_MODE,
    ZeroCopyReshapeMode,
)
from .model import ReshapeSymbolicProgram
from .runtime import (
    run_reshape_program,
    try_run_reshape_program,
    validate_rearrange_numel,
)
from .zero_copy import (
    normalize_zero_copy_reshape_error,
    reshape_shares_memory,
)


@dataclass(frozen=True, slots=True)
class ReshapeRuntimeStep(RuntimeStep[ReshapeSymbolicProgram]):
    """Runtime reshape step."""

    name: str
    input_arity: int
    output_arity: int
    program: ReshapeSymbolicProgram
    explicit_sizes: dict[str, int]
    target_shape_evaluator: (
        Callable[[tuple[int, ...]], tuple[int, ...] | None] | None
    ) = None
    runtime_backend_ops: BackendArrayOps | None = None
    runtime_xp: ArrayNamespace | None = None
    backend_profile: BackendProfile | None = None
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE
    compiled_unary_runner: Callable[[TensorLike], TensorLike | None] | None = None

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=f"reshape arity mismatch: expected 1 input, got {len(tensors)}",
                help="reshape runtime step requires one tensor",
                related=("reshape runtime",),
                data={"operation": "reshape"},
            )
        return (self.run_unary(tensors[0]),)

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute one unary reshape runtime step."""
        compiled_unary_runner = self.compiled_unary_runner
        if compiled_unary_runner is not None:
            try:
                compiled_output = compiled_unary_runner(tensor)
            except (TypeError, ValueError, RuntimeError):
                compiled_output = None
            if compiled_output is not None:
                return compiled_output

        if self.program.reject_not_a_view:
            raise ValidationError(
                code=ErrorCode.NOT_A_VIEW,
                message=(
                    "not a view: reshape mapping includes view-disallowed "
                    "unit-axis insertion"
                ),
                help=(
                    "remove explicit singleton-axis insertion from view mapping "
                    "or use repeat instead"
                ),
                related=("view affine mapping", "reshape"),
                data={"operation": "view"},
            )
        if self.zero_copy_mode not in {
            ZERO_COPY_ALLOWED_RESHAPE_MODE,
            ZERO_COPY_REQUIRED_RESHAPE_MODE,
        }:
            raise ValueError("reshape runtime step requires a valid zero_copy_mode")

        if self.zero_copy_mode == ZERO_COPY_REQUIRED_RESHAPE_MODE:
            return self._run_unary_zero_copy(tensor)
        return self._run_unary_allow_copy(tensor)

    def _run_unary_allow_copy(self, tensor: TensorLike, /) -> TensorLike:
        """Execute reshape runtime without zero-copy restriction."""
        compiled_program = self.program.compiled
        if compiled_program is None:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: reshape lowering invariant violated "
                    "(compiled reshape program is missing)"
                ),
                help="ensure lowering emits reshape only for direct unary shape transforms",
                related=("reshape lowering",),
                data={"operation": "reshape"},
            )

        target_shape = None
        target_shape_evaluator = self.target_shape_evaluator
        if target_shape_evaluator is not None:
            target_shape = target_shape_evaluator(tensor.shape)

        if target_shape is not None:
            try:
                return run_reshape_program(
                    tensor=tensor,
                    target_shape=target_shape,
                    backend_ops=self.runtime_backend_ops,
                    xp=self.runtime_xp,
                    zero_copy_mode=self.zero_copy_mode,
                )
            except (TypeError, ValueError, RuntimeError):
                pass

        symbolic_output = try_run_reshape_program(
            tensor=tensor,
            explicit_sizes=self.explicit_sizes,
            program=compiled_program,
            backend_ops=self.runtime_backend_ops,
            xp=self.runtime_xp,
            zero_copy_mode=self.zero_copy_mode,
        )
        if symbolic_output is not None:
            return symbolic_output

        runtime_context = build_runtime_execution_context(
            signature=self.program.signature,
            tensors=(tensor,),
            explicit_sizes=self.explicit_sizes,
        )
        target_shape = tuple(
            term_size(term, runtime_context.axis_sizes)
            for term in runtime_context.rhs_terms[0]
        )
        validate_rearrange_numel(
            input_shape=tensor.shape,
            target_shape=target_shape,
        )
        return run_reshape_program(
            tensor=tensor,
            target_shape=target_shape,
            backend_ops=self.runtime_backend_ops,
            xp=self.runtime_xp,
            zero_copy_mode=self.zero_copy_mode,
        )

    def _run_unary_zero_copy(self, tensor: TensorLike, /) -> TensorLike:
        """Execute reshape runtime and require output aliasing to input."""
        try:
            output = self._run_unary_allow_copy(tensor)
        except ValidationError as error:
            normalized = normalize_zero_copy_reshape_error(
                error,
                operation="view",
                backend=(
                    None
                    if self.backend_profile is None
                    else self.backend_profile.namespace_id
                ),
            )
            if normalized is error:
                raise
            raise normalized from error

        backend_profile = self.backend_profile
        if backend_profile is None:
            raise ValidationError(
                code=ErrorCode.NOT_A_VIEW,
                message=(
                    "not a view: reshape zero-copy check could not resolve "
                    "backend profile"
                ),
                help="use a strict view-capable backend (numpy or torch)",
                related=("view affine mapping", "backend capability"),
                data={"operation": "view"},
            )

        shares_memory = reshape_shares_memory(
            lhs=tensor,
            rhs=output,
            backend_profile=backend_profile,
        )
        if shares_memory is not True:
            raise ValidationError(
                code=ErrorCode.NOT_A_VIEW,
                message=(
                    "not a view: reshape output does not alias input storage "
                    "as a zero-copy view"
                ),
                help=(
                    "restrict view reshape to runtime layouts that preserve "
                    "zero-copy aliasing"
                ),
                related=("view affine mapping", "zero-copy check"),
                data={
                    "operation": "view",
                    "backend": backend_profile.namespace_id,
                },
            )
        return output


@dataclass(frozen=True, slots=True)
class ReshapeSymbolicStep(SymbolicStep[ReshapeSymbolicProgram]):
    """Primitive reshape symbolic step (1->1)."""

    program: ReshapeSymbolicProgram
    explicit_sizes_items: tuple[tuple[str, int], ...] = ()
    name: str = "reshape"
    input_arity: int = 1
    output_arity: int = 1

    def __post_init__(self) -> None:
        signature = self.program.signature
        if signature.input_arity != 1 or signature.output_arity != 1:
            raise ValueError("reshape symbolic step must be 1->1")
        compiled = self.program.compiled
        if compiled is None:
            raise ValueError("reshape symbolic step requires a compiled program")
        if self.program.zero_copy_mode not in {
            ZERO_COPY_ALLOWED_RESHAPE_MODE,
            ZERO_COPY_REQUIRED_RESHAPE_MODE,
        }:
            raise ValueError("reshape symbolic step requires a valid zero_copy_mode")

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        signature = self.program.signature
        explicit_sizes = signature.filter_explicit_sizes(
            dict(self.explicit_sizes_items)
        )
        compiled = self.program.compiled
        if compiled is None:
            raise ValueError("reshape symbolic step requires a compiled program")

        backend_profile = context.backend_profile
        backend_binding = bind_runtime_backend(
            context,
            required_namespace_methods=RESHAPE_REQUIRED_NAMESPACE_METHODS,
            bind_namespace_when_backend_ops_available=True,
        )

        target_shape_evaluator = compile_reshape_target_shape_evaluator(
            plan=compiled,
            explicit_sizes=explicit_sizes,
        )
        compiled_unary_runner: Callable[[TensorLike], TensorLike | None] | None = None
        if (
            self.program.zero_copy_mode == ZERO_COPY_ALLOWED_RESHAPE_MODE
            and target_shape_evaluator is not None
        ):
            compiled_rank_may_stay_equal = len(compiled.lhs_terms) == len(
                compiled.rhs_terms
            )
            backend_ops = backend_binding.backend_ops
            if backend_ops is not None:
                reshape_fn = backend_ops.reshape

                def run_compiled_with_backend_ops(
                    tensor: TensorLike,
                ) -> TensorLike | None:
                    target_shape = target_shape_evaluator(tensor.shape)
                    if target_shape is None:
                        return None
                    if compiled_rank_may_stay_equal and tensor.shape == target_shape:
                        return tensor
                    return reshape_fn(tensor, target_shape)

                compiled_unary_runner = run_compiled_with_backend_ops
            elif backend_binding.xp is not None:
                xp_reshape = backend_binding.xp.reshape

                def run_compiled_with_namespace(
                    tensor: TensorLike,
                ) -> TensorLike | None:
                    target_shape = target_shape_evaluator(tensor.shape)
                    if target_shape is None:
                        return None
                    if compiled_rank_may_stay_equal and tensor.shape == target_shape:
                        return tensor
                    return xp_reshape(tensor, target_shape)

                compiled_unary_runner = run_compiled_with_namespace
        return ReshapeRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=self.program,
            explicit_sizes=explicit_sizes,
            target_shape_evaluator=target_shape_evaluator,
            runtime_backend_ops=backend_binding.backend_ops,
            runtime_xp=backend_binding.xp,
            backend_profile=backend_profile,
            zero_copy_mode=self.program.zero_copy_mode,
            compiled_unary_runner=compiled_unary_runner,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=0,
            allocation_count=0,
            kernel_count=1,
        )


__all__ = [
    "ReshapeRuntimeStep",
    "ReshapeSymbolicStep",
]
