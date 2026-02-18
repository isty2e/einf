from dataclasses import dataclass

from einf.axis import AxisSide, expand_products_for_terms
from einf.axis.matching import collect_axis_mappings, scalar_terms_to_atomic_tokens
from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import PlanSelectionContext, build_runtime_execution_context
from einf.signature import Signature
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.tensor_types import TensorLike

from .base import AxisSideSymbolicStep
from .runtime import bind_runtime_backend

_PERMUTE_REQUIRED_METHODS = ("permute_dims",)


@dataclass(frozen=True, slots=True)
class PermuteSymbolicProgram(SymbolicProgram):
    """Precompiled unary permute program."""

    permutation: tuple[int, ...]
    has_non_identity_permutation: bool


@dataclass(frozen=True, slots=True)
class AxisPermuteSymbolicProgram(SymbolicProgram):
    """Precompiled axis-driven unary permute program."""

    lhs: AxisSide
    rhs: AxisSide
    explicit_sizes_items: tuple[tuple[str, int], ...]


def build_permute_symbolic_program(
    permutation: tuple[int, ...],
) -> PermuteSymbolicProgram:
    """Build one validated unary permute program."""
    rank = len(permutation)
    if tuple(sorted(permutation)) != tuple(range(rank)):
        raise ValueError("permute symbolic step requires a valid permutation tuple")
    return PermuteSymbolicProgram(
        permutation=permutation,
        has_non_identity_permutation=any(
            source_axis != target_axis
            for target_axis, source_axis in enumerate(permutation)
        ),
    )


def build_axis_permute_symbolic_program(
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
) -> AxisPermuteSymbolicProgram:
    """Build one precompiled axis-permute program from canonical sides."""
    if len(lhs) != 1 or len(rhs) != 1:
        raise ValueError("axis permute symbolic step must be 1->1")
    return AxisPermuteSymbolicProgram(
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=explicit_sizes_items,
    )


@dataclass(frozen=True, slots=True)
class PermuteRuntimeStep(RuntimeStep[PermuteSymbolicProgram]):
    """Runtime permute step (1->1)."""

    name: str
    input_arity: int
    output_arity: int
    program: PermuteSymbolicProgram
    runtime_backend_ops: BackendArrayOps | None = None
    runtime_xp: ArrayNamespace | None = None

    def run_unary(self, tensor: TensorLike, /) -> TensorLike:
        """Execute unary permute step and return one tensor."""
        if not self.program.has_non_identity_permutation:
            return tensor
        permutation = self.program.permutation
        runtime_backend_ops = self.runtime_backend_ops
        if runtime_backend_ops is not None:
            return runtime_backend_ops.permute(tensor, permutation)

        xp = self.runtime_xp
        if xp is None:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message=(
                    "backend dispatch unsupported input: "
                    "permute runtime requires one resolved backend namespace/profile"
                ),
                help="execute through AbstractPlan/TensorOp call path to resolve backend profile",
                related=("backend dispatch",),
                data={"operation": "permute"},
            )
        try:
            return xp.permute_dims(tensor, permutation)
        except Exception as error:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: rearrange backend primitive failed during "
                    f"reindex execution: {error}"
                ),
                help=(
                    "ensure shape mapping is valid and backend primitives support "
                    "the required operation on the given tensor layout"
                ),
                related=("backend execution",),
                data={"operation": "rearrange"},
            ) from error

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        if len(tensors) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message=(
                    f"permute arity mismatch: expected 1 input, got {len(tensors)}"
                ),
                help="permute runtime step requires one tensor",
                related=("permute runtime",),
                data={"operation": "permute"},
            )
        return (self.run_unary(tensors[0]),)


@dataclass(frozen=True, slots=True)
class PermuteSymbolicStep(SymbolicStep[PermuteSymbolicProgram]):
    """Primitive permute symbolic step (1->1)."""

    program: PermuteSymbolicProgram
    name: str = "permute"
    input_arity: int = 1
    output_arity: int = 1

    def __post_init__(self) -> None:
        if self.input_arity != 1 or self.output_arity != 1:
            raise ValueError("permute symbolic step must be 1->1")
        _ = build_permute_symbolic_program(self.program.permutation)

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        backend_binding = bind_runtime_backend(
            context,
            required_namespace_methods=_PERMUTE_REQUIRED_METHODS,
            bind_namespace_when_backend_ops_available=False,
        )
        return PermuteRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=self.program,
            runtime_backend_ops=backend_binding.backend_ops,
            runtime_xp=backend_binding.xp,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        _ = context
        return SymbolicStepScore(
            peak_einsum_numel=0,
            materialize_numel=0,
            allocation_count=0,
            kernel_count=0 if not self.program.has_non_identity_permutation else 1,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AxisPermuteSymbolicStep(AxisSideSymbolicStep[AxisPermuteSymbolicProgram]):
    """Symbolic permute step resolved from canonical axis sides at specialize time."""

    program: AxisPermuteSymbolicProgram
    name: str = "permute"

    def __post_init__(self) -> None:
        AxisSideSymbolicStep.__post_init__(self)
        if self.input_arity != 1 or self.output_arity != 1:
            raise ValueError("axis permute symbolic step must be 1->1")
        if self.program.lhs != self.lhs:
            raise ValueError("axis permute program lhs does not match step lhs")
        if self.program.rhs != self.rhs:
            raise ValueError("axis permute program rhs does not match step rhs")
        if self.program.explicit_sizes_items != self.explicit_sizes_items:
            raise ValueError(
                "axis permute program explicit sizes do not match step explicit sizes"
            )

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        signature = Signature(inputs=self.program.lhs, outputs=self.program.rhs)
        signature_explicit_sizes = signature.filter_explicit_sizes(
            dict(self.program.explicit_sizes_items)
        )
        normalized = build_runtime_execution_context(
            signature=signature,
            tensors=(),
            explicit_sizes=signature_explicit_sizes,
            input_shapes=context.input_shapes,
        )
        lhs_terms = expand_products_for_terms(normalized.lhs_terms[0])
        rhs_terms = expand_products_for_terms(normalized.rhs_terms[0])
        lhs_tokens = scalar_terms_to_atomic_tokens(lhs_terms, normalized.axis_sizes)
        rhs_tokens = scalar_terms_to_atomic_tokens(rhs_terms, normalized.axis_sizes)

        mappings = collect_axis_mappings(
            lhs_tokens,
            rhs_tokens,
            allow_broadcast=False,
        )
        if not mappings:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: unary permute lowering could not find a "
                    "valid axis mapping"
                ),
                help="use unary signatures for axis permutation lowering",
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        if len(mappings) > 1:
            raise ValidationError(
                code=ErrorCode.AMBIGUOUS_DIMS,
                message=(
                    "ambiguous dims: unary permute lowering has multiple valid "
                    "axis mappings"
                ),
                help=(
                    "add axis names or with_sizes constraints to make the permutation "
                    "mapping unique"
                ),
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        mapping_candidate: tuple[int | None, ...] | None = None
        for candidate in mappings:
            mapping_candidate = candidate
            break
        if mapping_candidate is None:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: unary permute lowering could not resolve "
                    "a unique axis mapping"
                ),
                help="use unary signatures for axis permutation lowering",
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        mapping = mapping_candidate
        if any(source is None for source in mapping):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: unary permute lowering introduced broadcast axes"
                ),
                help="use repeat/broadcast for synthetic-axis introduction",
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        permutation = tuple(source for source in mapping if source is not None)
        rank = len(lhs_tokens)
        if len(permutation) != rank:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: unary permute lowering changed rank",
                help="restrict unary permute lowering to rank-preserving mappings",
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        if tuple(sorted(permutation)) != tuple(range(rank)):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: unary permute lowering produced non-bijective mapping",
                help="restrict unary permute lowering to one-to-one axis permutations",
                related=("permute lowering",),
                data={"operation": "rearrange"},
            )
        backend_binding = bind_runtime_backend(
            context,
            required_namespace_methods=_PERMUTE_REQUIRED_METHODS,
            bind_namespace_when_backend_ops_available=False,
        )
        return PermuteRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=build_permute_symbolic_program(permutation),
            runtime_backend_ops=backend_binding.backend_ops,
            runtime_xp=backend_binding.xp,
        )

    def specialization_depends_on_input_shapes(self) -> bool:
        """Return whether specialization depends on runtime input shapes."""
        return True


__all__ = [
    "AxisPermuteSymbolicStep",
    "AxisPermuteSymbolicProgram",
    "build_axis_permute_symbolic_program",
    "build_permute_symbolic_program",
    "PermuteSymbolicProgram",
    "PermuteRuntimeStep",
    "PermuteSymbolicStep",
]
