from einf.axis import AxisSide, expand_products_for_terms
from einf.axis.matching import scalar_terms_to_atomic_tokens
from einf.backend import bind_array_namespace
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import build_runtime_execution_context
from einf.signature import Signature
from einf.tensor_types import TensorLike

from .solve import resolve_route_token_indices


def resolve_route_output_indices(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    tensors: tuple[TensorLike, ...],
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, ...]:
    """Resolve unique output->input tensor routing for one call context."""
    signature = Signature(inputs=lhs, outputs=rhs)
    if len(tensors) != signature.input_arity:
        raise ValidationError(
            code=ErrorCode.OP_ARITY_MISMATCH,
            message=(
                "route arity mismatch: "
                f"expected {signature.input_arity} inputs, got {len(tensors)}"
            ),
            help="pass exactly the tensors required by this route mapping",
            related=("route runtime",),
            data={"operation": "rearrange"},
        )

    explicit_sizes = signature.filter_explicit_sizes(dict(explicit_sizes_items))
    _validate_route_namespace(tensors=tensors)
    normalized = build_runtime_execution_context(
        signature=signature,
        tensors=tensors,
        explicit_sizes=explicit_sizes,
        input_shapes=input_shapes,
    )

    input_tokens = tuple(
        scalar_terms_to_atomic_tokens(
            expand_products_for_terms(axis_terms),
            normalized.axis_sizes,
        )
        for axis_terms in normalized.lhs_terms
    )
    output_tokens = tuple(
        scalar_terms_to_atomic_tokens(
            expand_products_for_terms(axis_terms),
            normalized.axis_sizes,
        )
        for axis_terms in normalized.rhs_terms
    )
    return resolve_route_token_indices(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def route_outputs(
    *,
    tensors: tuple[TensorLike, ...],
    output_indices: tuple[int, ...],
) -> tuple[TensorLike, ...]:
    """Apply resolved output routing indices to one tensor tuple."""
    return tuple(tensors[index] for index in output_indices)


def _validate_route_namespace(
    *,
    tensors: tuple[TensorLike, ...],
) -> None:
    """Validate that route tensors share one compatible Array API namespace."""
    try:
        namespace_binding = bind_array_namespace(*tensors)
    except Exception as error:
        raise ValidationError(
            code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
            message=(
                "backend dispatch unsupported input: "
                "route runtime requires one compatible array namespace family"
            ),
            help="use tensors from one compatible Array API namespace family",
            related=("backend dispatch",),
            data={"operation": "rearrange"},
        ) from error
    if namespace_binding.as_array_namespace(()) is not None:
        return
    raise ValidationError(
        code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
        message=(
            "backend dispatch unsupported input: "
            "route runtime requires one compatible array namespace family"
        ),
        help="use tensors from one compatible Array API namespace family",
        related=("backend dispatch",),
        data={"operation": "rearrange"},
    )


__all__ = [
    "resolve_route_output_indices",
    "route_outputs",
]
