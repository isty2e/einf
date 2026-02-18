from math import prod

from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    bind_array_namespace,
    derive_namespace_id,
    is_namespace_family,
    load_backend_module,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.tensor_types import TensorLike

from .constants import (
    RESHAPE_REQUIRED_NAMESPACE_METHODS,
    ZERO_COPY_ALLOWED_RESHAPE_MODE,
    ZERO_COPY_REQUIRED_RESHAPE_MODE,
    ZeroCopyReshapeMode,
)
from .model import ReshapeCompiledProgram
from .resolve import resolve_reshape_target_shape


def validate_rearrange_numel(
    *,
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> None:
    """Validate rearrange numel preservation for one unary reshape transform."""
    input_total = prod(input_shape, start=1)
    output_total = prod(target_shape, start=1)
    if output_total > input_total:
        raise ValidationError(
            code=ErrorCode.NUMEL_MISMATCH_GROW,
            message=(
                "numel mismatch grow: rearrange output numel is larger than input numel"
            ),
            help="keep output numel equal to input",
            related=("rearrange schema",),
            data={"operation": "rearrange"},
        )
    if output_total < input_total:
        raise ValidationError(
            code=ErrorCode.NUMEL_MISMATCH_SHRINK,
            message=(
                "numel mismatch shrink: rearrange output numel is smaller than input numel"
            ),
            help="keep output numel equal to input",
            related=("rearrange schema",),
            data={"operation": "rearrange"},
        )


def run_reshape_program(
    *,
    tensor: TensorLike,
    target_shape: tuple[int, ...],
    backend_ops: BackendArrayOps | None,
    xp: ArrayNamespace | None,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
) -> TensorLike:
    """Run one unary reshape program via backend-native reshape primitives."""
    if backend_ops is not None:
        if tensor.shape == target_shape:
            return tensor
        try:
            if (
                zero_copy_mode == ZERO_COPY_REQUIRED_RESHAPE_MODE
                and backend_ops.backend_family == "numpy"
            ):
                numpy_module = load_backend_module("numpy")
                return numpy_module.reshape(tensor, target_shape, order="A")
            return backend_ops.reshape(tensor, target_shape)
        except Exception:
            if xp is None:
                raise

    if xp is None:
        raise ValueError("array namespace is required for reshape runtime")
    if tensor.shape == target_shape:
        return tensor
    if zero_copy_mode == ZERO_COPY_REQUIRED_RESHAPE_MODE:
        try:
            namespace_id = derive_namespace_id(xp)
        except TypeError:
            namespace_id = ""
        if namespace_id and is_namespace_family(namespace_id, "numpy"):
            numpy_module = load_backend_module("numpy")
            return numpy_module.reshape(tensor, target_shape, order="A")
    return xp.reshape(tensor, target_shape)


def try_run_reshape_program(
    *,
    tensor: TensorLike,
    explicit_sizes: dict[str, int],
    program: ReshapeCompiledProgram,
    backend_ops: BackendArrayOps | None = None,
    xp: ArrayNamespace | None = None,
    zero_copy_mode: ZeroCopyReshapeMode = ZERO_COPY_ALLOWED_RESHAPE_MODE,
) -> TensorLike | None:
    """Try symbolic unary-reshape runtime; return None when ineligible."""
    target_shape = resolve_reshape_target_shape(
        tensor=tensor,
        explicit_sizes=explicit_sizes,
        program=program,
    )
    if target_shape is None:
        return None
    validate_rearrange_numel(
        input_shape=tensor.shape,
        target_shape=target_shape,
    )

    namespace_xp = xp
    if namespace_xp is None and backend_ops is None:
        try:
            namespace_binding = bind_array_namespace(tensor)
        except Exception:
            return None
        namespace_xp = namespace_binding.as_array_namespace(
            RESHAPE_REQUIRED_NAMESPACE_METHODS
        )
        if namespace_xp is None:
            return None

    try:
        return run_reshape_program(
            tensor=tensor,
            target_shape=target_shape,
            backend_ops=backend_ops,
            xp=namespace_xp,
            zero_copy_mode=zero_copy_mode,
        )
    except (TypeError, ValueError, RuntimeError):
        return None


__all__ = [
    "run_reshape_program",
    "try_run_reshape_program",
]
