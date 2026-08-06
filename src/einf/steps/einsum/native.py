from collections.abc import Callable
from typing import cast

from einf.backend.namespace import (
    ArrayNamespaceLike,
    derive_namespace_id,
    is_namespace_family,
)
from einf.backend.runtime import BackendRuntimeUnavailable, load_backend_module
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError
from einf.steps.runtime import FALLBACK_ELIGIBLE_BACKEND_ERRORS
from einf.tensor_types import TensorLike


def try_native_contract_einsum(
    *,
    equation: str,
    tensors: tuple[TensorLike, ...],
    namespace: ArrayNamespaceLike,
) -> TensorLike | None:
    """Run native backend einsum for contract execution."""
    try:
        namespace_einsum_candidate = getattr(namespace, "einsum", None)
    except TensorOpError:
        raise
    except AttributeError:
        namespace_einsum_candidate = None
    except Exception as error:
        raise _native_einsum_error(error) from error
    if callable(namespace_einsum_candidate):
        namespace_einsum = cast(Callable[..., TensorLike], namespace_einsum_candidate)
        try:
            return namespace_einsum(equation, *tensors)
        except TensorOpError:
            raise
        except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
            return None
        except Exception as error:
            raise _native_einsum_error(error) from error

    try:
        namespace_id = derive_namespace_id(namespace)
    except TensorOpError:
        raise
    except TypeError:
        return None
    except Exception as error:
        raise ExecutionError(
            code=ErrorCode.BACKEND_EXECUTION_FAILED,
            message=f"backend execution failed: native einsum failed: {error}",
            help="provide an array namespace with one stable identifier",
            related=("einsum execution",),
            data={"operation": "einsum"},
        ) from error
    if not is_namespace_family(namespace_id, "torch"):
        return None

    try:
        torch_module = load_backend_module("torch")
    except BackendRuntimeUnavailable:
        return None
    except TensorOpError:
        raise
    except Exception as error:
        raise ExecutionError(
            code=ErrorCode.BACKEND_EXECUTION_FAILED,
            message=f"backend execution failed: native einsum failed: {error}",
            help="ensure the selected backend runtime initializes correctly",
            related=("einsum execution",),
            data={"operation": "einsum"},
        ) from error
    try:
        torch_einsum_candidate = getattr(torch_module, "einsum", None)
    except TensorOpError:
        raise
    except AttributeError:
        torch_einsum_candidate = None
    except Exception as error:
        raise _native_einsum_error(error) from error
    if not callable(torch_einsum_candidate):
        return None
    torch_einsum = cast(Callable[..., TensorLike], torch_einsum_candidate)

    try:
        return torch_einsum(equation, *tensors)
    except TensorOpError:
        raise
    except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
        return None
    except Exception as error:
        raise _native_einsum_error(error) from error


def _native_einsum_error(error: Exception, /) -> ExecutionError:
    """Build one structured native-einsum execution failure."""
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: native einsum failed: {error}",
        help="ensure the selected backend supports einsum for the operands",
        related=("einsum execution",),
        data={"operation": "einsum"},
    )


__all__ = ["try_native_contract_einsum"]
