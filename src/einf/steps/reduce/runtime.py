from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeGuard, cast

from array_api_compat import array_namespace

from einf.backend.namespace import (
    derive_namespace_id,
    infer_backend_family,
)

try:
    from typing import Never
except ImportError:  # pragma: no cover
    from typing_extensions import Never

from einf.backend import BackendArrayOps, BackendProfile
from einf.backend.runtime import is_trusted_backend_array_ops
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError, ValidationError
from einf.reduction.callable import ReducerResult
from einf.reduction.schema import (
    CanonicalReducer,
    ReducerCallable,
    ReducerName,
)
from einf.steps.runtime import backend_specialization_error
from einf.tensor_types import TensorLike

from .callable_contract import ReducerCallMode, resolve_callable_reducer_mode

NamespaceReducer = Callable[..., ReducerResult]


class ReducerArrayNamespace(Protocol):
    """Array namespace surface required by reducer output normalization."""

    __name__: str

    def asarray(
        self,
        value: bool | complex,
        /,
    ) -> TensorLike:
        """Convert a scalar reducer result to a backend tensor.

        Parameters
        ----------
        value
            Scalar reducer output.

        Returns
        -------
        TensorLike
            Scalar tensor owned by this namespace.
        """
        ...


def bind_reducer_namespace(namespace: object, /) -> ReducerArrayNamespace:
    """Bind the namespace surface required by reducer execution.

    Parameters
    ----------
    namespace
        Runtime namespace to normalize.

    Returns
    -------
    ReducerArrayNamespace
        Namespace with scalar output coercion support.

    Raises
    ------
    ValidationError
        The namespace does not expose a callable ``asarray`` primitive.
    ExecutionError
        Namespace attribute lookup fails unexpectedly.
    """
    try:
        asarray = getattr(namespace, "asarray", None)
    except TensorOpError:
        raise
    except Exception as error:
        raise backend_specialization_error(
            operation="reduce",
            error=error,
        ) from error
    if not callable(asarray):
        raise ValidationError(
            code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
            message=(
                "backend dispatch unsupported input: "
                "reduce requires namespace scalar coercion"
            ),
            help="provide an array namespace with a callable asarray primitive",
            related=("backend dispatch",),
            data={"operation": "reduce"},
        )
    return cast(ReducerArrayNamespace, namespace)


def resolve_namespace_reducer(
    xp: ReducerArrayNamespace,
    reducer_name: ReducerName,
) -> NamespaceReducer | None:
    """Resolve one canonical reducer name to a namespace primitive.

    Parameters
    ----------
    xp
        Canonical reducer namespace.
    reducer_name
        Selected named reducer.

    Returns
    -------
    Callable or None
        Selected reducer callable, or ``None`` when unavailable.

    Raises
    ------
    ExecutionError
        Namespace attribute lookup fails unexpectedly.
    """
    try:
        reducer_candidate = getattr(xp, reducer_name.value, None)
    except TensorOpError:
        raise
    except AttributeError:
        return None
    except Exception as error:
        raise backend_specialization_error(
            operation="reduce",
            error=error,
        ) from error
    if not callable(reducer_candidate):
        return None
    return cast(NamespaceReducer, reducer_candidate)


@dataclass(frozen=True, slots=True)
class ReducerRuntimeContext:
    """Runtime reducer execution context for one backend namespace."""

    xp: ReducerArrayNamespace
    backend_ops: BackendArrayOps | None = None

    def native_reducer_ops(
        self,
        tensor: TensorLike,
        /,
    ) -> BackendArrayOps | None:
        """Return the adapter when tensor and adapter prove a native route.

        Parameters
        ----------
        tensor
            Runtime tensor considered for native reducer execution.

        Returns
        -------
        BackendArrayOps or None
            Canonical adapter for a trusted native route, or ``None``.
        """
        backend_ops = self.backend_ops
        if backend_ops is None or not is_trusted_backend_array_ops(
            backend_ops=backend_ops,
            tensor=tensor,
        ):
            return None
        return backend_ops

    def apply_string_reducer(
        self,
        *,
        reducer_name: ReducerName,
        reducer_fn: NamespaceReducer,
        tensor: TensorLike,
        axes: tuple[int, ...],
    ) -> TensorLike:
        """Apply one string reducer and normalize output/error contracts."""
        if not axes:
            return tensor

        backend_ops = self.native_reducer_ops(tensor)
        if backend_ops is not None:
            try:
                reduced = backend_ops.reduce(
                    reducer_name=reducer_name.value,
                    tensor=tensor,
                    axes=axes,
                )
            except TensorOpError:
                raise
            except Exception as error:
                raise self.string_reducer_error(
                    reducer_name=reducer_name,
                    tensor=tensor,
                    axes=axes,
                    error=error,
                ) from error
            return self.coerce_output(reduced)

        return self.apply_namespace_reducer(
            reducer_name=reducer_name,
            reducer_fn=reducer_fn,
            tensor=tensor,
            axes=axes,
        )

    def apply_namespace_reducer(
        self,
        *,
        reducer_name: ReducerName,
        reducer_fn: NamespaceReducer,
        tensor: TensorLike,
        axes: tuple[int, ...],
    ) -> TensorLike:
        """Apply the reducer selected from the resolved array namespace.

        Parameters
        ----------
        reducer_name
            Canonical reducer name used in diagnostics.
        reducer_fn
            Reducer callable selected from the resolved namespace.
        tensor
            Runtime tensor to reduce.
        axes
            Concrete axis positions to reduce. An empty tuple leaves the input
            unchanged.

        Returns
        -------
        TensorLike
            Normalized reducer output.

        Raises
        ------
        TensorOpError
            The reducer or output normalization fails.
        """
        if not axes:
            return tensor

        try:
            reduced = reducer_fn(tensor, axis=axes)
        except TensorOpError:
            raise
        except Exception as error:
            raise self.string_reducer_error(
                reducer_name=reducer_name,
                tensor=tensor,
                axes=axes,
                error=error,
            ) from error
        return self.coerce_output(reduced)

    def coerce_output(
        self,
        reduced: ReducerResult,
        /,
    ) -> TensorLike:
        """Coerce reducer outputs to TensorLike, allowing scalar outputs."""
        if self._is_tensor_like(reduced):
            self._validate_output_ownership(reduced)
            return reduced

        if isinstance(reduced, (bool, int, float, complex)):
            try:
                coerced = self.xp.asarray(reduced)
            except TensorOpError:
                raise
            except Exception as error:
                raise ExecutionError(
                    code=ErrorCode.BACKEND_EXECUTION_FAILED,
                    message=(
                        f"backend execution failed: reducer output coercion failed: {error}"
                    ),
                    help=("ensure the backend can materialize scalar reducer outputs"),
                    related=("reduce reducer output",),
                    data={"operation": "reduce"},
                ) from error
            if self._is_tensor_like(coerced):
                self._validate_output_ownership(coerced)
                return coerced

        self.raise_output_type_error()

    def _validate_output_ownership(self, output: TensorLike) -> None:
        """Reject reducer outputs owned by a different backend namespace."""
        try:
            output_namespace = array_namespace(output)
            output_namespace_id = derive_namespace_id(output_namespace)
            input_namespace_id = derive_namespace_id(self.xp)
        except TensorOpError:
            raise
        except Exception as error:
            raise ExecutionError(
                code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
                message=(
                    "reduce output protocol violation: reducer output namespace "
                    f"could not be resolved: {error}"
                ),
                help="return a tensor with a valid backend namespace",
                related=("reduce reducer output",),
                data={"operation": "reduce"},
            ) from error

        input_family = infer_backend_family(input_namespace_id)
        if (input_family is None and output_namespace is not self.xp) or (
            input_family is not None
            and infer_backend_family(output_namespace_id) != input_family
        ):
            raise ExecutionError(
                code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
                message=(
                    "reduce output protocol violation: reducer output belongs to a "
                    "different backend namespace"
                ),
                help=(
                    "return tensors from the same backend namespace as "
                    "the reduced input"
                ),
                related=("reduce reducer output",),
                data={"operation": "reduce"},
            )

    def raise_output_type_error(self) -> Never:
        """Raise normalized reducer output contract error."""
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message="reduce output protocol violation: output must be tensor-like",
            help="return a tensor or scalar value from reducer",
            related=("reduce reducer output",),
            data={"operation": "reduce"},
        )

    def string_reducer_error(
        self,
        *,
        reducer_name: ReducerName,
        tensor: TensorLike,
        axes: tuple[int, ...],
        error: Exception,
    ) -> ValidationError | ExecutionError:
        """Classify one string-reducer runtime error from canonical facts."""
        if reducer_name in {ReducerName.MAX, ReducerName.MIN} and any(
            tensor.shape[axis] == 0 for axis in axes
        ):
            return ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: backend reducer "
                    f"{reducer_name.value!r} failed: {error}"
                ),
                help="use a non-empty reduction domain for max/min",
                related=("reduce reducer",),
                data={"operation": "reduce", "reducer": reducer_name.value},
            )

        return ExecutionError(
            code=ErrorCode.BACKEND_EXECUTION_FAILED,
            message=(
                "backend execution failed: backend reducer "
                f"{reducer_name.value!r} failed: {error}"
            ),
            help="ensure the active backend reducer is operational for this tensor",
            related=("reduce reducer", "backend execution"),
            data={"operation": "reduce", "reducer": reducer_name.value},
        )

    def custom_reducer_error(
        self,
        *,
        error: Exception,
    ) -> ValidationError:
        """Build one normalized custom-reducer runtime error."""
        return ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=f"inconsistent dims: custom reducer failed: {error}",
            help="ensure reducer domain is valid for selected axes",
            related=("reduce reducer",),
            data={"operation": "reduce"},
        )

    def raise_unsupported_reducer_signature(self) -> Never:
        """Raise normalized unsupported reducer signature error."""
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: reducer signature is unsupported",
            help="use (tensor), (tensor, axes), or (tensor, *, axis=...)",
            related=("reduce reducer",),
            data={"operation": "reduce"},
        )

    def _is_tensor_like(self, value: ReducerResult) -> TypeGuard[TensorLike]:
        """Return whether one reducer result satisfies TensorLike contract."""
        try:
            shape = getattr(value, "shape", None)
            if not isinstance(shape, tuple):
                return False
            for dim in shape:
                if isinstance(dim, bool) or not isinstance(dim, int):
                    return False
            return callable(getattr(value, "__getitem__", None))
        except TensorOpError:
            raise
        except Exception:  # noqa: BLE001 - malformed user output is a type failure
            return False


@dataclass(frozen=True, slots=True)
class ReducerRuntimeBinding:
    """Canonical backend identity and reducer capability binding.

    Parameters
    ----------
    profile : BackendProfile
        Backend identity selected for runtime specialization.
    context : ReducerRuntimeContext
        Reducer namespace and native adapter bound from ``profile``.

    Raises
    ------
    ValueError
        ``context`` was bound from a different backend namespace or family.
    """

    profile: BackendProfile
    context: ReducerRuntimeContext

    def __post_init__(self) -> None:
        if self.context.xp is not self.profile.namespace:
            raise ValueError(
                "reducer runtime context namespace must match backend profile"
            )
        backend_ops = self.context.backend_ops
        if (
            backend_ops is not None
            and backend_ops.backend_family != self.profile.backend_family
        ):
            raise ValueError(
                "reducer runtime context family must match backend profile"
            )


class CompiledReducer(Protocol):
    """Compiled reducer protocol for runtime phase execution."""

    def apply(
        self,
        *,
        tensor: TensorLike,
        axes: tuple[int, ...],
        context: ReducerRuntimeContext,
    ) -> TensorLike:
        """Apply compiled reducer over concrete axis indices."""
        ...


@dataclass(frozen=True, slots=True)
class CompiledStringReducer:
    """Compiled string reducer resolved against one backend namespace."""

    name: ReducerName
    reducer_fn: NamespaceReducer

    def apply(
        self,
        *,
        tensor: TensorLike,
        axes: tuple[int, ...],
        context: ReducerRuntimeContext,
    ) -> TensorLike:
        """Apply one compiled string reducer."""
        return context.apply_string_reducer(
            reducer_name=self.name,
            reducer_fn=self.reducer_fn,
            tensor=tensor,
            axes=axes,
        )


@dataclass(frozen=True, slots=True)
class CallableReducerInvoker:
    """Runtime callable-reducer invocation strategy."""

    reducer: ReducerCallable
    call_mode: ReducerCallMode

    def invoke(
        self,
        *,
        tensor: TensorLike,
        axes: tuple[int, ...],
        context: ReducerRuntimeContext,
    ) -> ReducerResult:
        """Invoke callable reducer with configured call mode."""
        match self.call_mode:
            case "axis_keyword":
                return self._run_checked(
                    call_attempt=lambda: self.reducer(tensor, axis=axes),
                    context=context,
                )
            case "axis_positional":
                return self._run_checked(
                    call_attempt=lambda: self.reducer(tensor, axes),
                    context=context,
                )
            case "tensor_only":
                return self._run_checked(
                    call_attempt=lambda: self.reducer(tensor),
                    context=context,
                )
        return context.raise_unsupported_reducer_signature()

    def _run_checked(
        self,
        *,
        call_attempt: Callable[[], ReducerResult],
        context: ReducerRuntimeContext,
    ) -> ReducerResult:
        """Run one inspectable call with normalized error mapping."""
        try:
            return call_attempt()
        except TensorOpError:
            raise
        except TypeError:
            raise
        except Exception as error:
            raise context.custom_reducer_error(error=error) from error


@dataclass(frozen=True, slots=True)
class CompiledCallableReducer:
    """Compiled callable reducer with resolved invocation strategy."""

    invoker: CallableReducerInvoker

    def apply(
        self,
        *,
        tensor: TensorLike,
        axes: tuple[int, ...],
        context: ReducerRuntimeContext,
    ) -> TensorLike:
        """Apply one compiled callable reducer."""
        if not axes:
            return tensor
        reduced = self.invoker.invoke(
            tensor=tensor,
            axes=axes,
            context=context,
        )
        return context.coerce_output(reduced)


@dataclass(frozen=True, slots=True)
class ReducerCompiler:
    """Compile runtime reducers from reducer declarations."""

    def compile(
        self,
        *,
        reducer: CanonicalReducer,
        axes: tuple[int, ...],
        xp: ReducerArrayNamespace,
    ) -> CompiledReducer:
        """Compile one reducer against a runtime namespace and call shape.

        Parameters
        ----------
        reducer
            Canonical named or callable reducer.
        axes
            Concrete axes reduced by the compiled invocation.
        xp
            Canonical reducer namespace.

        Returns
        -------
        CompiledReducer
            Runtime reducer bound to the selected call form.

        Raises
        ------
        ValidationError
            The selected reducer is unavailable or has an invalid call form.
        ExecutionError
            Namespace capability lookup fails unexpectedly.
        """
        if isinstance(reducer, ReducerName):
            return self._compile_string_reducer(
                reducer_name=reducer,
                xp=xp,
            )
        return CompiledCallableReducer(
            invoker=CallableReducerInvoker(
                reducer=reducer.materialize(),
                call_mode=resolve_callable_reducer_mode(reducer, axes=axes),
            ),
        )

    def _compile_string_reducer(
        self,
        *,
        reducer_name: ReducerName,
        xp: ReducerArrayNamespace,
    ) -> CompiledStringReducer:
        """Compile one string reducer by resolving namespace callable."""
        reducer_fn = resolve_namespace_reducer(xp, reducer_name)
        if reducer_fn is None:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: backend reducer "
                    f"{reducer_name.value!r} is unavailable"
                ),
                help="choose a reducer available on the active backend namespace",
                related=("reduce reducer",),
                data={
                    "operation": "reduce",
                    "reducer": reducer_name.value,
                },
            )
        return CompiledStringReducer(
            name=reducer_name,
            reducer_fn=reducer_fn,
        )


REDUCER_COMPILER = ReducerCompiler()


__all__ = [
    "REDUCER_COMPILER",
    "CallableReducerInvoker",
    "CompiledCallableReducer",
    "CompiledReducer",
    "CompiledStringReducer",
    "ReducerArrayNamespace",
    "ReducerCompiler",
    "ReducerRuntimeBinding",
    "ReducerRuntimeContext",
    "bind_reducer_namespace",
    "resolve_namespace_reducer",
]
