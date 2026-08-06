import inspect
from collections.abc import Callable
from dataclasses import dataclass
from dis import get_instructions
from functools import partial
from types import CodeType, TracebackType
from typing import Literal, Protocol, TypeGuard, cast

from array_api_compat import array_namespace

from einf.backend.namespace import (
    derive_namespace_id,
    infer_backend_family,
)

try:
    from typing import Never
except ImportError:  # pragma: no cover
    from typing_extensions import Never

from einf.backend import ArrayNamespace, BackendArrayOps
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError, ValidationError
from einf.reduction.schema import (
    CanonicalReducer,
    ReducerCallable,
    ReducerName,
    ReducerResult,
)
from einf.tensor_types import TensorLike

ReducerCallMode = Literal[
    "axis_keyword",
    "axis_positional",
    "tensor_only",
    "fallback",
]

NamespaceReducer = Callable[..., ReducerResult]


def resolve_namespace_reducer(
    xp: ArrayNamespace,
    reducer_name: ReducerName,
) -> NamespaceReducer | None:
    """Resolve one canonical reducer name to a callable namespace primitive."""
    reducer_candidate = getattr(xp, reducer_name.value, None)
    if not callable(reducer_candidate):
        return None
    return cast(NamespaceReducer, reducer_candidate)


@dataclass(frozen=True, slots=True)
class ReducerRuntimeContext:
    """Runtime reducer execution context for one backend namespace."""

    xp: ArrayNamespace
    backend_ops: BackendArrayOps | None = None

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

        if self.backend_ops is not None:
            try:
                reduced = self.backend_ops.reduce(
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
        else:
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
            case "fallback":
                return self._run_fallback(
                    tensor=tensor,
                    axes=axes,
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

    def _run_fallback(
        self,
        *,
        tensor: TensorLike,
        axes: tuple[int, ...],
        context: ReducerRuntimeContext,
    ) -> ReducerResult:
        """Try supported call forms for uninspectable callables."""
        call_attempts = (
            lambda: self.reducer(tensor, axis=axes),
            lambda: self.reducer(tensor, axes),
            lambda: self.reducer(tensor),
        )
        for call_attempt in call_attempts:
            try:
                return call_attempt()
            except TensorOpError:
                raise
            except TypeError as error:
                if self._is_binding_typeerror(error=error):
                    continue
                raise
            except Exception as error:
                raise context.custom_reducer_error(error=error) from error
        return context.raise_unsupported_reducer_signature()

    def _is_binding_typeerror(
        self,
        *,
        error: TypeError,
    ) -> bool:
        """Return whether TypeError occurred before entering reducer code."""
        body_codes = self._callable_body_codes()
        if body_codes is None:
            return False
        traceback_entry = error.__traceback__
        while traceback_entry is not None:
            entered_body = traceback_entry.tb_frame.f_code in body_codes
            failed_while_forwarding = (
                traceback_entry.tb_next is None
                and self._is_forwarding_call(traceback_entry)
            )
            if entered_body and not failed_while_forwarding:
                return False
            traceback_entry = traceback_entry.tb_next

        message = str(error)
        markers = (
            "unexpected keyword argument",
            "positional argument",
            "required positional argument",
            "missing 1 required",
            "missing 2 required",
            "takes",
            "given",
        )
        return any(marker in message for marker in markers)

    def _is_forwarding_call(self, traceback_entry: TracebackType, /) -> bool:
        """Return whether a callable failed while forwarding variadic arguments."""
        try:
            return any(
                instruction.offset == traceback_entry.tb_lasti
                and instruction.opname == "CALL_FUNCTION_EX"
                for instruction in get_instructions(traceback_entry.tb_frame.f_code)
            )
        except Exception:  # noqa: BLE001 - uncertain forwarding evidence fails closed
            return False

    def _callable_body_codes(self) -> frozenset[CodeType] | None:
        """Return leaf callable code objects that establish reducer body entry."""
        body_codes: set[CodeType] = set()
        candidates: list[object] = [self.reducer]
        seen: set[int] = set()
        while candidates:
            candidate = candidates.pop()
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            try:
                code = getattr(candidate, "__code__", None)
                wrapped = getattr(candidate, "__wrapped__", None)
            except Exception:  # noqa: BLE001 - preserve the original reducer error
                return None
            if isinstance(code, CodeType):
                body_codes.add(code)

            delegates: list[object] = []
            if wrapped is not None and id(wrapped) not in seen:
                delegates.append(wrapped)
            if isinstance(candidate, partial):
                delegate = candidate.func
                if id(delegate) not in seen:
                    delegates.append(delegate)
            if delegates:
                candidates.extend(delegates)
                continue

            try:
                call = type(candidate).__call__
                call_code = getattr(call, "__code__", None)
            except Exception:  # noqa: BLE001 - uncertain body entry must fail closed
                return None
            if isinstance(call_code, CodeType):
                body_codes.add(call_code)
        return frozenset(body_codes)


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
        tensor: TensorLike,
        xp: ArrayNamespace,
    ) -> CompiledReducer:
        """Compile one reducer against runtime backend and call-shape."""
        if isinstance(reducer, ReducerName):
            return self._compile_string_reducer(
                reducer_name=reducer,
                xp=xp,
            )
        return CompiledCallableReducer(
            invoker=CallableReducerInvoker(
                reducer=reducer,
                call_mode=self._resolve_callable_call_mode(
                    reducer=reducer,
                    axes=axes,
                    tensor=tensor,
                ),
            ),
        )

    def _compile_string_reducer(
        self,
        *,
        reducer_name: ReducerName,
        xp: ArrayNamespace,
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

    def _resolve_callable_call_mode(
        self,
        *,
        reducer: ReducerCallable,
        axes: tuple[int, ...],
        tensor: TensorLike,
    ) -> ReducerCallMode:
        """Resolve callable reducer invocation mode from inspectable signature."""
        try:
            signature = inspect.signature(reducer)
        except (TypeError, ValueError):
            return "fallback"

        if self._can_bind(signature, (tensor,), {"axis": axes}):
            return "axis_keyword"
        if self._can_bind(signature, (tensor, axes), {}):
            return "axis_positional"
        if self._can_bind(signature, (tensor,), {}):
            return "tensor_only"
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: reducer signature is unsupported",
            help="use (tensor), (tensor, axes), or (tensor, *, axis=...)",
            related=("reduce reducer",),
            data={"operation": "reduce"},
        )

    def _can_bind(
        self,
        signature: inspect.Signature,
        args: tuple[TensorLike] | tuple[TensorLike, tuple[int, ...]],
        kwargs: dict[str, tuple[int, ...]],
    ) -> bool:
        """Return whether reducer signature can bind given arguments."""
        _ = self
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            return False
        return True


REDUCER_COMPILER = ReducerCompiler()


__all__ = [
    "REDUCER_COMPILER",
    "CallableReducerInvoker",
    "CompiledCallableReducer",
    "CompiledReducer",
    "CompiledStringReducer",
    "ReducerCompiler",
    "ReducerRuntimeContext",
    "resolve_namespace_reducer",
]
