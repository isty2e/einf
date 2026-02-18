import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeGuard

try:
    from typing import Never
except ImportError:  # pragma: no cover
    from typing_extensions import Never

from einf.backend import ArrayNamespace, BackendArrayOps
from einf.diagnostics import ErrorCode, ValidationError
from einf.reduction.schema import (
    STRING_REDUCERS,
    Reducer,
    ReducerCallable,
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


@dataclass(frozen=True, slots=True)
class ReducerRuntimeContext:
    """Runtime reducer execution context for one backend namespace."""

    xp: ArrayNamespace
    backend_ops: BackendArrayOps | None = None

    def apply_string_reducer(
        self,
        *,
        reducer_name: str,
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
                    reducer_name=reducer_name,
                    tensor=tensor,
                    axes=axes,
                )
            except Exception as error:
                self.raise_string_reducer_error(
                    reducer_name=reducer_name,
                    error=error,
                )
        else:
            try:
                reduced = reducer_fn(tensor, axis=axes)
            except Exception as error:
                self.raise_string_reducer_error(
                    reducer_name=reducer_name,
                    error=error,
                )
        return self.coerce_output(reduced)

    def coerce_output(
        self,
        reduced: ReducerResult,
        /,
    ) -> TensorLike:
        """Coerce reducer outputs to TensorLike, allowing scalar outputs."""
        if self._is_tensor_like(reduced):
            return reduced

        if isinstance(reduced, (bool, int, float, complex)):
            coerced = self.xp.asarray(reduced)
            if self._is_tensor_like(coerced):
                return coerced

        self.raise_output_type_error()

    def raise_output_type_error(self) -> Never:
        """Raise normalized reducer output contract error."""
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: reducer output must be tensor-like",
            help="return a tensor or scalar value from reducer",
            related=("reduce reducer output",),
            data={},
        )

    def raise_string_reducer_error(
        self,
        *,
        reducer_name: str,
        error: Exception,
    ) -> Never:
        """Raise normalized string-reducer runtime error."""
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                f"inconsistent dims: backend reducer {reducer_name!r} failed: {error}"
            ),
            help=(
                "ensure reducer domain is valid for the selected axes "
                "(for example non-empty domain for max/min)"
            ),
            related=("reduce reducer",),
            data={"reducer": reducer_name},
        ) from error

    def raise_custom_reducer_error(
        self,
        *,
        error: Exception,
    ) -> Never:
        """Raise normalized custom-reducer runtime error."""
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=f"inconsistent dims: custom reducer failed: {error}",
            help="ensure reducer domain is valid for selected axes",
            related=("reduce reducer",),
            data={},
        ) from error

    def raise_unsupported_reducer_signature(self) -> Never:
        """Raise normalized unsupported reducer signature error."""
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: reducer signature is unsupported",
            help="use (tensor), (tensor, axes), or (tensor, *, axis=...)",
            related=("reduce reducer",),
            data={},
        )

    def _is_tensor_like(self, value: ReducerResult) -> TypeGuard[TensorLike]:
        """Return whether one reducer result satisfies TensorLike contract."""
        shape = getattr(value, "shape", None)
        if not isinstance(shape, tuple):
            return False
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int):
                return False
        return True


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

    name: str
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
        except TypeError:
            raise
        except Exception as error:
            return context.raise_custom_reducer_error(error=error)

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
            except TypeError as error:
                if self._is_binding_typeerror(error=error):
                    continue
                raise
            except Exception as error:
                return context.raise_custom_reducer_error(error=error)
        return context.raise_unsupported_reducer_signature()

    def _is_binding_typeerror(
        self,
        *,
        error: TypeError,
    ) -> bool:
        """Return whether TypeError likely came from call-signature binding."""
        reducer_name = getattr(self.reducer, "__name__", None)
        traceback_entry = error.__traceback__
        while traceback_entry is not None:
            frame_name = traceback_entry.tb_frame.f_code.co_name
            if frame_name == "__call__":
                return False
            if isinstance(reducer_name, str) and frame_name == reducer_name:
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
        reducer: Reducer,
        axes: tuple[int, ...],
        tensor: TensorLike,
        xp: ArrayNamespace,
    ) -> CompiledReducer:
        """Compile one reducer against runtime backend and call-shape."""
        if isinstance(reducer, str):
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
        reducer_name: str,
        xp: ArrayNamespace,
    ) -> CompiledStringReducer:
        """Compile one string reducer by resolving namespace callable."""
        if reducer_name not in STRING_REDUCERS:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: unsupported reducer {reducer_name!r}",
                help="use one of: sum, prod, mean, max, min, all, any",
                related=("reduce reducer",),
                data={},
            )
        reducer_candidate = getattr(xp, reducer_name, None)
        if not callable(reducer_candidate):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: backend reducer "
                    f"{reducer_name!r} is unavailable"
                ),
                help="choose a reducer available on the active backend namespace",
                related=("reduce reducer",),
                data={},
            )
        return CompiledStringReducer(
            name=reducer_name,
            reducer_fn=reducer_candidate,
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
            data={},
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
    "CallableReducerInvoker",
    "CompiledCallableReducer",
    "CompiledReducer",
    "CompiledStringReducer",
    "REDUCER_COMPILER",
    "ReducerCompiler",
    "ReducerRuntimeContext",
]
