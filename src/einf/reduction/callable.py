import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

from ..diagnostics import ErrorCode, ValidationError
from ..tensor_types import TensorLike

ReducerResult: TypeAlias = TensorLike | bool | int | float | complex
ReducerCallable: TypeAlias = Callable[..., ReducerResult]


@dataclass(frozen=True, slots=True)
class _SignatureProxy:
    __signature__: inspect.Signature

    def __call__(self, *args: object, **kwargs: object) -> ReducerResult:
        raise RuntimeError("signature proxy is not executable")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CallableReducerBinding:
    """Immutable canonical invocation binding for a callable reducer.

    Parameters
    ----------
    reducer
        Callable reducer to normalize. Exact ``functools.partial`` chains are
        flattened into immutable argument and keyword tuples.

    Raises
    ------
    ValidationError
        If callable signature metadata is invalid or conflicts with configured
        partial bindings, the partial chain visible at ingress contains a
        subclass or unresolved ``functools.Placeholder``, or a partialmethod
        does not come through normal descriptor binding.
    """

    target: ReducerCallable
    positional_args: tuple[object, ...]
    keyword_items: tuple[tuple[str, object], ...]
    visible_signature: inspect.Signature
    target_signature: inspect.Signature

    def __init__(self, reducer: ReducerCallable) -> None:
        (
            target,
            positional_args,
            keyword_items,
            is_partial,
            has_partial_signature_metadata,
        ) = _capture_partial_chain(reducer)
        if _requires_partialmethod_descriptor_binding(target):
            raise _partialmethod_descriptor_binding_error()
        if has_partial_signature_metadata and (positional_args or keyword_items):
            raise _configured_partial_metadata_error()
        target_signature = _inspect_callable_signature(target)
        if has_partial_signature_metadata:
            visible_signature = _inspect_callable_signature(reducer)
        elif is_partial:
            visible_signature = _project_partial_signature(
                target_signature=target_signature,
                positional_args=positional_args,
                keyword_items=keyword_items,
            )
        else:
            visible_signature = target_signature

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "positional_args", positional_args)
        object.__setattr__(self, "keyword_items", keyword_items)
        object.__setattr__(self, "visible_signature", visible_signature)
        object.__setattr__(self, "target_signature", target_signature)

    def materialize(self) -> ReducerCallable:
        """Return the executable callable represented by this binding.

        Returns
        -------
        ReducerCallable
            Callable with the captured positional and keyword configuration.
        """
        if not self.positional_args and not self.keyword_items:
            return self.target
        return functools.partial(
            self.target,
            *self.positional_args,
            **dict(self.keyword_items),
        )

    def bind_configuration(self) -> inspect.BoundArguments:
        """Bind captured configuration against the target call surface.

        Returns
        -------
        inspect.BoundArguments
            Parameter binding for the captured configuration only.

        Raises
        ------
        TypeError
            If the captured configuration does not fit the target signature.
        """
        return self.target_signature.bind_partial(
            *self.positional_args,
            **dict(self.keyword_items),
        )

    def __hash__(self) -> int:
        """Hash every captured fact that can change invocation structure."""
        return hash(
            (
                id(self.target),
                tuple(id(value) for value in self.positional_args),
                tuple((name, id(value)) for name, value in self.keyword_items),
                _signature_shape(self.visible_signature),
                _signature_shape(self.target_signature),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare bindings without invoking user-defined value equality."""
        if not isinstance(other, CallableReducerBinding):
            return False
        return bool(
            self.target is other.target
            and _values_are_identical(
                self.positional_args,
                other.positional_args,
            )
            and _keyword_items_are_identical(
                self.keyword_items,
                other.keyword_items,
            )
            and _signature_shape(self.visible_signature)
            == _signature_shape(other.visible_signature)
            and _signature_shape(self.target_signature)
            == _signature_shape(other.target_signature)
        )


def _capture_partial_chain(
    reducer: ReducerCallable,
    /,
) -> tuple[
    ReducerCallable,
    tuple[object, ...],
    tuple[tuple[str, object], ...],
    bool,
    bool,
]:
    layers: list[tuple[tuple[object, ...], tuple[tuple[str, object], ...]]] = []
    current = reducer
    has_partial_signature_metadata = False
    while isinstance(current, functools.partial):
        if type(current) is not functools.partial:
            raise _partial_subclass_error()
        has_partial_signature_metadata = bool(
            has_partial_signature_metadata or _has_visible_signature_metadata(current)
        )
        layers.append(
            (
                current.args,
                tuple((current.keywords or {}).items()),
            )
        )
        current = cast(ReducerCallable, current.func)

    positional_args: tuple[object, ...] = ()
    keywords: dict[str, object] = {}
    for layer_args, layer_keywords in reversed(layers):
        positional_args = _compose_partial_args(positional_args, layer_args)
        keywords.update(layer_keywords)
    _reject_unresolved_placeholders(positional_args)
    return (
        current,
        positional_args,
        tuple(keywords.items()),
        bool(layers),
        has_partial_signature_metadata,
    )


def _compose_partial_args(
    configured_args: tuple[object, ...],
    outer_args: tuple[object, ...],
    /,
) -> tuple[object, ...]:
    placeholder = getattr(functools, "Placeholder", None)
    if placeholder is None:
        return (*configured_args, *outer_args)

    remaining_outer_args = iter(outer_args)
    composed_args: list[object] = []
    for configured_arg in configured_args:
        if configured_arg is not placeholder:
            composed_args.append(configured_arg)
            continue
        try:
            composed_args.append(next(remaining_outer_args))
        except StopIteration:
            composed_args.append(placeholder)
    composed_args.extend(remaining_outer_args)
    return tuple(composed_args)


def _project_partial_signature(
    *,
    target_signature: inspect.Signature,
    positional_args: tuple[object, ...],
    keyword_items: tuple[tuple[str, object], ...],
) -> inspect.Signature:
    proxy = _SignatureProxy(target_signature)
    configured = functools.partial(
        proxy,
        *positional_args,
        **dict(keyword_items),
    )
    try:
        return inspect.signature(configured)
    except Exception as error:
        raise _callable_signature_error() from error


def _inspect_callable_signature(reducer: ReducerCallable, /) -> inspect.Signature:
    signature_target = reducer
    if not (
        _has_visible_signature_metadata(reducer)
        or inspect.isroutine(reducer)
        or inspect.isclass(reducer)
        or isinstance(reducer, functools.partial)
    ):
        signature_target = _bind_call_descriptor(reducer)

    try:
        return inspect.signature(signature_target)
    except Exception as error:
        raise _callable_signature_error() from error


def _has_visible_signature_metadata(reducer: ReducerCallable, /) -> bool:
    for attribute_name in ("__signature__", "__wrapped__"):
        try:
            metadata = inspect.getattr_static(reducer, attribute_name)
        except AttributeError:
            continue
        if attribute_name == "__wrapped__" or metadata is not None:
            return True
    return False


def _bind_call_descriptor(reducer: ReducerCallable, /) -> ReducerCallable:
    try:
        descriptor = inspect.getattr_static(type(reducer), "__call__")
        descriptor_get = getattr(descriptor, "__get__", None)
        bound_call = (
            descriptor_get(reducer, type(reducer))
            if callable(descriptor_get)
            else descriptor
        )
    except Exception as error:
        raise _callable_signature_error() from error
    if not callable(bound_call):
        raise _callable_signature_error()
    return cast(ReducerCallable, bound_call)


def _reject_unresolved_placeholders(args: tuple[object, ...], /) -> None:
    placeholder = getattr(functools, "Placeholder", None)
    if placeholder is not None and any(value is placeholder for value in args):
        raise _placeholder_error()


def _requires_partialmethod_descriptor_binding(
    reducer: ReducerCallable,
    /,
) -> bool:
    candidate = reducer.__func__ if inspect.ismethod(reducer) else reducer
    if not inspect.isfunction(candidate):
        return False
    for attribute_name in ("_partialmethod", "__partialmethod__"):
        try:
            metadata = inspect.getattr_static(candidate, attribute_name)
        except AttributeError:
            continue
        if isinstance(metadata, functools.partialmethod):
            return True
    return False


def _signature_shape(
    signature: inspect.Signature,
    /,
) -> tuple[tuple[str, int, bool], ...]:
    return tuple(
        (
            parameter.name,
            int(parameter.kind),
            parameter.default is not inspect.Parameter.empty,
        )
        for parameter in signature.parameters.values()
    )


def _values_are_identical(
    first: tuple[object, ...],
    second: tuple[object, ...],
    /,
) -> bool:
    return bool(
        len(first) == len(second)
        and all(left is right for left, right in zip(first, second))
    )


def _keyword_items_are_identical(
    first: tuple[tuple[str, object], ...],
    second: tuple[tuple[str, object], ...],
    /,
) -> bool:
    return bool(
        len(first) == len(second)
        and all(
            left_name == right_name and left_value is right_value
            for (left_name, left_value), (right_name, right_value) in zip(
                first,
                second,
            )
        )
    )


def _callable_signature_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "callable reducer is ambiguous: its signature is unavailable or "
            "contains only *args/**kwargs"
        ),
        help=(
            "use a named reducer (for example, 'sum') or wrap the callable "
            "in a function with one of these signatures: (tensor), "
            "(tensor, axes), or (tensor, *, axis=...)"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _partial_subclass_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: functools.partial subclasses are unsupported",
        help="use functools.partial directly or an explicit reducer wrapper",
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _partialmethod_descriptor_binding_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: partialmethod reducer must use descriptor binding"
        ),
        help=(
            "pass the reducer through its instance (for example, instance.reducer); "
            "MethodType and manual __get__ binding hide partialmethod configuration"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _configured_partial_metadata_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: configured partial signature metadata is ambiguous"
        ),
        help=(
            "put __wrapped__ or __signature__ on the underlying explicit wrapper "
            "before applying functools.partial"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _placeholder_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: unresolved functools.Placeholder binding",
        help="fill every Placeholder or replace the partial with an explicit wrapper",
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


__all__ = ["CallableReducerBinding", "ReducerCallable", "ReducerResult"]
