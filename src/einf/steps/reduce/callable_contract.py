import inspect
from collections.abc import (
    Iterable,
    MappingView,
    MutableMapping,
    MutableSequence,
    MutableSet,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from einf.diagnostics import ErrorCode, ValidationError
from einf.reduction.callable import CallableReducerBinding
from einf.tensor_types import TensorLike

ReducerCallMode: TypeAlias = Literal[
    "axis_keyword",
    "axis_positional",
    "tensor_only",
]
_UnsafeConfigurationKind: TypeAlias = Literal["tensor_like", "unstable_container"]

_AXES_PARAMETER_NAMES = frozenset({"axis", "axes", "reducer_axes"})
_TENSOR_ARGUMENT = object()
_AXES_ARGUMENT = object()


@dataclass(frozen=True, slots=True)
class _EffectiveCallableContract:
    signature: inspect.Signature
    axes_parameter: inspect.Parameter | None
    axes_are_configured: bool

    @classmethod
    def from_reducer(
        cls,
        reducer: CallableReducerBinding,
        /,
        *,
        axes: tuple[int, ...],
    ) -> "_EffectiveCallableContract":
        signature = reducer.visible_signature
        if _has_only_variadic_parameters(signature):
            raise _callable_signature_error()
        axes_parameters = _axes_parameters(signature)
        if len(axes_parameters) > 1:
            raise _multiple_axes_authorities_error()
        axes_parameter = axes_parameters[0] if axes_parameters else None
        target_axes_parameters = _axes_parameters(reducer.target_signature)
        if len(target_axes_parameters) > 1:
            raise _multiple_axes_authorities_error()
        target_axes_parameter = (
            target_axes_parameters[0] if target_axes_parameters else None
        )
        try:
            configured = reducer.bind_configuration()
        except TypeError as error:
            raise _unsupported_signature_error() from error
        axes_are_configured = bool(
            target_axes_parameter is not None
            and target_axes_parameter.name in configured.arguments
        )
        if (
            target_axes_parameter is not None
            and axes_are_configured
            and not _axes_value_matches(
                configured.arguments[target_axes_parameter.name],
                axes,
            )
        ):
            raise _configured_axes_mismatch_error()
        _validate_captured_configuration(
            bound=configured,
            axes_parameter=target_axes_parameter,
        )

        return cls(
            signature=signature,
            axes_parameter=axes_parameter,
            axes_are_configured=axes_are_configured,
        )

    def resolve_mode(self) -> ReducerCallMode:
        if self.axes_are_configured:
            if self._binds("tensor_only"):
                return "tensor_only"
            raise _unsupported_signature_error()

        axes_parameter = self.axes_parameter
        if axes_parameter is None:
            if self._binds("tensor_only"):
                return "tensor_only"
            raise _unsupported_signature_error()

        if (
            axes_parameter.name == "axis"
            and axes_parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and self._binds("axis_keyword")
        ):
            return "axis_keyword"
        if axes_parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        } and self._binds("axis_positional"):
            return "axis_positional"
        raise _unsupported_signature_error()

    def _binds(self, mode: ReducerCallMode) -> bool:
        match mode:
            case "axis_keyword":
                args = (_TENSOR_ARGUMENT,)
                kwargs = {"axis": _AXES_ARGUMENT}
            case "axis_positional":
                args = (_TENSOR_ARGUMENT, _AXES_ARGUMENT)
                kwargs = {}
            case "tensor_only":
                args = (_TENSOR_ARGUMENT,)
                kwargs = {}
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError:
            return False
        if (
            _bound_parameter_for(
                signature=self.signature,
                bound=bound,
                argument=_TENSOR_ARGUMENT,
            )
            is None
        ):
            return False
        return bool(
            mode == "tensor_only"
            or (
                self.axes_parameter is not None
                and _bound_parameter_for(
                    signature=self.signature,
                    bound=bound,
                    argument=_AXES_ARGUMENT,
                )
                is self.axes_parameter
            )
        )


def resolve_callable_reducer_mode(
    reducer: CallableReducerBinding,
    /,
    *,
    axes: tuple[int, ...],
) -> ReducerCallMode:
    """Resolve one deterministic invocation mode for a callable reducer.

    Parameters
    ----------
    reducer
        Callable reducer whose bound call surface and partial configuration
        define the contract.
    axes
        Canonical reduction axes selected by the compiled operation.

    Returns
    -------
    {"axis_keyword", "axis_positional", "tensor_only"}
        The single invocation mode to use at runtime.

    Raises
    ------
    ValidationError
        If the signature or partial configuration does not establish one safe
        invocation mode.
    """
    contract = _EffectiveCallableContract.from_reducer(reducer, axes=axes)
    return contract.resolve_mode()


def _axes_parameters(
    signature: inspect.Signature,
    /,
) -> tuple[inspect.Parameter, ...]:
    parameters = tuple(signature.parameters.values())
    return tuple(
        parameter
        for index, parameter in enumerate(parameters)
        if parameter.name in _AXES_PARAMETER_NAMES
        and parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        and any(
            preceding.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for preceding in parameters[:index]
        )
    )


def _bound_parameter_for(
    *,
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
    argument: object,
) -> inspect.Parameter | None:
    for parameter_name, value in bound.arguments.items():
        if value is not argument:
            continue
        parameter = signature.parameters[parameter_name]
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            return None
        return parameter
    return None


def _validate_captured_configuration(
    *,
    bound: inspect.BoundArguments,
    axes_parameter: inspect.Parameter | None,
) -> None:
    parameters = tuple(bound.signature.parameters.values())
    axes_index = (
        parameters.index(axes_parameter) if axes_parameter is not None else None
    )
    for name, value in bound.arguments.items():
        parameter = bound.signature.parameters[name]
        if (
            axes_index is not None
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            and parameters.index(parameter) > axes_index
        ):
            continue
        configuration_values: Iterable[object]
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if not isinstance(value, dict):
                raise _unsupported_signature_error()
            configuration_values = value.values()
        else:
            configuration_values = (value,)

        for configuration_value in configuration_values:
            match _unsafe_configuration_kind(configuration_value):
                case "tensor_like":
                    raise _tensor_like_partial_error()
                case "unstable_container":
                    raise _unstable_partial_configuration_error()
                case None:
                    continue


def _unsafe_configuration_kind(
    value: object,
    /,
) -> _UnsafeConfigurationKind | None:
    pending = [value]
    visited_containers: set[int] = set()
    while pending:
        candidate = pending.pop()
        if _is_tensor_like(candidate):
            return "tensor_like"
        if isinstance(
            candidate,
            (
                MappingProxyType,
                MappingView,
                MutableMapping,
                MutableSequence,
                MutableSet,
            ),
        ):
            return "unstable_container"
        if isinstance(candidate, tuple):
            candidate_id = id(candidate)
            if candidate_id in visited_containers:
                continue
            visited_containers.add(candidate_id)
            pending.extend(tuple.__iter__(candidate))
        elif isinstance(candidate, frozenset):
            candidate_id = id(candidate)
            if candidate_id in visited_containers:
                continue
            visited_containers.add(candidate_id)
            pending.extend(frozenset.__iter__(candidate))
    return None


def _is_tensor_like(value: object, /) -> bool:
    try:
        return isinstance(value, TensorLike)
    except Exception:  # noqa: BLE001 - malformed captured values fail closed
        return True


def _axes_value_matches(value: object, axes: tuple[int, ...], /) -> bool:
    return bool(
        type(value) is tuple
        and all(type(axis) is int for axis in value)
        and value == axes
    )


def _has_only_variadic_parameters(signature: inspect.Signature, /) -> bool:
    parameters = tuple(signature.parameters.values())
    return bool(
        parameters
        and all(
            parameter.kind
            in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            for parameter in parameters
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


def _unsupported_signature_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: reducer signature is unsupported",
        help="use (tensor), (tensor, axes), or (tensor, *, axis=...)",
        related=("reduce reducer",),
        data={"operation": "reduce"},
    )


def _tensor_like_partial_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: callable reducer binds a tensor-like value",
        help=(
            "bind tensor-like configuration to an explicit parameter after the axes "
            "parameter, or capture it inside an explicit reducer wrapper"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _unstable_partial_configuration_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: callable reducer binds a mutable or live container "
            "outside a post-axes parameter"
        ),
        help=(
            "bind mutable or live configuration to an explicit parameter after "
            "the axes parameter, use stable tuple/frozenset structure, or capture "
            "it inside an explicit reducer wrapper"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _configured_axes_mismatch_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: callable reducer configures different reduction axes",
        help=(
            "leave reducer axes open or configure the exact tuple of axes "
            "selected by the reduction"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )


def _multiple_axes_authorities_error() -> ValidationError:
    return ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: callable reducer has multiple axes authorities",
        help=(
            "declare exactly one axes parameter named 'axis', 'axes', or 'reducer_axes'"
        ),
        related=("reduce reducer", "callable signature"),
        data={"operation": "reduce"},
    )
