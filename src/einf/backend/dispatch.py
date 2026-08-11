from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import opt_einsum.backends as oe_backends
from array_api_compat import array_namespace

from ..diagnostics import (
    ErrorCode,
    ExecutionError,
    TensorOpError,
    ValidationError,
)
from ..tensor_types import TensorLike
from .namespace import (
    ArrayNamespaceLike,
    BackendFamily,
    derive_family_key,
    derive_namespace_id,
    infer_backend_family,
)

_STRICT_VIEW_FAMILIES = frozenset(("numpy", "torch"))


def _missing_einsum_extension_error(operation: str) -> ValidationError:
    return ValidationError(
        code=ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING,
        message=(
            "backend required extension missing: "
            f"{operation} requires an einsum-capable backend extension"
        ),
        help=(
            "use an einsum-capable backend family "
            "or avoid operations requiring contraction lowering"
        ),
        related=("backend capability",),
        data={"operation": operation},
    )


def _normalize_operation_name(op_name: str) -> str:
    if not isinstance(op_name, str):
        raise TypeError("op_name must be a non-empty string")

    normalized_op_name = op_name.strip().lower()
    if not normalized_op_name:
        raise TypeError("op_name must be a non-empty string")
    return normalized_op_name


@dataclass(frozen=True, slots=True, eq=False)
class BackendExecutionIdentity:
    """Canonical backend identity for runtime specialization.

    Parameters
    ----------
    namespace
        Array namespace object selected for the runtime inputs.
    Attributes
    ----------
    namespace_id
        Stable identifier derived from ``namespace``.
    backend_family
        Built-in backend family derived from ``namespace_id``, if known.

    Raises
    ------
    TypeError
        ``namespace`` has no canonical namespace identifier.
    """

    namespace: ArrayNamespaceLike
    namespace_id: str = field(init=False)
    backend_family: BackendFamily | None = field(init=False)

    def __post_init__(self) -> None:
        namespace_id = derive_namespace_id(self.namespace)
        object.__setattr__(self, "namespace_id", namespace_id)
        object.__setattr__(
            self,
            "backend_family",
            infer_backend_family(namespace_id),
        )

    def __hash__(self) -> int:
        return hash(
            (
                id(self.namespace),
                self.namespace_id,
                self.backend_family,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackendExecutionIdentity):
            return False
        return (
            self.namespace is other.namespace
            and self.namespace_id == other.namespace_id
            and self.backend_family == other.backend_family
        )


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Resolved backend namespace profile for one TensorOp call.

    Parameters
    ----------
    namespace
        Array namespace object selected for the runtime inputs.
    Attributes
    ----------
    namespace_id
        Stable identifier derived from ``namespace``.
    backend_family
        Built-in backend family derived from ``namespace_id``, if known.
    execution_identity
        Hashable canonical facts used by runtime specialization caches.

    Raises
    ------
    TypeError
        ``namespace`` has no canonical namespace identifier.
    """

    namespace: ArrayNamespaceLike
    execution_identity: BackendExecutionIdentity = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_identity",
            BackendExecutionIdentity(namespace=self.namespace),
        )

    @property
    def namespace_id(self) -> str:
        """Return the namespace-derived backend identifier.

        Returns
        -------
        str
            Canonical identifier derived from ``namespace``.
        """
        return self.execution_identity.namespace_id

    @property
    def backend_family(self) -> BackendFamily | None:
        """Return the namespace-derived built-in backend family.

        Returns
        -------
        BackendFamily or None
            Registered family for ``namespace_id``, if one exists.
        """
        return self.execution_identity.backend_family


class BackendPolicy:
    """Backend capability policy for operations and selected plans."""

    def validate_einsum_capability(
        self,
        *,
        profile: BackendProfile,
        op_name: str,
    ) -> None:
        """Validate a selected plan's einsum requirement.

        Parameters
        ----------
        profile
            Resolved namespace profile for the runtime operands.
        op_name
            Public operation that selected the einsum plan.

        Raises
        ------
        ValidationError
            No namespace or ``opt_einsum`` route is available.
        ExecutionError
            Namespace capability lookup fails unexpectedly.
        """
        normalized_op_name = _normalize_operation_name(op_name)
        if self.resolve_namespace_einsum(profile) is not None:
            return
        if self.supports_opt_einsum(profile.backend_family):
            return
        raise _missing_einsum_extension_error(normalized_op_name)

    def resolve_namespace_einsum(
        self,
        profile: BackendProfile,
        /,
    ) -> Callable[..., TensorLike] | None:
        """Resolve one namespace-provided einsum primitive.

        Parameters
        ----------
        profile
            Resolved namespace profile to inspect.

        Returns
        -------
        Callable or None
            Namespace ``einsum`` callable, or ``None`` when absent.

        Raises
        ------
        ExecutionError
            Namespace attribute lookup fails unexpectedly.
        """
        try:
            candidate = getattr(profile.namespace, "einsum", None)
        except TensorOpError:
            raise
        except AttributeError:
            return None
        except Exception as error:
            raise ExecutionError(
                code=ErrorCode.BACKEND_EXECUTION_FAILED,
                message=(
                    "backend execution failed: einsum capability lookup failed: "
                    f"{error}"
                ),
                help="ensure the array namespace exposes stable operation attributes",
                related=("einsum capability",),
                data={"operation": "einsum"},
            ) from error
        if not callable(candidate):
            return None
        return cast(Callable[..., TensorLike], candidate)

    def supports_opt_einsum(self, backend_family: BackendFamily | None) -> bool:
        """Return whether ``opt_einsum`` admits one backend family.

        Parameters
        ----------
        backend_family
            Canonical backend family, if one was resolved.

        Returns
        -------
        bool
            Whether ``opt_einsum`` reports an einsum implementation.
        """
        if backend_family is None:
            return False
        try:
            return bool(oe_backends.has_einsum(backend_family))
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False

    def supports_strict_view(
        self,
        *,
        backend_family: BackendFamily | None,
    ) -> bool:
        """Return whether a backend family supports strict views.

        Parameters
        ----------
        backend_family
            Canonical backend family, if one was resolved.

        Returns
        -------
        bool
            Whether the family has a supported storage-alias proof.
        """
        return backend_family in _STRICT_VIEW_FAMILIES


class BackendResolver:
    """Resolve canonical backend profiles from runtime tensors."""

    def lookup(self, *tensors: object, op_name: str) -> BackendProfile:
        """Lookup backend profile from runtime input tensors."""
        if not tensors:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message="backend dispatch unsupported input: no tensor inputs provided",
                help="pass one or more tensors to resolve backend dispatch",
                related=("backend dispatch",),
                data={"operation": op_name},
            )

        namespace_ids: list[str] = []
        family_keys: list[str] = []
        namespaces: list[ArrayNamespaceLike] = []
        for tensor in tensors:
            try:
                namespace = array_namespace(tensor)
                namespace_id = derive_namespace_id(namespace)
            except TensorOpError:
                raise
            except Exception as exc:
                raise ValidationError(
                    code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                    message=(
                        "backend dispatch unsupported input: "
                        f"tensor type {type(tensor).__name__!r} is not Array API compatible"
                    ),
                    help=(
                        "use tensors from a supported Array API compatible "
                        "backend family"
                    ),
                    related=("backend dispatch",),
                    data={"operation": op_name},
                ) from exc

            namespaces.append(namespace)
            namespace_ids.append(namespace_id)
            family_keys.append(derive_family_key(namespace_id))

        unique_families = tuple(dict.fromkeys(family_keys))
        if len(unique_families) != 1:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_MIXED_FAMILY,
                message=(
                    "backend dispatch mixed family: "
                    "all inputs must share one backend family"
                ),
                help="do not mix tensor families in one TensorOp call",
                related=("backend dispatch",),
                data={
                    "operation": op_name,
                    "families": len(unique_families),
                    "namespace_ids": ",".join(tuple(dict.fromkeys(namespace_ids))),
                },
            )
        if infer_backend_family(namespace_ids[0]) is None and any(
            namespace is not namespaces[0] for namespace in namespaces
        ):
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message=(
                    "backend dispatch unsupported input: "
                    "inputs must resolve to one namespace object for "
                    "unknown backend families"
                ),
                help=(
                    "pass tensors from the same backend namespace instance "
                    "in one TensorOp call"
                ),
                related=("backend dispatch",),
                data={"operation": op_name},
            )
        return BackendProfile(namespace=namespaces[0])

    def resolve(self, *tensors: object, op_name: str) -> BackendProfile:
        """Resolve one backend namespace profile for a call.

        Parameters
        ----------
        *tensors
            Runtime operands whose namespaces must be compatible.
        op_name
            Operation name used in validation diagnostics.

        Returns
        -------
        BackendProfile
            Canonical namespace profile for the operands.

        Raises
        ------
        TypeError
            ``op_name`` is not a non-empty string.
        ValidationError
            The operands do not provide one compatible namespace.
        """
        normalized_op_name = _normalize_operation_name(op_name)
        return self.lookup(*tensors, op_name=normalized_op_name)


BACKEND_POLICY = BackendPolicy()
BACKEND_RESOLVER = BackendResolver()
