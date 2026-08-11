from dataclasses import dataclass, field

import opt_einsum.backends as oe_backends
from array_api_compat import array_namespace

from ..diagnostics import ErrorCode, TensorOpError, ValidationError
from .namespace import (
    ArrayNamespaceLike,
    BackendFamily,
    derive_family_key,
    derive_namespace_id,
    infer_backend_family,
)

_STRICT_VIEW_FAMILIES = frozenset(("numpy", "torch"))
_EINSUM_REQUIRED_OPS = frozenset(("contract",))


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


@dataclass(frozen=True, slots=True, eq=False)
class BackendExecutionIdentity:
    """Canonical backend facts that determine runtime specialization.

    Parameters
    ----------
    namespace
        Array namespace object selected for the runtime inputs.
    supports_einsum
        Whether the resolved execution policy permits einsum routes.
    supports_strict_view
        Whether the resolved execution policy permits strict view routes.

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
    supports_einsum: bool
    supports_strict_view: bool
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
                self.supports_einsum,
                self.supports_strict_view,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackendExecutionIdentity):
            return False
        return (
            self.namespace is other.namespace
            and self.namespace_id == other.namespace_id
            and self.backend_family == other.backend_family
            and self.supports_einsum == other.supports_einsum
            and self.supports_strict_view == other.supports_strict_view
        )


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Resolved backend profile for one TensorOp call.

    Parameters
    ----------
    namespace
        Array namespace object selected for the runtime inputs.
    supports_einsum
        Whether the resolved execution policy permits einsum routes.
    supports_strict_view
        Whether the resolved execution policy permits strict view routes.

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
    supports_einsum: bool
    supports_strict_view: bool
    execution_identity: BackendExecutionIdentity = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_identity",
            BackendExecutionIdentity(
                namespace=self.namespace,
                supports_einsum=self.supports_einsum,
                supports_strict_view=self.supports_strict_view,
            ),
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

    def normalize_operation_name(self, op_name: str) -> str:
        """Normalize and validate one operation name for backend checks."""
        if not isinstance(op_name, str):
            raise TypeError("op_name must be a non-empty string")

        normalized_op_name = op_name.strip().lower()
        if not normalized_op_name:
            raise TypeError("op_name must be a non-empty string")

        return normalized_op_name

    def validate_profile(self, *, profile: BackendProfile, op_name: str) -> None:
        """Validate one backend profile against operation-level requirements."""
        normalized_op_name = self.normalize_operation_name(op_name)
        if normalized_op_name in _EINSUM_REQUIRED_OPS and not profile.supports_einsum:
            raise _missing_einsum_extension_error(normalized_op_name)

    def validate_einsum_capability(
        self,
        *,
        profile: BackendProfile,
        op_name: str,
    ) -> None:
        """Validate a selected plan's einsum requirement."""
        normalized_op_name = self.normalize_operation_name(op_name)
        if not profile.supports_einsum:
            raise _missing_einsum_extension_error(normalized_op_name)

    def supports_einsum(self, backend_family: BackendFamily) -> bool:
        """Return whether one backend family supports einsum execution."""
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
        namespace_id: str,
        backend_family: BackendFamily | None,
    ) -> bool:
        """Return whether backend is strict zero-copy view capable."""
        return backend_family in _STRICT_VIEW_FAMILIES


class BackendResolver:
    """Resolve backend profile from runtime tensors and validate policy."""

    def __init__(self, *, policy: BackendPolicy) -> None:
        self.policy = policy

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
        namespace = namespaces[0]
        namespace_id = namespace_ids[0]
        backend_family = infer_backend_family(namespace_id)
        supports_einsum = backend_family is not None and self.policy.supports_einsum(
            backend_family
        )
        supports_view = self.policy.supports_strict_view(
            namespace_id=namespace_id,
            backend_family=backend_family,
        )
        return BackendProfile(
            namespace=namespace,
            supports_einsum=supports_einsum,
            supports_strict_view=supports_view,
        )

    def validate(self, *tensors: object, op_name: str) -> None:
        """Validate backend dispatch and required capabilities for one call."""
        normalized_op_name = self.policy.normalize_operation_name(op_name)
        profile = self.lookup(*tensors, op_name=normalized_op_name)
        self.policy.validate_profile(profile=profile, op_name=normalized_op_name)

    def resolve(self, *tensors: object, op_name: str) -> BackendProfile:
        """Resolve and validate backend profile for one call."""
        normalized_op_name = self.policy.normalize_operation_name(op_name)
        profile = self.lookup(*tensors, op_name=normalized_op_name)
        self.policy.validate_profile(profile=profile, op_name=normalized_op_name)
        return profile


BACKEND_POLICY = BackendPolicy()
BACKEND_RESOLVER = BackendResolver(policy=BACKEND_POLICY)
