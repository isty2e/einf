import sys
from types import EllipsisType
from typing import Literal, Protocol, TypeAlias, runtime_checkable

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


IndexAtom: TypeAlias = int | slice | EllipsisType | None
IndexKey: TypeAlias = IndexAtom | tuple[IndexAtom, ...]
TrustedTensorFamily: TypeAlias = Literal["numpy", "torch"]

_TRUSTED_TENSOR_TYPE_NAMES: tuple[tuple[TrustedTensorFamily, str], ...] = (
    ("numpy", "ndarray"),
    ("torch", "Tensor"),
)


@runtime_checkable
class TensorLike(Protocol):
    """Tensor protocol shared across planning, execution, and backend dispatch."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        ...

    def __getitem__(self, key: IndexKey, /) -> Self: ...


def trusted_tensor_family(
    tensor_type: type[object],
    /,
) -> TrustedTensorFamily | None:
    """Return the built-in family whose canonical tensor type is provided.

    Parameters
    ----------
    tensor_type
        Runtime tensor type to compare by object identity.

    Returns
    -------
    {"numpy", "torch"} or None
        Canonical built-in family, or ``None`` for an untrusted type.
    """
    for family, type_name in _TRUSTED_TENSOR_TYPE_NAMES:
        runtime_module = sys.modules.get(family)
        if runtime_module is None:
            continue
        try:
            canonical_type = vars(runtime_module).get(type_name)
        except TypeError:
            continue
        if tensor_type is canonical_type:
            return family
    return None


def is_trusted_tensor_type(tensor_type: type[object], /) -> bool:
    """Return whether one tensor type has a stable built-in shape contract."""
    return trusted_tensor_family(tensor_type) is not None
