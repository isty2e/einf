from types import EllipsisType
from typing import Protocol, TypeAlias, runtime_checkable

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


IndexAtom: TypeAlias = int | slice | EllipsisType | None
IndexKey: TypeAlias = IndexAtom | tuple[IndexAtom, ...]


@runtime_checkable
class TensorLike(Protocol):
    """Tensor protocol shared across planning, execution, and backend dispatch."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        ...

    def __getitem__(self, key: IndexKey, /) -> Self: ...
