from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from ..signature import Signature
from .equations import DimSolveResult, EquationSolver
from .matching import PartialState, ShapeMatcher
from .normalize import normalize_explicit_sizes, normalize_shape
from .search import DimSearch

_SOLVE_CACHE_MAX_ENTRIES = 4_096


@dataclass(frozen=True, slots=True)
class _SolveCacheKey:
    """Structural cache key for one canonical dim-solver invocation."""

    signature: Signature
    input_shapes: tuple[tuple[int, ...], ...]
    explicit_sizes: tuple[tuple[str, int], ...]


class _SolveCache:
    """Thread-safe bounded LRU cache for `solve_dimensions`."""

    def __init__(self, *, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[_SolveCacheKey, DimSolveResult] = OrderedDict()
        self._lock = RLock()

    def get(self, key: _SolveCacheKey) -> DimSolveResult | None:
        """Return a defensive copy of one cached solve result."""
        with self._lock:
            cached = self._entries.get(key)
            if cached is None:
                return None

            self._entries.move_to_end(key)
            return _clone_solve_result(cached)

    def put(self, *, key: _SolveCacheKey, result: DimSolveResult) -> None:
        """Insert one solve result into cache with LRU eviction."""
        cloned = _clone_solve_result(result)
        with self._lock:
            self._entries[key] = cloned
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


def _clone_solve_result(result: DimSolveResult) -> DimSolveResult:
    """Clone mutable mappings in one dim-solve result."""
    return DimSolveResult(
        axis_sizes=dict(result.axis_sizes),
        pack_sizes=dict(result.pack_sizes),
    )


_SOLVE_CACHE = _SolveCache(max_entries=_SOLVE_CACHE_MAX_ENTRIES)


def solve_dimensions(
    signature: Signature,
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    explicit_sizes: dict[str, int] | None = None,
) -> DimSolveResult:
    """Solve symbolic axis variables and packs for concrete input shapes.

    Parameters
    ----------
    signature
        Transform signature to solve.
    input_shapes
        Concrete input tensor shapes in input-arity order.
    explicit_sizes
        Optional pre-bound scalar axis sizes, equivalent to `.with_sizes(...)`.

    Returns
    -------
    DimSolveResult
        Unique resolved scalar and pack assignments.

    Raises
    ------
    ValidationError
        If no valid assignment exists or assignment is ambiguous.
    ValueError
        If input contracts are malformed.
    TypeError
        If argument types are invalid.
    """
    if len(input_shapes) != signature.input_arity:
        raise ValueError(
            f"expected {signature.input_arity} input shapes, got {len(input_shapes)}"
        )

    normalized_shapes = tuple(normalize_shape(shape) for shape in input_shapes)
    axis_names = signature.axis_names()
    pack_names = signature.pack_names()

    initial_axis_sizes = normalize_explicit_sizes(
        explicit_sizes=explicit_sizes,
        axis_names=axis_names,
        pack_names=pack_names,
    )
    cache_key = _SolveCacheKey(
        signature=signature,
        input_shapes=normalized_shapes,
        explicit_sizes=tuple(sorted(initial_axis_sizes.items())),
    )
    cached = _SOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    search = DimSearch(
        signature=signature,
        normalized_shapes=normalized_shapes,
        initial_state=PartialState(
            axis_sizes=initial_axis_sizes,
            pack_sizes={},
            equations=(),
        ),
        matcher=ShapeMatcher(),
        equation_solver=EquationSolver(
            axis_names=axis_names,
            pack_names=pack_names,
            shapes=normalized_shapes,
        ),
    )
    result = search.run()
    _SOLVE_CACHE.put(key=cache_key, result=result)
    return _clone_solve_result(result)


__all__ = [
    "solve_dimensions",
]
