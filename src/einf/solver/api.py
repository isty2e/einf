from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from ..diagnostics import ErrorCode, ValidationError
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


@dataclass(frozen=True, slots=True)
class _NormalizedSolveRequest:
    """Canonical normalized inputs shared by feasibility and unique solving."""

    cache_key: _SolveCacheKey
    initial_state: PartialState
    axis_names: set[str]
    pack_names: set[str]


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

    def contains(self, key: _SolveCacheKey) -> bool:
        """Return whether one unique solve result is cached without cloning it."""
        with self._lock:
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

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


def _normalize_solve_request(
    signature: Signature,
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    explicit_sizes: dict[str, int] | None = None,
) -> _NormalizedSolveRequest:
    """Normalize one dimension-solving request to canonical search inputs."""
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
    return _NormalizedSolveRequest(
        cache_key=_SolveCacheKey(
            signature=signature,
            input_shapes=normalized_shapes,
            explicit_sizes=tuple(sorted(initial_axis_sizes.items())),
        ),
        initial_state=PartialState(
            axis_sizes=initial_axis_sizes,
            pack_sizes={},
            equations=(),
        ),
        axis_names=axis_names,
        pack_names=pack_names,
    )


def _build_dim_search(request: _NormalizedSolveRequest, /) -> DimSearch:
    """Build mutable search state from one canonical solve request."""
    cache_key = request.cache_key
    return DimSearch(
        signature=cache_key.signature,
        normalized_shapes=cache_key.input_shapes,
        initial_state=request.initial_state,
        matcher=ShapeMatcher(),
        equation_solver=EquationSolver(
            axis_names=request.axis_names,
            pack_names=request.pack_names,
            shapes=cache_key.input_shapes,
        ),
    )


def validate_dimensions(
    signature: Signature,
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    explicit_sizes: dict[str, int] | None = None,
) -> None:
    """Validate that at least one dimension assignment is consistent.

    Unlike `solve_dimensions`, this accepts ambiguous assignments when execution
    does not require their concrete values.

    Parameters
    ----------
    signature
        Transform signature to validate.
    input_shapes
        Concrete input tensor shapes in input-arity order.
    explicit_sizes
        Optional pre-bound scalar axis sizes, equivalent to `.with_sizes(...)`.

    Raises
    ------
    ValidationError
        If no consistent assignment exists.
    ValueError
        If input contracts are malformed.
    TypeError
        If argument types are invalid.
    """
    request = _normalize_solve_request(
        signature,
        input_shapes,
        explicit_sizes=explicit_sizes,
    )
    if _SOLVE_CACHE.contains(request.cache_key):
        return
    if _build_dim_search(request).has_feasible_assignment():
        return
    raise ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message="inconsistent dims: dim solver found no valid assignment",
        help="provide consistent non-negative dimensions and with_sizes bindings",
        related=("dim solver",),
        data={},
    )


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
    request = _normalize_solve_request(
        signature,
        input_shapes,
        explicit_sizes=explicit_sizes,
    )
    cache_key = request.cache_key
    cached = _SOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    result = _build_dim_search(request).run()
    _SOLVE_CACHE.put(key=cache_key, result=result)
    return _clone_solve_result(result)


__all__ = [
    "solve_dimensions",
    "validate_dimensions",
]
