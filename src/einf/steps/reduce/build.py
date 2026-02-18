from dataclasses import dataclass
from threading import RLock
from typing import TypeGuard

from einf.axis import AxisTerms, ScalarAxisTerms
from einf.backend import (
    ArrayNamespace,
    BackendArrayOps,
    BackendProfile,
    derive_namespace_id,
    get_backend_array_ops,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import expand_pack_terms
from einf.reduction.schema import Reducer
from einf.tensor_types import TensorLike

from .runtime import REDUCER_COMPILER, CompiledReducer

_REDUCE_RUNTIME_CACHE_MAX_ENTRIES = 2_048
_REDUCE_NAMESPACE_METHODS = (
    "asarray",
    "sum",
    "prod",
    "mean",
    "max",
    "min",
    "all",
    "any",
)


@dataclass(frozen=True, slots=True)
class _ReduceCompileKey:
    """Cache key for one compiled unary reduce phase."""

    lhs_terms: ScalarAxisTerms
    reduce_axes: AxisTerms
    pack_ranks: tuple[tuple[str, int], ...]
    namespace_id: str
    reducer_kind: str
    reducer_token: str | int


@dataclass(frozen=True, slots=True)
class ReduceAxesResolution:
    """Resolved reduce-axis indices and resulting output axis terms."""

    axes: tuple[int, ...]
    output_terms: ScalarAxisTerms


@dataclass(frozen=True, slots=True)
class ReduceAxesResolver:
    """Resolve reduce axis terms to concrete positions in current axis terms."""

    @classmethod
    def resolve(
        cls,
        *,
        current_terms: ScalarAxisTerms,
        reduce_terms: ScalarAxisTerms,
    ) -> ReduceAxesResolution:
        """Resolve one reduce term list against current terms."""
        reduce_indices = cls._match_indices(
            current_terms=current_terms,
            reduce_terms=reduce_terms,
        )
        return ReduceAxesResolution(
            axes=reduce_indices,
            output_terms=cls._drop_indices(
                current_terms=current_terms,
                reduce_indices=reduce_indices,
            ),
        )

    @staticmethod
    def _match_indices(
        *,
        current_terms: ScalarAxisTerms,
        reduce_terms: ScalarAxisTerms,
    ) -> tuple[int, ...]:
        """Map reduce terms to concrete current-axis indices."""
        selected: list[int] = []
        used: set[int] = set()
        for reduce_term in reduce_terms:
            found: int | None = None
            for index in range(len(current_terms) - 1, -1, -1):
                current_term = current_terms[index]
                if index in used:
                    continue
                if current_term == reduce_term:
                    found = index
                    break
            if found is None:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        "inconsistent dims: reduce_by phase terms "
                        "not found in current tensor axes"
                    ),
                    help="ensure phased reducers partition reduced terms exactly",
                    related=("reduce_by phase mapping",),
                    data={},
                )
            selected.append(found)
            used.add(found)
        return tuple(selected)

    @staticmethod
    def _drop_indices(
        *,
        current_terms: ScalarAxisTerms,
        reduce_indices: tuple[int, ...],
    ) -> ScalarAxisTerms:
        """Drop reduced indices from current scalar terms."""
        removed = set(reduce_indices)
        return ScalarAxisTerms(
            tuple(
                term for index, term in enumerate(current_terms) if index not in removed
            )
        )


@dataclass(frozen=True, slots=True)
class ReduceCompiledProgram:
    """Prevalidated runtime program for one unary reduce primitive."""

    axes: tuple[int, ...]
    compiled_reducer: CompiledReducer
    xp: ArrayNamespace
    backend_ops: BackendArrayOps | None


_REDUCE_RUNTIME_CACHE_ENTRIES: dict[
    _ReduceCompileKey,
    tuple[tuple[int, ...], CompiledReducer, ScalarAxisTerms],
] = {}
_REDUCE_RUNTIME_CACHE_ORDER: list[_ReduceCompileKey] = []
_REDUCE_RUNTIME_CACHE_LOCK = RLock()


def build_reduce_compiled_program(
    *,
    tensor: TensorLike,
    lhs_terms: ScalarAxisTerms,
    expected_output_terms: ScalarAxisTerms,
    axis_sizes: dict[str, int],
    pack_sizes: dict[str, tuple[int, ...]],
    pack_ranks: tuple[tuple[str, int], ...],
    reduce_axes: AxisTerms,
    reducer: Reducer,
    backend_profile: BackendProfile,
) -> ReduceCompiledProgram:
    """Build one unary reduce runtime program from canonical terms and sizes."""
    namespace_candidate = backend_profile.namespace
    if not _has_reduce_namespace_methods(namespace_candidate):
        raise ValidationError(
            code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
            message=(
                "backend dispatch unsupported input: "
                "backend namespace is missing required Array API primitives"
            ),
            help="use tensors backed by a complete Array API namespace for this operation",
            related=("backend dispatch",),
            data={"operation": "reduce"},
        )

    normalized_reduce_axes = AxisTerms.from_spec(reduce_axes)
    xp = namespace_candidate
    backend_ops = get_backend_array_ops(backend_profile.backend_family)
    cache_key = _build_reduce_compile_key(
        lhs_terms=lhs_terms,
        reduce_axes=normalized_reduce_axes,
        pack_ranks=pack_ranks,
        reducer=reducer,
        xp=xp,
    )
    cached_plan = _get_cached_reduce_compiled_program(cache_key)
    if cached_plan is None:
        axes, compiled_reducer, output_terms = _compile_reduce_runtime_phase(
            lhs_terms=lhs_terms,
            reduce_axes=normalized_reduce_axes,
            reducer=reducer,
            pack_sizes=pack_sizes,
            axis_sizes=axis_sizes,
            tensor=tensor,
            xp=xp,
        )
        _put_cached_reduce_compiled_program(
            key=cache_key,
            axes=axes,
            compiled_reducer=compiled_reducer,
            output_terms=output_terms,
        )
    else:
        axes, compiled_reducer, output_terms = cached_plan

    if output_terms != expected_output_terms:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                "inconsistent dims: reduce lowering invariant violated "
                "(reduce output terms must match rhs terms)"
            ),
            help=(
                "ensure lowering emits post-reduce primitive steps "
                "(for example permute/reshape/axis_slice/concat) before runtime"
            ),
            related=("reduce lowering",),
            data={"operation": "reduce"},
        )

    return ReduceCompiledProgram(
        axes=axes,
        compiled_reducer=compiled_reducer,
        xp=xp,
        backend_ops=backend_ops,
    )


def _has_reduce_namespace_methods(namespace: object) -> TypeGuard[ArrayNamespace]:
    """Return whether one namespace exposes required reducer methods."""
    for method_name in _REDUCE_NAMESPACE_METHODS:
        if not callable(getattr(namespace, method_name, None)):
            return False
    return True


def _compile_reduce_runtime_phase(
    *,
    lhs_terms: ScalarAxisTerms,
    reduce_axes: AxisTerms,
    reducer: Reducer,
    pack_sizes: dict[str, tuple[int, ...]],
    axis_sizes: dict[str, int],
    tensor: TensorLike,
    xp: ArrayNamespace,
) -> tuple[tuple[int, ...], CompiledReducer, ScalarAxisTerms]:
    """Compile one unary reduce phase to concrete reducer execution."""
    reduce_terms = expand_pack_terms(
        AxisTerms.from_spec(reduce_axes),
        pack_sizes,
        axis_sizes,
    )
    resolved = ReduceAxesResolver.resolve(
        current_terms=lhs_terms,
        reduce_terms=reduce_terms,
    )
    compiled_reducer = REDUCER_COMPILER.compile(
        reducer=reducer,
        axes=resolved.axes,
        tensor=tensor,
        xp=xp,
    )
    return resolved.axes, compiled_reducer, resolved.output_terms


def _build_reduce_compile_key(
    *,
    lhs_terms: ScalarAxisTerms,
    reduce_axes: AxisTerms,
    pack_ranks: tuple[tuple[str, int], ...],
    reducer: Reducer,
    xp: ArrayNamespace,
) -> _ReduceCompileKey:
    """Build one structural cache key for unary reduce phase compilation."""
    reducer_kind, reducer_token = _reducer_cache_token(reducer)
    return _ReduceCompileKey(
        lhs_terms=lhs_terms,
        reduce_axes=reduce_axes,
        pack_ranks=pack_ranks,
        namespace_id=derive_namespace_id(xp),
        reducer_kind=reducer_kind,
        reducer_token=reducer_token,
    )


def _reducer_cache_token(reducer: Reducer) -> tuple[str, str | int]:
    """Build stable cache token for one reducer."""
    if isinstance(reducer, str):
        return "string", reducer
    return "callable", id(reducer)


def _get_cached_reduce_compiled_program(
    key: _ReduceCompileKey,
    /,
) -> tuple[tuple[int, ...], CompiledReducer, ScalarAxisTerms] | None:
    """Lookup one cached unary reduce compile result."""
    with _REDUCE_RUNTIME_CACHE_LOCK:
        return _REDUCE_RUNTIME_CACHE_ENTRIES.get(key)


def _put_cached_reduce_compiled_program(
    *,
    key: _ReduceCompileKey,
    axes: tuple[int, ...],
    compiled_reducer: CompiledReducer,
    output_terms: ScalarAxisTerms,
) -> None:
    """Store one cached unary reduce compile result with bounded eviction."""
    with _REDUCE_RUNTIME_CACHE_LOCK:
        if key in _REDUCE_RUNTIME_CACHE_ENTRIES:
            _REDUCE_RUNTIME_CACHE_ENTRIES[key] = (
                axes,
                compiled_reducer,
                output_terms,
            )
            return

        _REDUCE_RUNTIME_CACHE_ENTRIES[key] = (
            axes,
            compiled_reducer,
            output_terms,
        )
        _REDUCE_RUNTIME_CACHE_ORDER.append(key)
        while len(_REDUCE_RUNTIME_CACHE_ORDER) > _REDUCE_RUNTIME_CACHE_MAX_ENTRIES:
            oldest = _REDUCE_RUNTIME_CACHE_ORDER.pop(0)
            _REDUCE_RUNTIME_CACHE_ENTRIES.pop(oldest, None)


__all__ = [
    "ReduceAxesResolver",
    "ReduceCompiledProgram",
    "build_reduce_compiled_program",
]
