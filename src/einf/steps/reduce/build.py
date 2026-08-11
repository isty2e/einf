from dataclasses import dataclass
from threading import RLock

from einf.axis import AxisTerms, ScalarAxisTerms
from einf.backend import (
    BackendExecutionIdentity,
)
from einf.diagnostics import ErrorCode, ValidationError
from einf.reduction.callable import CallableReducerBinding
from einf.reduction.schema import CanonicalReducer, ReducerName
from einf.steps.context import expand_pack_terms

from .runtime import (
    REDUCER_COMPILER,
    CompiledReducer,
    ReducerArrayNamespace,
    ReducerRuntimeBinding,
)

_REDUCE_RUNTIME_CACHE_MAX_ENTRIES = 2_048


@dataclass(frozen=True, slots=True)
class _ReduceCompileKey:
    """Cache key for one compiled unary reduce phase."""

    lhs_terms: ScalarAxisTerms
    reduce_axes: AxisTerms
    pack_ranks: tuple[tuple[str, int], ...]
    backend_identity: BackendExecutionIdentity | None
    reducer_kind: str
    reducer_token: str | CallableReducerBinding


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
                    data={"operation": "reduce"},
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
    """Prevalidated shape-dependent facts for one unary reduce primitive.

    Parameters
    ----------
    axes : tuple[int, ...]
        Concrete input axes reduced by the program.
    compiled_reducer : CompiledReducer
        Reducer invocation strategy compiled for ``axes``.
    """

    axes: tuple[int, ...]
    compiled_reducer: CompiledReducer


_REDUCE_RUNTIME_CACHE_ENTRIES: dict[
    _ReduceCompileKey,
    tuple[tuple[int, ...], CompiledReducer, ScalarAxisTerms],
] = {}
_REDUCE_RUNTIME_CACHE_ORDER: list[_ReduceCompileKey] = []
_REDUCE_RUNTIME_CACHE_LOCK = RLock()


def build_reduce_compiled_program(
    *,
    lhs_terms: ScalarAxisTerms,
    expected_output_terms: ScalarAxisTerms,
    axis_sizes: dict[str, int],
    pack_sizes: dict[str, tuple[int, ...]],
    pack_ranks: tuple[tuple[str, int], ...],
    reduce_axes: AxisTerms,
    reducer: CanonicalReducer,
    runtime_binding: ReducerRuntimeBinding,
) -> ReduceCompiledProgram:
    """Compile shape-dependent reduce facts against stable backend capabilities.

    Parameters
    ----------
    lhs_terms : ScalarAxisTerms
        Concrete scalar terms on the input side.
    expected_output_terms : ScalarAxisTerms
        Scalar terms required after reduction.
    axis_sizes : dict[str, int]
        Resolved scalar-axis sizes.
    pack_sizes : dict[str, tuple[int, ...]]
        Resolved variadic pack expansions.
    pack_ranks : tuple[tuple[str, int], ...]
        Structural pack ranks used by the compile cache.
    reduce_axes : AxisTerms
        Canonical terms selected for reduction.
    reducer : CanonicalReducer
        Canonical named or callable reducer.
    runtime_binding : ReducerRuntimeBinding
        Validated backend identity and reducer capability binding.

    Returns
    -------
    ReduceCompiledProgram
        Concrete axes and reducer invocation strategy.

    Raises
    ------
    ValidationError
        Lowering terms or reducer configuration violate the reduce contract.
    """
    normalized_reduce_axes = AxisTerms.from_spec(reduce_axes)
    cache_key = _build_reduce_compile_key(
        lhs_terms=lhs_terms,
        reduce_axes=normalized_reduce_axes,
        pack_ranks=pack_ranks,
        reducer=reducer,
        backend_identity=runtime_binding.profile.execution_identity,
    )
    cached_plan = _get_cached_reduce_compiled_program(cache_key)
    if cached_plan is None:
        axes, compiled_reducer, output_terms = _compile_reduce_runtime_phase(
            lhs_terms=lhs_terms,
            reduce_axes=normalized_reduce_axes,
            reducer=reducer,
            pack_sizes=pack_sizes,
            axis_sizes=axis_sizes,
            xp=runtime_binding.context.xp,
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
    )


def _compile_reduce_runtime_phase(
    *,
    lhs_terms: ScalarAxisTerms,
    reduce_axes: AxisTerms,
    reducer: CanonicalReducer,
    pack_sizes: dict[str, tuple[int, ...]],
    axis_sizes: dict[str, int],
    xp: ReducerArrayNamespace,
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
        xp=xp,
    )
    return resolved.axes, compiled_reducer, resolved.output_terms


def _build_reduce_compile_key(
    *,
    lhs_terms: ScalarAxisTerms,
    reduce_axes: AxisTerms,
    pack_ranks: tuple[tuple[str, int], ...],
    reducer: CanonicalReducer,
    backend_identity: BackendExecutionIdentity,
) -> _ReduceCompileKey:
    """Build one structural cache key for unary reduce phase compilation."""
    reducer_kind, reducer_token = _reducer_cache_token(reducer)
    return _ReduceCompileKey(
        lhs_terms=lhs_terms,
        reduce_axes=reduce_axes,
        pack_ranks=pack_ranks,
        backend_identity=(
            backend_identity if isinstance(reducer, ReducerName) else None
        ),
        reducer_kind=reducer_kind,
        reducer_token=reducer_token,
    )


def _reducer_cache_token(
    reducer: CanonicalReducer,
) -> tuple[str, str | CallableReducerBinding]:
    """Build stable cache token for one reducer."""
    if isinstance(reducer, ReducerName):
        return "string", reducer.value
    return "callable", reducer


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
