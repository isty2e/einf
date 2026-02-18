from collections.abc import Iterable
from dataclasses import dataclass

from ..axis import (
    Axis,
    AxisExpr,
    AxisInt,
    AxisPack,
    AxisTerms,
    CanonicalScalarExpr,
    ScalarAxisTermBase,
)


@dataclass(frozen=True, slots=True)
class ExprEquation:
    """One scalar equation extracted from a concrete operand shape."""

    expr: AxisExpr
    canonical_expr: CanonicalScalarExpr
    target: int
    variables: frozenset[str]


@dataclass(frozen=True, slots=True)
class PartialState:
    """Partial matching state while consuming input operands."""

    axis_sizes: dict[str, int]
    pack_sizes: dict[str, tuple[int, ...]]
    equations: tuple[ExprEquation, ...]


@dataclass(frozen=True, slots=True)
class ShapeMatcher:
    """Shape/axis-list matcher for one solver run."""

    def match_axis_list(
        self,
        *,
        axis_list: AxisTerms,
        shape: tuple[int, ...],
        state: PartialState,
        term_index: int,
        dim_index: int,
    ) -> Iterable[PartialState]:
        """Recursively match one axis-term tuple against one concrete shape."""
        if term_index == len(axis_list):
            if dim_index == len(shape):
                yield state
            return

        term = axis_list[term_index]
        if isinstance(term, AxisPack):
            pack_name = term.name
            if pack_name in state.pack_sizes:
                pack_value = state.pack_sizes[pack_name]
                pack_length = len(pack_value)
                if dim_index + pack_length > len(shape):
                    return
                if tuple(shape[dim_index : dim_index + pack_length]) != pack_value:
                    return
                yield from self.match_axis_list(
                    axis_list=axis_list,
                    shape=shape,
                    state=state,
                    term_index=term_index + 1,
                    dim_index=dim_index + pack_length,
                )
                return

            min_rest = self._min_dims_required(
                axis_list=axis_list,
                start=term_index + 1,
                pack_sizes=state.pack_sizes,
            )
            remaining = len(shape) - dim_index
            max_current = remaining - min_rest
            if max_current < 0:
                return

            for count in range(0, max_current + 1):
                pack_value = tuple(shape[dim_index : dim_index + count])
                next_pack_sizes = dict(state.pack_sizes)
                next_pack_sizes[pack_name] = pack_value
                next_state = PartialState(
                    axis_sizes=state.axis_sizes,
                    pack_sizes=next_pack_sizes,
                    equations=state.equations,
                )
                yield from self.match_axis_list(
                    axis_list=axis_list,
                    shape=shape,
                    state=next_state,
                    term_index=term_index + 1,
                    dim_index=dim_index + count,
                )
            return

        if dim_index >= len(shape):
            return
        concrete_dim = shape[dim_index]
        if not isinstance(term, ScalarAxisTermBase):
            return

        constrained = self._constrain_scalar_term(
            term=term,
            concrete_dim=concrete_dim,
            state=state,
        )
        if constrained is None:
            return

        yield from self.match_axis_list(
            axis_list=axis_list,
            shape=shape,
            state=constrained,
            term_index=term_index + 1,
            dim_index=dim_index + 1,
        )

    @staticmethod
    def _min_dims_required(
        *,
        axis_list: AxisTerms,
        start: int,
        pack_sizes: dict[str, tuple[int, ...]],
    ) -> int:
        """Return minimum required concrete rank for remaining terms."""
        required = 0
        for term in axis_list[start:]:
            if isinstance(term, AxisPack):
                if term.name in pack_sizes:
                    required += len(pack_sizes[term.name])
                continue
            required += 1
        return required

    @staticmethod
    def _constrain_scalar_term(
        *,
        term: ScalarAxisTermBase,
        concrete_dim: int,
        state: PartialState,
    ) -> PartialState | None:
        """Apply one scalar-term constraint to a partial state."""
        if isinstance(term, AxisInt):
            if concrete_dim != term.value:
                return None
            return state

        if isinstance(term, Axis):
            axis_name = term.name
            if axis_name in state.axis_sizes:
                if state.axis_sizes[axis_name] != concrete_dim:
                    return None
                return state
            next_axis_sizes = dict(state.axis_sizes)
            next_axis_sizes[axis_name] = concrete_dim
            return PartialState(
                axis_sizes=next_axis_sizes,
                pack_sizes=state.pack_sizes,
                equations=state.equations,
            )

        if isinstance(term, AxisExpr):
            maybe_eval = term.evaluate(state.axis_sizes)
            if maybe_eval is not None:
                if maybe_eval != concrete_dim:
                    return None
                return state
            canonical_expr = CanonicalScalarExpr.from_term(term)
            for equation in state.equations:
                if equation.canonical_expr != canonical_expr:
                    continue
                if equation.target != concrete_dim:
                    return None
                return state
            equation = ExprEquation(
                expr=term,
                canonical_expr=canonical_expr,
                target=concrete_dim,
                variables=frozenset(term.axis_names()),
            )
            return PartialState(
                axis_sizes=state.axis_sizes,
                pack_sizes=state.pack_sizes,
                equations=state.equations + (equation,),
            )

        raise TypeError("unsupported scalar term while constraining shape")
