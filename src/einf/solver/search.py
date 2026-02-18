from dataclasses import dataclass, field

from ..diagnostics import ErrorCode, ValidationError
from ..signature import Signature
from .equations import DimSolveResult, EquationSolver
from .matching import PartialState, ShapeMatcher


@dataclass(slots=True)
class DimSearch:
    """Mutable solve state for one `solve_dimensions` invocation."""

    signature: Signature
    normalized_shapes: tuple[tuple[int, ...], ...]
    initial_state: PartialState
    matcher: ShapeMatcher
    equation_solver: EquationSolver
    solutions: list[DimSolveResult] = field(default_factory=list)
    seen_solutions: set[
        tuple[tuple[tuple[str, int], ...], tuple[tuple[str, tuple[int, ...]], ...]]
    ] = field(default_factory=set)
    unresolved_pack_ambiguity: bool = False
    unresolved_axis_ambiguity: bool = False

    def _commit_solution(self, result: DimSolveResult) -> None:
        """Record one unique candidate solution."""
        key = (
            tuple(sorted(result.axis_sizes.items())),
            tuple(sorted(result.pack_sizes.items())),
        )
        if key in self.seen_solutions:
            return

        self.seen_solutions.add(key)
        self.solutions.append(result)

    def _search_operands(self, index: int, state: PartialState) -> None:
        """Depth-first operand matching with early-exit on second solution."""
        if len(self.solutions) >= 2:
            return

        if index == self.signature.input_arity:
            finalization = self.equation_solver.finalize_state(state=state)
            if finalization.status == "inconsistent":
                return
            if finalization.status == "unresolved_pack":
                self.unresolved_pack_ambiguity = True
                return
            if finalization.status == "unresolved_axis":
                self.unresolved_axis_ambiguity = True
                return

            for result in finalization.results:
                self._commit_solution(result)
                if len(self.solutions) >= 2:
                    return
            return

        axis_list = self.signature.inputs[index]
        shape = self.normalized_shapes[index]
        for matched_state in self.matcher.match_axis_list(
            axis_list=axis_list,
            shape=shape,
            state=state,
            term_index=0,
            dim_index=0,
        ):
            if len(self.solutions) >= 2:
                return
            self._search_operands(index + 1, matched_state)

    def run(self) -> DimSolveResult:
        """Run search and return one unique solve result or raise."""
        self._search_operands(0, self.initial_state)

        if self.unresolved_pack_ambiguity:
            raise ValidationError(
                code=ErrorCode.AMBIGUOUS_DIMS,
                message=(
                    "ambiguous dims: dim solver cannot uniquely determine "
                    "axis packs from inputs"
                ),
                help="add with_sizes constraints to make dim solve unique",
                related=("dim solver",),
                data={},
            )
        if self.unresolved_axis_ambiguity:
            raise ValidationError(
                code=ErrorCode.AMBIGUOUS_DIMS,
                message=(
                    "ambiguous dims: dim solver cannot uniquely determine "
                    "all axis variables"
                ),
                help="add with_sizes constraints to make dim solve unique",
                related=("dim solver",),
                data={},
            )

        if len(self.solutions) >= 2:
            raise ValidationError(
                code=ErrorCode.AMBIGUOUS_DIMS,
                message="ambiguous dims: dim solver found multiple valid assignments",
                help="add with_sizes constraints to make dim solve unique",
                related=("dim solver",),
                data={},
            )
        if len(self.solutions) == 1:
            return self.solutions[0]

        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: dim solver found no valid assignment",
            help="provide consistent non-negative dimensions and with_sizes bindings",
            related=("dim solver",),
            data={},
        )
