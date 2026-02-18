from collections.abc import Iterable
from dataclasses import dataclass

from .matching import ExprEquation, PartialState


@dataclass(frozen=True, slots=True)
class DimSolveResult:
    """Resolved symbolic sizes for one transform application."""

    axis_sizes: dict[str, int]
    pack_sizes: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _FinalizeState:
    """Finalization result for one full input-segmentation candidate."""

    status: str
    results: tuple[DimSolveResult, ...] = ()


@dataclass(frozen=True, slots=True)
class EquationSolver:
    """Deferred scalar-equation resolver for one solver run."""

    axis_names: set[str]
    pack_names: set[str]
    shapes: tuple[tuple[int, ...], ...]

    def finalize_state(self, *, state: PartialState) -> _FinalizeState:
        """Resolve deferred equations and classify unresolved ambiguities."""
        has_unresolved_pack = bool(self.pack_names - set(state.pack_sizes))

        base_bound = self._base_axis_bound(
            equations=state.equations,
            axis_sizes=state.axis_sizes,
        )
        equation_solutions = self._solve_equations(
            equations=state.equations,
            initial_axis_sizes=state.axis_sizes,
            base_bound=base_bound,
            limit=2,
        )
        if equation_solutions == []:
            return _FinalizeState(status="inconsistent")

        if equation_solutions is None:
            if has_unresolved_pack:
                return _FinalizeState(status="unresolved_pack")
            return _FinalizeState(status="unresolved_axis")

        if has_unresolved_pack:
            return _FinalizeState(status="unresolved_pack")

        return _FinalizeState(
            status="ok",
            results=tuple(
                DimSolveResult(
                    axis_sizes=axis_solution,
                    pack_sizes=state.pack_sizes,
                )
                for axis_solution in equation_solutions
            ),
        )

    def _base_axis_bound(
        self,
        *,
        equations: tuple[ExprEquation, ...],
        axis_sizes: dict[str, int],
    ) -> int:
        """Compute a conservative finite search bound for unconstrained axes."""
        maxima = [8]
        maxima.extend(axis_sizes.values())
        for shape in self.shapes:
            maxima.extend(shape)
        for equation in equations:
            maxima.append(equation.target)
            maxima.append(equation.expr.max_literal())
        return max(maxima)

    def _solve_equations(
        self,
        *,
        equations: tuple[ExprEquation, ...],
        initial_axis_sizes: dict[str, int],
        base_bound: int,
        limit: int,
    ) -> list[dict[str, int]] | None:
        """Solve scalar equations with finite-domain backtracking."""
        axis_sizes = dict(initial_axis_sizes)

        for equation in equations:
            maybe = equation.expr.evaluate(axis_sizes)
            if maybe is not None and maybe != equation.target:
                return []

        unknown_axis_names = self.axis_names - set(axis_sizes)
        constrained_axis_names = set().union(*(eq.variables for eq in equations))
        constrained_unknown_axis_names = unknown_axis_names & constrained_axis_names
        free_axis_names = unknown_axis_names - constrained_axis_names

        if not constrained_unknown_axis_names:
            if free_axis_names:
                return None
            return [axis_sizes]

        variable_bounds = {
            name: self._variable_bound(
                variable=name,
                equations=equations,
                base_bound=base_bound,
            )
            for name in constrained_unknown_axis_names
        }
        ordered_variables = sorted(
            constrained_unknown_axis_names,
            key=lambda name: variable_bounds[name],
        )
        solutions: list[dict[str, int]] = []

        def backtrack(variable_index: int, current: dict[str, int]) -> None:
            if len(solutions) >= limit:
                return

            if variable_index == len(ordered_variables):
                for equation in equations:
                    value = equation.expr.evaluate(current)
                    if value is None or value != equation.target:
                        return
                solutions.append(dict(current))
                return

            variable = ordered_variables[variable_index]
            bound = variable_bounds[variable]
            for candidate in self._candidate_values_for_variable(
                variable=variable,
                equations=equations,
                current=current,
                bound=bound,
            ):
                current[variable] = candidate
                if self._equations_feasible(
                    equations=equations,
                    current=current,
                    variable_bounds=variable_bounds,
                ):
                    backtrack(variable_index + 1, current)
                del current[variable]
                if len(solutions) >= limit:
                    return

        backtrack(0, axis_sizes)
        if not solutions:
            return []
        if free_axis_names:
            return None
        return solutions

    @staticmethod
    def _variable_bound(
        *,
        variable: str,
        equations: tuple[ExprEquation, ...],
        base_bound: int,
    ) -> int:
        """Choose a conservative finite domain bound for one variable."""
        local_targets = [
            equation.target for equation in equations if variable in equation.variables
        ]
        if not local_targets:
            return base_bound
        return max(base_bound, *local_targets)

    @staticmethod
    def _candidate_values_for_variable(
        *,
        variable: str,
        equations: tuple[ExprEquation, ...],
        current: dict[str, int],
        bound: int,
    ) -> Iterable[int]:
        """Generate candidate values using single-unknown equation filtering."""
        singleton_constraints: list[set[int]] = []
        for equation in equations:
            if variable not in equation.variables:
                continue
            unresolved = [
                name
                for name in equation.variables
                if name not in current and name != variable
            ]
            if unresolved:
                continue
            candidates = {
                value
                for value in range(0, bound + 1)
                if equation.expr.evaluate({**current, variable: value})
                == equation.target
            }
            singleton_constraints.append(candidates)

        if singleton_constraints:
            allowed = set.intersection(*singleton_constraints)
            yield from sorted(allowed)
            return

        yield from range(0, bound + 1)

    @staticmethod
    def _equations_feasible(
        *,
        equations: tuple[ExprEquation, ...],
        current: dict[str, int],
        variable_bounds: dict[str, int],
    ) -> bool:
        """Check whether all equations remain feasible under partial assignments."""
        for equation in equations:
            min_value, max_value = equation.expr.evaluate_bounds(
                current=current,
                variable_bounds=variable_bounds,
            )
            if equation.target < min_value or equation.target > max_value:
                return False
            evaluated = equation.expr.evaluate(current)
            if evaluated is not None and evaluated != equation.target:
                return False
        return True
