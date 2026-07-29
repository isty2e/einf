from collections.abc import Iterable
from dataclasses import dataclass
from math import isqrt

from ..diagnostics import ErrorCode, ValidationError
from .matching import ExprEquation, PartialState

_DEFAULT_EQUATION_SEARCH_WORK_LIMIT = 65_536


def _equation_search_limit_error(*, limit: int, attempted: int) -> ValidationError:
    """Build the stable diagnostic for excessive equation-search work."""
    return ValidationError(
        code=ErrorCode.DIM_SOLVE_TOO_COMPLEX,
        message="dimension solving exceeded the equation search work limit",
        help="bind more axis sizes or simplify coupled axis expressions",
        related=("dim solver",),
        data={
            "complexity_kind": "equation_search_work",
            "limit": limit,
            "attempted": attempted,
        },
    )


@dataclass(slots=True)
class _EquationSearchBudget:
    """Deterministic candidate-search work budget for one equation solve.

    Divisor probes and candidate assignments are distinct work units. A divisor
    that yields a candidate consumes one unit for the probe and another when
    the candidate enters feasibility checking. A zero limit therefore permits
    only outcomes established without either operation.
    """

    limit: int
    used: int = 0

    def consume(self) -> None:
        """Consume one search work unit or fail before exceeding the limit."""
        attempted = self.used + 1
        if attempted > self.limit:
            raise _equation_search_limit_error(
                limit=self.limit,
                attempted=attempted,
            )
        self.used = attempted


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
    work_limit: int = _DEFAULT_EQUATION_SEARCH_WORK_LIMIT

    def __post_init__(self) -> None:
        if isinstance(self.work_limit, bool) or not isinstance(self.work_limit, int):
            raise TypeError("equation search work limit must be an int")
        if self.work_limit < 0:
            raise ValueError("equation search work limit must be non-negative")

    def finalize_state(
        self,
        *,
        state: PartialState,
        solution_limit: int,
    ) -> _FinalizeState:
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
            limit=solution_limit,
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
        budget = _EquationSearchBudget(limit=self.work_limit)

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
        candidate_equations = tuple(
            sorted(
                equations,
                key=lambda equation: (
                    equation.target,
                    equation.canonical_expr.stable_token(),
                ),
            )
        )
        ordered_variables = sorted(
            constrained_unknown_axis_names,
            key=lambda name: (variable_bounds[name], name),
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
                equations=candidate_equations,
                current=current,
                bound=bound,
                budget=budget,
            ):
                budget.consume()
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

    @classmethod
    def _candidate_values_for_variable(
        cls,
        *,
        variable: str,
        equations: tuple[ExprEquation, ...],
        current: dict[str, int],
        bound: int,
        budget: _EquationSearchBudget,
    ) -> Iterable[int]:
        """Generate exact, divisor-derived, or bounded fallback candidates."""
        singleton_constraints: list[set[int]] = []
        for equation in equations:
            if variable not in equation.variables:
                continue
            candidates = cls._single_variable_candidates(
                variable=variable,
                equation=equation,
                current=current,
                bound=bound,
            )
            if candidates is not None:
                singleton_constraints.append(set(candidates))

        if singleton_constraints:
            allowed = set.intersection(*singleton_constraints)
            yield from sorted(allowed)
            return

        for equation in equations:
            if variable not in equation.variables:
                continue
            product_candidates = cls._product_divisor_candidates(
                variable=variable,
                equation=equation,
                current=current,
                bound=bound,
                budget=budget,
            )
            if product_candidates is None:
                continue
            yield from product_candidates
            return

        yield from range(bound + 1)

    @staticmethod
    def _single_variable_candidates(
        *,
        variable: str,
        equation: ExprEquation,
        current: dict[str, int],
        bound: int,
    ) -> tuple[int, ...] | None:
        """Invert supported affine and pure-square single-variable equations."""
        if any(name != variable and name not in current for name in equation.variables):
            return None

        coefficients_by_power: dict[int, int] = {}
        for monomial in equation.canonical_expr.monomials:
            coefficient = monomial.coefficient
            variable_power = 0
            for factor in monomial.factors:
                if factor == variable:
                    variable_power += 1
                else:
                    coefficient *= current[factor]
            if coefficient:
                coefficients_by_power[variable_power] = (
                    coefficients_by_power.get(variable_power, 0) + coefficient
                )

        constant = coefficients_by_power.pop(0, 0)
        residual = equation.target - constant
        if residual < 0:
            return ()
        if not coefficients_by_power:
            return None if residual == 0 else ()

        if len(coefficients_by_power) == 1:
            variable_power, coefficient = next(iter(coefficients_by_power.items()))
            if residual % coefficient:
                return ()
            quotient = residual // coefficient
            if variable_power == 1:
                return (quotient,) if quotient <= bound else ()
            if variable_power == 2:
                candidate = isqrt(quotient)
            else:
                candidate = EquationSolver._exact_nth_root(
                    value=quotient,
                    power=variable_power,
                )
                if candidate is None:
                    return ()

            if candidate > bound:
                return ()
            if variable_power == 2 and candidate * candidate != quotient:
                return ()
            return (candidate,)

        return None

    @classmethod
    def _exact_nth_root(cls, *, value: int, power: int) -> int | None:
        """Return an exact non-negative integer root when one exists."""
        if value in {0, 1}:
            return value

        lower = 1
        upper = 1 << ((value.bit_length() + power - 1) // power)
        while lower <= upper:
            candidate = (lower + upper) // 2
            comparison = cls._compare_power(
                candidate=candidate,
                power=power,
                target=value,
            )
            if comparison == 0:
                return candidate
            if comparison < 0:
                lower = candidate + 1
            else:
                upper = candidate - 1
        return None

    @staticmethod
    def _compare_power(*, candidate: int, power: int, target: int) -> int:
        """Compare a non-negative candidate power with one target."""
        value = 1
        for _ in range(power):
            value *= candidate
            if value > target:
                return 1
        if value < target:
            return -1
        return 0

    @classmethod
    def _product_divisor_candidates(
        cls,
        *,
        variable: str,
        equation: ExprEquation,
        current: dict[str, int],
        bound: int,
        budget: _EquationSearchBudget,
    ) -> Iterable[int] | None:
        """Return lazy divisor candidates for one shifted product constraint."""
        product_monomial = None
        constant = 0
        for monomial in equation.canonical_expr.monomials:
            if variable in monomial.factors:
                if product_monomial is not None:
                    return None
                product_monomial = monomial
                continue

            monomial_value = monomial.coefficient
            for factor in monomial.factors:
                if factor not in current:
                    return None
                monomial_value *= current[factor]
            constant += monomial_value

        if product_monomial is None:
            return None

        residual = equation.target - constant
        if residual < 0:
            return ()
        if residual == 0:
            return None

        variable_power = product_monomial.factors.count(variable)
        known_product = product_monomial.coefficient
        for factor in product_monomial.factors:
            if factor == variable or factor not in current:
                continue
            known_product *= current[factor]

        if known_product == 0 or residual % known_product:
            return ()

        quotient = residual // known_product
        return cls._iter_divisor_candidates(
            quotient=quotient,
            variable_power=variable_power,
            bound=bound,
            budget=budget,
        )

    @classmethod
    def _iter_divisor_candidates(
        cls,
        *,
        quotient: int,
        variable_power: int,
        bound: int,
        budget: _EquationSearchBudget,
    ) -> Iterable[int]:
        """Yield deterministic factor-pair candidates without eager factorization."""
        for lower in range(1, isqrt(quotient) + 1):
            budget.consume()
            if quotient % lower:
                continue

            upper = quotient // lower
            if lower <= bound and cls._power_divides(
                quotient=quotient,
                candidate=lower,
                power=variable_power,
            ):
                yield lower
            if (
                upper != lower
                and upper <= bound
                and cls._power_divides(
                    quotient=quotient,
                    candidate=upper,
                    power=variable_power,
                )
            ):
                yield upper

    @staticmethod
    def _power_divides(*, quotient: int, candidate: int, power: int) -> bool:
        """Return whether one candidate power divides a positive quotient."""
        divisor = 1
        for _ in range(power):
            divisor *= candidate
            if divisor > quotient:
                return False
        return quotient % divisor == 0

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
