from dataclasses import dataclass

from ..axis import AxisSide, AxisTermBase, AxisTerms
from ..diagnostics import DiagnosticValue, ErrorCode, ValidationError
from .schema import Reducer, ReducerPhase, ReducerPlan


def infer_unary_reduced_terms(
    *,
    lhs: AxisTerms,
    rhs: AxisTerms,
    op_name: str,
    include_operation_data: bool = True,
) -> AxisTerms:
    """Infer unary reduced terms and reject partial-multiplicity ambiguities."""
    lhs_counts = lhs.term_counts()
    rhs_counts = rhs.term_counts()
    ambiguous_terms = tuple(
        dsl
        for _, dsl in sorted(
            (
                term.stable_token(),
                term.to_dsl(),
            )
            for term, lhs_count in lhs_counts.items()
            if lhs_count > 1 and 0 < rhs_counts.get(term, 0) < lhs_count
        )
    )
    if ambiguous_terms:
        data: dict[str, DiagnosticValue] = {"terms": ",".join(ambiguous_terms)}
        if include_operation_data:
            data["operation"] = op_name
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message=(
                "ambiguous dims: repeated axis appears on both lhs and rhs "
                "with partial multiplicity"
            ),
            help=(
                "rename duplicated axes (for example b1/b2), or keep/reduce all "
                "occurrences of each repeated axis symbol"
            ),
            related=(f"{op_name} multiplicity mapping",),
            data=data,
        )
    return lhs - rhs


@dataclass(frozen=True, slots=True)
class ReducerPlanParser:
    """Normalize and validate reducer declarations for unary reduce signatures."""

    lhs: AxisSide
    rhs: AxisSide

    def parse(
        self,
        *,
        reducer: Reducer | tuple[AxisTerms, Reducer],
        phases: tuple[tuple[AxisTerms, Reducer], ...],
    ) -> ReducerPlan:
        """Parse `TensorOp.reduce_by(...)` arguments to canonical `ReducerPlan`."""
        if isinstance(reducer, tuple):
            return self.parse_ordered(phases=(reducer, *phases))
        if phases:
            raise TypeError(
                "phase reducer arguments must start with a phase tuple "
                "(ax[...], reducer)"
            )
        return self.parse_single(reducer=reducer)

    def parse_single(self, *, reducer: Reducer) -> ReducerPlan:
        """Normalize one reducer over all reduced terms."""
        if not isinstance(reducer, str) and not callable(reducer):
            raise TypeError("reducer must be a string or callable")
        return (ReducerPhase(axes=self.reduced_terms(), reducer=reducer),)

    def parse_ordered(
        self,
        *,
        phases: tuple[tuple[AxisTerms, Reducer], ...],
    ) -> ReducerPlan:
        """Normalize ordered reducer phases with strict partition checks."""
        if not phases:
            raise TypeError("reduce_by phase form requires at least one phase")

        reduced_terms = self.reduced_terms()
        reduced_counts = reduced_terms.term_counts()

        normalized_phases: list[ReducerPhase] = []
        covered_terms: list[AxisTermBase] = []
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, tuple) or len(phase) != 2:
                raise TypeError("each reducer phase must be (ax[...], reducer)")

            phase_axes, phase_reducer = phase
            if not isinstance(phase_axes, tuple):
                raise TypeError("phase axes must be ax[...]")
            try:
                normalized_phase_axes = AxisTerms.from_spec(phase_axes)
            except (TypeError, ValueError) as exc:
                raise TypeError("phase axes must be ax[...]") from exc

            if not normalized_phase_axes:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message="inconsistent dims: reduce_by phase axis-list cannot be empty",
                    help="provide one or more reduced axis terms for each phase",
                    related=("reduce_by phase axis-list",),
                    data={"phase_index": phase_index},
                )

            if not isinstance(phase_reducer, str) and not callable(phase_reducer):
                raise TypeError("phase reducer must be a string or callable")
            normalized_phases.append(
                ReducerPhase(
                    axes=normalized_phase_axes,
                    reducer=phase_reducer,
                )
            )
            covered_terms.extend(normalized_phase_axes)

        covered_counts = AxisTerms(tuple(covered_terms)).term_counts()
        if covered_counts != reduced_counts:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: reduce_by phase axes must exactly partition "
                    "reduced terms from lhs to rhs"
                ),
                help=(
                    "ensure phase axes cover every reduced term exactly once, "
                    "with no duplicates and no extras"
                ),
                related=("reduce_by phase partition",),
                data={},
            )

        return tuple(normalized_phases)

    def reduced_terms(self) -> AxisTerms:
        """Return reduced terms for unary reduce-by form as `lhs - rhs`."""
        if len(self.lhs) != 1 or len(self.rhs) != 1:
            raise ValidationError(
                code=ErrorCode.OP_ARITY_MISMATCH,
                message="reduce_by is only defined for unary reduce signatures in v0.1",
                help="use unary reduce signatures when configuring reducers",
                related=("reduce_by contract",),
                data={"lhs_arity": len(self.lhs), "rhs_arity": len(self.rhs)},
            )
        return infer_unary_reduced_terms(
            lhs=self.lhs[0],
            rhs=self.rhs[0],
            op_name="reduce",
            include_operation_data=False,
        )


__all__ = ["infer_unary_reduced_terms", "ReducerPlanParser"]
