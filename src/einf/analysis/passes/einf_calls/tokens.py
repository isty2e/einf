from einf.analysis.model import (
    AnalysisDiagnostic,
    AxisCrossSideRelation,
    AxisOccurrenceSide,
    AxisOperationRole,
    AxisStructuralKind,
)

from .diagnostics import _ANALYSIS_AXIS_NOT_IN_INPUT, _diagnostic
from .model import _AxisOccurrence, _CallParseResult

_VALIDATE_RHS_SUBSET_OPS = frozenset({"contract", "einop", "reduce"})
_AXIS_STRUCTURAL_KINDS: tuple[AxisStructuralKind, ...] = ("axis", "pack")


def _axis_relation(
    *,
    call: _CallParseResult,
    occurrence: _AxisOccurrence,
) -> AxisCrossSideRelation:
    """Derive one occurrence's cross-side relation."""
    name = occurrence.name
    kind = occurrence.kind
    in_lhs = name in call.lhs.symbol_names(kind)
    in_rhs = name in call.rhs.symbol_names(kind)
    if in_lhs and in_rhs:
        return "shared"
    return "side_only"


def _axis_role(
    *,
    op_name: str,
    side: AxisOccurrenceSide,
    relation: AxisCrossSideRelation,
) -> AxisOperationRole | None:
    """Derive one occurrence's operation-specific role."""
    if relation == "shared":
        return None
    if side == "lhs":
        if op_name in {"contract", "einop"}:
            return "contracted"
        if op_name == "reduce":
            return "reduced"
        return None
    if op_name == "repeat":
        return "introduced"
    return None


def _build_missing_rhs_symbol_diagnostics(
    *,
    call: _CallParseResult,
) -> tuple[AnalysisDiagnostic, ...]:
    """Build diagnostics for rhs axis symbols missing from lhs."""
    if call.op_name not in _VALIDATE_RHS_SUBSET_OPS:
        return ()

    diagnostics: list[AnalysisDiagnostic] = []
    for kind in _AXIS_STRUCTURAL_KINDS:
        missing_names = sorted(
            call.rhs.symbol_names(kind) - call.lhs.symbol_names(kind)
        )
        symbol_label = "axis" if kind == "axis" else "axis pack"
        for missing_name in missing_names:
            matched_occurrences = [
                occurrence
                for occurrence in call.rhs.occurrences
                if occurrence.kind == kind and occurrence.name == missing_name
            ]
            message = f"{symbol_label} '{missing_name}' appears on rhs but not in lhs"
            if not matched_occurrences:
                diagnostics.append(
                    _diagnostic(
                        code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                        message=message,
                        span=None,
                    )
                )
                continue

            diagnostics.extend(
                _diagnostic(
                    code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                    message=message,
                    span=occurrence.span,
                )
                for occurrence in matched_occurrences
            )
    return tuple(diagnostics)
