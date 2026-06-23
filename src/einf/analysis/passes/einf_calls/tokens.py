from einf.analysis.model import AnalysisDiagnostic

from .diagnostics import _ANALYSIS_AXIS_NOT_IN_INPUT, _diagnostic
from .model import _CallParseResult, _CallSide

_VALIDATE_RHS_SUBSET_OPS = frozenset({"contract", "einop", "reduce"})

def _axis_roles(
    *,
    op_name: str,
    side: _CallSide,
    axis_name: str,
    lhs_axis_names: frozenset[str],
    rhs_axis_names: frozenset[str],
) -> tuple[str, ...]:
    """Derive semantic token roles for one axis occurrence."""
    roles: list[str] = [side]
    in_lhs = axis_name in lhs_axis_names
    in_rhs = axis_name in rhs_axis_names

    if in_lhs and in_rhs:
        roles.append("shared")
    if op_name in {"contract", "einop"} and in_lhs and not in_rhs:
        roles.append("contracted")
    if op_name == "reduce" and in_lhs and not in_rhs:
        roles.append("reduced")
    if op_name == "repeat" and in_rhs and not in_lhs:
        roles.append("introduced")
    return tuple(roles)


def _build_missing_rhs_axis_diagnostics(
    *,
    call: _CallParseResult,
) -> tuple[AnalysisDiagnostic, ...]:
    """Build diagnostics for rhs axes missing from lhs."""
    if call.op_name not in _VALIDATE_RHS_SUBSET_OPS:
        return ()

    missing_axis_names = sorted(call.rhs.axis_names - call.lhs.axis_names)
    if not missing_axis_names:
        return ()

    diagnostics: list[AnalysisDiagnostic] = []
    for missing_axis_name in missing_axis_names:
        matched_occurrences = [
            occurrence
            for occurrence in call.rhs.occurrences
            if occurrence.name == missing_axis_name
        ]
        if not matched_occurrences:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                    message=(
                        f"axis '{missing_axis_name}' appears on rhs but not in lhs"
                    ),
                    span=None,
                )
            )
            continue

        for occurrence in matched_occurrences:
            diagnostics.append(
                _diagnostic(
                    code=_ANALYSIS_AXIS_NOT_IN_INPUT,
                    message=(
                        f"axis '{missing_axis_name}' appears on rhs but not in lhs"
                    ),
                    span=occurrence.span,
                )
            )
    return tuple(diagnostics)
