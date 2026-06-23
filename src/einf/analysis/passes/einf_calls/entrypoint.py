from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextSpan
from einf.analysis.parser import ParsedModule
from einf.analysis.passes.call_resolution import build_call_bindings
from einf.analysis.source import SourceText

from .semantics import _parse_call_expression
from .tokens import _axis_roles, _build_missing_rhs_axis_diagnostics

def _parent_map(module: ParsedModule) -> dict[int, int]:
    """Build node-id parent index for one parsed module."""
    parent_by_child: dict[int, int] = {}
    for node in module.nodes:
        for child_id in node.child_ids:
            parent_by_child[child_id] = node.node_id
    return parent_by_child


def _is_method_chain_inner_call(
    *,
    module: ParsedModule,
    parent_by_child: dict[int, int],
    node_id: int,
) -> bool:
    """Return whether one call node is inner call of `call().method(...)` chain."""
    parent_id = parent_by_child.get(node_id)
    if parent_id is None:
        return False
    parent = module.node(parent_id)
    if parent.kind != "Attribute":
        return False

    grandparent_id = parent_by_child.get(parent_id)
    if grandparent_id is None:
        return False
    grandparent = module.node(grandparent_id)
    return grandparent.kind == "Call"


def analyze_einf_calls(
    module: ParsedModule,
) -> tuple[tuple[AnalysisDiagnostic, ...], tuple[AxisToken, ...]]:
    """Analyze `einf` calls in one parsed module."""
    module_source = SourceText(module.source)
    call_bindings = build_call_bindings(
        source=module.source,
        path=module.path,
        source_text=module_source,
    )
    diagnostics: list[AnalysisDiagnostic] = []
    raw_tokens: list[tuple[str, TextSpan, tuple[str, ...]]] = []
    seen_base_call_spans: set[TextSpan] = set()

    parent_by_child = _parent_map(module)
    for parsed_node in module.nodes:
        if parsed_node.kind != "Call" or parsed_node.span is None:
            continue
        if _is_method_chain_inner_call(
            module=module,
            parent_by_child=parent_by_child,
            node_id=parsed_node.node_id,
        ):
            continue

        bindings = call_bindings.get(parsed_node.span)
        if bindings is None:
            continue

        call_result = _parse_call_expression(
            module_source=module_source,
            call_span=parsed_node.span,
            bindings=bindings,
        )
        if call_result is None:
            continue

        diagnostics.extend(call_result.diagnostics)
        call = call_result.base_call
        if call is None:
            continue
        if call.span in seen_base_call_spans:
            continue
        seen_base_call_spans.add(call.span)

        lhs_axis_names = call.lhs.axis_names
        rhs_axis_names = call.rhs.axis_names
        diagnostics.extend(_build_missing_rhs_axis_diagnostics(call=call))

        for occurrence in (*call.lhs.occurrences, *call.rhs.occurrences):
            raw_tokens.append(
                (
                    occurrence.name,
                    occurrence.span,
                    _axis_roles(
                        op_name=call.op_name,
                        side=occurrence.side,
                        axis_name=occurrence.name,
                        lhs_axis_names=lhs_axis_names,
                        rhs_axis_names=rhs_axis_names,
                    ),
                )
            )

    group_by_axis_name: dict[str, int] = {}
    axis_tokens: list[AxisToken] = []
    for axis_name, span, roles in raw_tokens:
        group = group_by_axis_name.get(axis_name)
        if group is None:
            group = len(group_by_axis_name)
            group_by_axis_name[axis_name] = group
        axis_tokens.append(
            AxisToken(
                name=axis_name,
                span=span,
                group=group,
                roles=roles,
            )
        )

    axis_tokens.sort(
        key=lambda token: (token.span.start.line, token.span.start.column, token.name)
    )
    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic.span.start.line if diagnostic.span is not None else -1,
            diagnostic.span.start.column if diagnostic.span is not None else -1,
            diagnostic.code,
        )
    )
    return tuple(diagnostics), tuple(axis_tokens)
