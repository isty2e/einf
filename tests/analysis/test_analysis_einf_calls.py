from pathlib import Path

from einf.analysis.engine import AnalysisOutput, analyze_module
from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextPosition, TextSpan
from einf.analysis.parser import AstParserBackend

_TEST_IMPORT_PREFIX = (
    "from einf import contract, einop, rearrange, reduce, repeat, view\n"
)


def _shift_position(position: TextPosition, *, line_delta: int) -> TextPosition:
    return TextPosition(line=position.line + line_delta, column=position.column)


def _shift_span(span: TextSpan | None, *, line_delta: int) -> TextSpan | None:
    if span is None:
        return None
    return TextSpan(
        start=_shift_position(span.start, line_delta=line_delta),
        end=_shift_position(span.end, line_delta=line_delta),
    )


def _shift_diagnostic(
    diagnostic: AnalysisDiagnostic,
    *,
    line_delta: int,
) -> AnalysisDiagnostic:
    return AnalysisDiagnostic(
        code=diagnostic.code,
        message=diagnostic.message,
        severity=diagnostic.severity,
        span=_shift_span(diagnostic.span, line_delta=line_delta),
    )


def _shift_axis_token(token: AxisToken, *, line_delta: int) -> AxisToken:
    return AxisToken(
        name=token.name,
        span=TextSpan(
            start=_shift_position(token.span.start, line_delta=line_delta),
            end=_shift_position(token.span.end, line_delta=line_delta),
        ),
        group=token.group,
        roles=token.roles,
    )


def _with_default_imports(source: str) -> str:
    return _TEST_IMPORT_PREFIX + source


def _strip_import_prefix(output: AnalysisOutput) -> AnalysisOutput:
    return AnalysisOutput(
        module=output.module,
        diagnostics=tuple(
            _shift_diagnostic(diagnostic, line_delta=-1)
            for diagnostic in output.diagnostics
        ),
        axis_tokens=tuple(
            _shift_axis_token(token, line_delta=-1) for token in output.axis_tokens
        ),
    )


def _analyze(source: str) -> AnalysisOutput:
    return _strip_import_prefix(
        analyze_module(
            source=_with_default_imports(source),
            path=Path("sample.py"),
            parser_backend=AstParserBackend(),
        )
    )


def _analyze_exact(source: str) -> AnalysisOutput:
    return analyze_module(
        source=source,
        path=Path("sample.py"),
        parser_backend=AstParserBackend(),
    )


def _slice_span_text(source: str, *, line: int, start: int, end: int) -> str:
    lines = source.splitlines(keepends=True)
    target_line = lines[line - 1]
    return target_line[start:end]


def test_contract_marks_contracted_axes() -> None:
    result = _analyze("contract((ax[b, n, d], ax[d, j]), ax[b, n, j])\n")
    contracted_d_tokens = [
        token
        for token in result.axis_tokens
        if token.name == "d" and "contracted" in token.roles
    ]
    assert len(contracted_d_tokens) == 2
    assert result.diagnostics == ()


def test_reduce_reports_rhs_axis_missing_from_lhs() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b, z])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_NOT_IN_INPUT"
    assert "z" in diagnostic.message
    assert diagnostic.severity == "error"
    assert diagnostic.span is not None


def test_same_axis_name_keeps_same_group() -> None:
    source = (
        "rearrange(ax[b, n, d], ax[b, d, n])\nrearrange(ax[b, d, n], ax[b, n, d])\n"
    )
    result = _analyze(source)
    b_groups = {token.group for token in result.axis_tokens if token.name == "b"}
    assert len(b_groups) == 1


def test_invalid_side_expression_reports_span() -> None:
    result = _analyze("rearrange(lhs, ax[b, n])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None
    assert diagnostic.span.start.line == 1


def test_with_sizes_negative_maps_to_validation_code() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(n=-1)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "inconsistent_dims"
    assert "negative with_sizes binding" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_on_unsupported_op_reports_error() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).reduce_by('sum')\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "does not support" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_phase_is_parsed_without_duplicate_base_diagnostics() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by((ax[n, d], 'sum'))\n")
    assert result.diagnostics == ()


def test_reduce_by_callable_symbol_is_allowed() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by(my_reducer)\n")
    assert result.diagnostics == ()


def test_with_sizes_rejects_positional_arguments() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(1)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert "only accepts keyword bindings" in diagnostic.message
    assert diagnostic.span is not None


def test_with_sizes_rejects_non_literal_values() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(n=size)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert "integer literals" in diagnostic.message
    assert diagnostic.span is not None


def test_with_sizes_rejects_unpack_kwargs() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(**sizes)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert "**kwargs" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_keyword_arguments() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by(reducer='sum')\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "positional arguments" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_empty_call() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by()\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "at least one reducer argument" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_malformed_phase_tuple() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by((ax[n, d],))\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "(ax[...], reducer)" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_lambda_reducer() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by(lambda x: x)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "string literals or simple callable symbols" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_extra_phases_after_literal_reducer() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by('sum', (ax[n], 'prod'))\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "no extra phase arguments" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_non_phase_tail_after_phase_first() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by((ax[n], 'sum'), 'prod')\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_REDUCE_BY_ERROR"
    assert "phase must be `(ax[...], reducer)`" in diagnostic.message
    assert diagnostic.span is not None


def test_missing_rhs_axis_reports_all_rhs_occurrences() -> None:
    result = _analyze("reduce(ax[b, n], ax[z, z])\n")
    assert len(result.diagnostics) == 2
    assert all(
        diagnostic.code == "ANALYSIS_AXIS_NOT_IN_INPUT"
        for diagnostic in result.diagnostics
    )
    spans = [diagnostic.span for diagnostic in result.diagnostics]
    assert all(span is not None for span in spans)
    assert spans[0] != spans[1]


def test_chain_analysis_does_not_duplicate_tokens_from_inner_calls() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).with_sizes(n=4)\n")
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    d_tokens = [token for token in result.axis_tokens if token.name == "d"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 1
    assert len(d_tokens) == 1
    assert result.diagnostics == ()


def test_axis_expr_rejects_unsupported_operator() -> None:
    result = _analyze("rearrange(ax[b, n - 1], ax[b, n])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "only '+' and '*'" in diagnostic.message
    assert diagnostic.span is not None


def test_axis_pack_requires_named_symbol() -> None:
    result = _analyze("rearrange(ax[*foo()], ax[b])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "axis pack must be a named symbol" in diagnostic.message
    assert diagnostic.span is not None


def test_call_shape_error_for_missing_rhs_argument() -> None:
    result = _analyze("rearrange(ax[b, n])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_CALL_SHAPE_ERROR"
    assert "missing a required argument" in diagnostic.message
    assert diagnostic.span is not None


def test_non_einf_calls_are_ignored() -> None:
    result = _analyze("foo(ax[b, n], ax[b, n]).with_sizes(n=3)\n")
    assert result.diagnostics == ()
    assert result.axis_tokens == ()


def test_imported_entrypoint_alias_is_recognized() -> None:
    source = "from einf import rearrange as r\nr(ax[b, n], ax[b, n])\n"
    result = _analyze_exact(source)
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_imported_module_alias_is_recognized() -> None:
    source = "import einf as ef\nef.rearrange(ax[b, n], ax[b, n])\n"
    result = _analyze_exact(source)
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_assignment_alias_to_imported_entrypoint_is_recognized() -> None:
    source = (
        "from einf.operations import rearrange\nr = rearrange\nr(ax[b, n], ax[b, n])\n"
    )
    result = _analyze_exact(source)
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_arbitrary_method_named_like_einf_op_is_ignored() -> None:
    result = _analyze_exact("obj.rearrange(ax[b, n], ax[b, n])\n")
    assert result.diagnostics == ()
    assert result.axis_tokens == ()


def test_shadowed_imported_name_is_not_treated_as_einf_call() -> None:
    source = (
        "from einf import rearrange\nrearrange = other\nrearrange(ax[b, n], ax[b, n])\n"
    )
    result = _analyze_exact(source)
    assert result.diagnostics == ()
    assert result.axis_tokens == ()


def test_with_sizes_accepts_unary_plus_literal() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(n=+3)\n")
    assert result.diagnostics == ()


def test_with_sizes_rejects_bool_literal() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(n=True)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert "integer literals" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_attribute_callable_is_allowed() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by(mod.reducer)\n")
    assert result.diagnostics == ()


def test_contract_non_atomic_axis_reports_validation_code() -> None:
    result = _analyze("contract((ax[b, (n + d)], ax[d, j]), ax[b, j])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "contract_non_atomic_axis"
    assert diagnostic.span is not None


def test_unknown_method_chain_keeps_base_call_analysis() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).foo(1)\n")
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 2


def test_unknown_intermediate_method_stops_with_sizes_chain_analysis() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).foo(1).with_sizes(n=size)\n")
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 2


def test_side_spec_rejects_empty_tuple_with_span_text() -> None:
    source = "rearrange((), ax[b, n])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None
    assert (
        _slice_span_text(
            source,
            line=diagnostic.span.start.line,
            start=diagnostic.span.start.column,
            end=diagnostic.span.end.column,
        )
        == "()"
    )


def test_side_spec_rejects_mixed_tuple_entry_with_precise_span() -> None:
    source = "rearrange((ax[b], rhs), ax[b])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None
    assert (
        _slice_span_text(
            source,
            line=diagnostic.span.start.line,
            start=diagnostic.span.start.column,
            end=diagnostic.span.end.column,
        )
        == "rhs"
    )


def test_axis_term_rejects_string_literal() -> None:
    source = "rearrange(ax['b', n], ax[n, n])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert diagnostic.span is not None
    assert (
        _slice_span_text(
            source,
            line=diagnostic.span.start.line,
            start=diagnostic.span.start.column,
            end=diagnostic.span.end.column,
        )
        == "'b'"
    )


def test_axis_term_rejects_float_literal() -> None:
    result = _analyze("rearrange(ax[1.5], ax[1])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert diagnostic.span is not None


def test_axis_term_rejects_bool_literal() -> None:
    result = _analyze("rearrange(ax[True], ax[1])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert diagnostic.span is not None


def test_axis_term_rejects_negative_integer_literal() -> None:
    result = _analyze("rearrange(ax[-1], ax[1])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "only unary '+'" in diagnostic.message
    assert diagnostic.span is not None


def test_axis_pack_rejects_attribute_symbol() -> None:
    result = _analyze("rearrange(ax[*pkg.T], ax[b])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "axis pack must be a named symbol" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_rejects_non_ax_phase_axis_spec() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by((n, 'sum'))\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert "ax[...]" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_accepts_two_phase_specifications() -> None:
    result = _analyze(
        "reduce(ax[b, n, d], ax[b]).reduce_by((ax[n], 'sum'), (ax[d], 'prod'))\n"
    )
    assert result.diagnostics == ()


def test_reduce_by_then_unknown_method_keeps_valid_base_analysis() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).reduce_by('sum').foo(1)\n")
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    d_tokens = [token for token in result.axis_tokens if token.name == "d"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 1
    assert len(d_tokens) == 1


def test_unknown_intermediate_method_stops_reduce_by_chain_analysis() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b]).foo(1).reduce_by('sum')\n")
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    d_tokens = [token for token in result.axis_tokens if token.name == "d"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 1
    assert len(d_tokens) == 1


def test_with_sizes_rejects_unary_plus_bool_literal_basic() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n]).with_sizes(n=+True)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert diagnostic.span is not None


def test_diagnostics_are_sorted_by_column_on_same_line() -> None:
    result = _analyze("reduce(ax[b, n], ax[z]).with_sizes(n=size)\n")
    assert len(result.diagnostics) >= 2
    starts: list[tuple[int, int]] = []
    for diagnostic in result.diagnostics:
        span = diagnostic.span
        if span is None:
            continue
        starts.append((span.start.line, span.start.column))
    assert starts == sorted(starts)


def test_repeat_marks_rhs_only_axis_as_introduced() -> None:
    result = _analyze("repeat(ax[b], ax[b, n])\n")
    introduced_tokens = [
        token
        for token in result.axis_tokens
        if token.name == "n" and "introduced" in token.roles
    ]
    assert len(introduced_tokens) == 1
    assert "rhs" in introduced_tokens[0].roles


def test_reduce_marks_lhs_only_axes_as_reduced() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b])\n")
    reduced_tokens = [
        token
        for token in result.axis_tokens
        if token.name in {"n", "d"} and "reduced" in token.roles
    ]
    assert len(reduced_tokens) == 2
    assert all("lhs" in token.roles for token in reduced_tokens)


def test_base_call_rejects_extra_positional_arguments() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n], ax[d])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_CALL_SHAPE_ERROR"
    assert "too many positional arguments" in diagnostic.message
    assert diagnostic.span is not None


def test_base_call_rejects_unexpected_keyword_argument() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n], extra=1)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_CALL_SHAPE_ERROR"
    assert "unexpected keyword argument" in diagnostic.message
    assert diagnostic.span is not None


def test_base_call_accepts_lhs_rhs_keywords() -> None:
    result = _analyze("rearrange(lhs=ax[b, n], rhs=ax[b, n])\n")
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_base_call_accepts_mixed_positional_and_rhs_keyword() -> None:
    result = _analyze("rearrange(ax[b, n], rhs=ax[b, n])\n")
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_base_call_rejects_starred_positional_arguments() -> None:
    result = _analyze("rearrange(*(ax[b, n], ax[b, n]))\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_CALL_SHAPE_ERROR"
    assert "starred positional arguments" in diagnostic.message
    assert diagnostic.span is not None


def test_base_call_rejects_unpack_kwargs() -> None:
    result = _analyze("rearrange(**spec)\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_CALL_SHAPE_ERROR"
    assert "**kwargs" in diagnostic.message
    assert diagnostic.span is not None


def test_attribute_ax_subscript_is_accepted() -> None:
    result = _analyze("rearrange(mod.ax[b, n], mod.ax[b, n])\n")
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 2


def test_axis_pack_only_specs_do_not_emit_named_axis_tokens() -> None:
    result = _analyze("rearrange(ax[*T], ax[*T])\n")
    assert result.diagnostics == ()
    assert result.axis_tokens == ()


def test_rearrange_does_not_emit_rhs_subset_diagnostic() -> None:
    result = _analyze("rearrange(ax[b, n], ax[b, n, z])\n")
    assert all(
        diagnostic.code != "ANALYSIS_AXIS_NOT_IN_INPUT"
        for diagnostic in result.diagnostics
    )


def test_axis_expr_rejects_unary_minus() -> None:
    result = _analyze("rearrange(ax[b, -n], ax[b, n])\n")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "only unary '+'" in diagnostic.message
    assert diagnostic.span is not None


def test_chain_with_invalid_base_reports_single_diagnostic() -> None:
    result = _analyze("rearrange(lhs, ax[b, n]).with_sizes(n=3)\n")
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "ANALYSIS_SIDE_SPEC_ERROR"


def test_missing_rhs_axis_span_points_to_axis_token_text() -> None:
    source = "reduce(ax[b, n, d], ax[b, z])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.span is not None

    lines = source.splitlines(keepends=True)
    line = lines[diagnostic.span.start.line - 1]
    token_text = line[diagnostic.span.start.column : diagnostic.span.end.column]
    assert token_text == "z"


def test_missing_rhs_axis_span_survives_unicode_prefix_on_same_line() -> None:
    source = "é = 1; reduce(ax[b, n, d], ax[b, z])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.span is not None

    lines = source.splitlines(keepends=True)
    line = lines[diagnostic.span.start.line - 1]
    token_text = line[diagnostic.span.start.column : diagnostic.span.end.column]
    assert token_text == "z"


def test_axis_tokens_survive_unicode_prefix_on_same_line() -> None:
    source = "é = 1; rearrange(ax[b, n], ax[b, n])\n"
    result = _analyze(source)
    assert result.diagnostics == ()
    assert len([token for token in result.axis_tokens if token.name == "b"]) == 2
    assert len([token for token in result.axis_tokens if token.name == "n"]) == 2


def test_diagnostics_are_sorted_by_source_position() -> None:
    source = "rearrange(lhs1, ax[b, n])\nrearrange(lhs2, ax[b, n])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 2
    first, second = result.diagnostics
    assert first.span is not None
    assert second.span is not None
    assert first.span.start.line == 1
    assert second.span.start.line == 2


def test_empty_tuple_side_reports_side_spec_error_with_exact_span() -> None:
    source = "rearrange((), ax[b])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None
    snippet = _slice_span_text(
        source,
        line=diagnostic.span.start.line,
        start=diagnostic.span.start.column,
        end=diagnostic.span.end.column,
    )
    assert snippet == "()"


def test_mixed_tuple_side_entry_reports_invalid_member_span() -> None:
    source = "rearrange((ax[b], 5), ax[b])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None
    snippet = _slice_span_text(
        source,
        line=diagnostic.span.start.line,
        start=diagnostic.span.start.column,
        end=diagnostic.span.end.column,
    )
    assert snippet == "5"


def test_axis_string_literal_is_rejected() -> None:
    source = "rearrange(ax[b, 'n'], ax[b, n])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "non-negative integers" in diagnostic.message
    assert diagnostic.span is not None


def test_axis_float_literal_is_rejected() -> None:
    source = "rearrange(ax[b, 1.25], ax[b, n])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "non-negative integers" in diagnostic.message
    assert diagnostic.span is not None


def test_axis_pack_attribute_expression_is_rejected() -> None:
    source = "rearrange(ax[*pkg.T, b], ax[b])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_AXIS_TERM_ERROR"
    assert "axis pack must be a named symbol" in diagnostic.message
    assert diagnostic.span is not None


def test_reduce_by_phase_with_non_ax_entry_reports_side_spec_error() -> None:
    source = "reduce(ax[b, n], ax[b]).reduce_by((lhs, 'sum'))\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_SIDE_SPEC_ERROR"
    assert diagnostic.span is not None


def test_reduce_by_two_phase_plan_is_accepted() -> None:
    source = "reduce(ax[b, n, d], ax[b]).reduce_by((ax[n], 'sum'), (ax[d], 'sum'))\n"
    result = _analyze(source)
    assert result.diagnostics == ()


def test_reduce_by_then_unknown_method_chain_keeps_base_token_analysis() -> None:
    source = "reduce(ax[b, n, d], ax[b]).reduce_by('sum').foo(1)\n"
    result = _analyze(source)
    assert result.diagnostics == ()
    b_tokens = [token for token in result.axis_tokens if token.name == "b"]
    n_tokens = [token for token in result.axis_tokens if token.name == "n"]
    d_tokens = [token for token in result.axis_tokens if token.name == "d"]
    assert len(b_tokens) == 2
    assert len(n_tokens) == 1
    assert len(d_tokens) == 1


def test_with_sizes_rejects_unary_plus_bool_literal() -> None:
    source = "rearrange(ax[b, n], ax[b, n]).with_sizes(n=+True)\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ANALYSIS_WITH_SIZES_ERROR"
    assert "integer literals" in diagnostic.message
    assert diagnostic.span is not None


def test_diagnostics_same_line_are_sorted_by_column() -> None:
    source = "rearrange(lhs1, ax[b]); rearrange(lhs2, ax[b])\n"
    result = _analyze(source)
    assert len(result.diagnostics) == 2
    first, second = result.diagnostics
    assert first.span is not None
    assert second.span is not None
    assert first.span.start.line == second.span.start.line == 1
    assert first.span.start.column < second.span.start.column


def test_repeat_marks_introduced_axis_role() -> None:
    result = _analyze("repeat(ax[b, n], ax[b, n, z])\n")
    z_tokens = [token for token in result.axis_tokens if token.name == "z"]
    assert len(z_tokens) == 1
    assert "introduced" in z_tokens[0].roles
    assert "rhs" in z_tokens[0].roles


def test_reduce_marks_reduced_axis_roles() -> None:
    result = _analyze("reduce(ax[b, n, d], ax[b])\n")
    reduced_tokens = [
        token
        for token in result.axis_tokens
        if token.name in {"n", "d"} and "reduced" in token.roles
    ]
    assert len(reduced_tokens) == 2
