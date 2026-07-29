from pathlib import Path

import pytest
from lsprotocol import types as lsp

from einf.analysis.lsp import LspService, encode_semantic_tokens
from einf.analysis.lsp.position_codec import LspPositionCodec
from einf.analysis.validator.run import build_parser_backend, run_validation
from tests.analysis.conformance_analysis_cases import LSP_CASES, VALIDATOR_CASES


@pytest.mark.parametrize("case", VALIDATOR_CASES, ids=lambda case: case.name)
def test_validator_conformance_cases(tmp_path: Path, case) -> None:
    target = tmp_path / f"{case.name}.py"
    target.write_text(case.source, encoding="utf-8")

    report = run_validation(
        targets=(target,),
        parser_backend=build_parser_backend(case.parser),
    )

    assert report.exit_code() == case.expected_exit_code
    assert report.checker_failures == ()
    assert len(report.files) == 1

    file_report = report.files[0]
    assert tuple(diagnostic.code for diagnostic in file_report.diagnostics) == (
        case.expected_diagnostic_codes
    )
    assert tuple(failure.kind for failure in file_report.failures) == (
        case.expected_failure_kinds
    )
    assert len(file_report.axis_tokens) == case.expected_axis_token_count


@pytest.mark.parametrize("case", LSP_CASES, ids=lambda case: case.name)
def test_lsp_service_conformance_open_cases(tmp_path: Path, case) -> None:
    target = tmp_path / f"{case.name}.py"
    service = LspService()

    state = service.open_document(
        uri=target.resolve().as_uri(),
        source=case.source,
        version=1,
    )

    assert tuple(diagnostic.code for diagnostic in state.report.diagnostics) == (
        case.expected_diagnostic_codes
    )
    assert tuple(failure.kind for failure in state.report.failures) == (
        case.expected_failure_kinds
    )
    assert state.checker_result is None
    assert len(state.report.axis_tokens) == case.expected_axis_token_count
    assert len(
        encode_semantic_tokens(
            state.report.axis_tokens,
            position_codec=LspPositionCodec(
                lines=state.source_lines,
                encoding=lsp.PositionEncodingKind.Utf16,
            ),
        )
    ) == (case.expected_semantic_token_int_count)
