from pathlib import Path

import pytest

from einf.analysis.lsp import LspConfig, LspService, encode_semantic_tokens
from einf.analysis.lsp.hover import build_hover
from einf.analysis.lsp.inlay_hints import build_inlay_hints
from einf.analysis.model import TextPosition
from einf.analysis.validator.run import build_parser_backend, run_validation
from tests.conformance_analysis_cases import LSP_CASES, VALIDATOR_CASES


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
    service = LspService(LspConfig())

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
    assert state.checker_fresh is False
    assert len(state.report.axis_tokens) == case.expected_axis_token_count
    assert len(encode_semantic_tokens(state.report.axis_tokens)) == (
        case.expected_semantic_token_int_count
    )
    if case.expected_inlay_labels:
        assert (
            tuple(
                str(hint.label)
                for hint in build_inlay_hints(
                    axis_tokens=state.report.axis_tokens,
                    visible_range=None,
                )
            )
            == case.expected_inlay_labels
        )
    if case.hover_line is not None and case.hover_column is not None:
        hover = build_hover(
            axis_tokens=state.report.axis_tokens,
            position=TextPosition(line=case.hover_line, column=case.hover_column),
        )
        assert hover is not None
        hover_text = str(getattr(hover.contents, "value", hover.contents))
        for token in case.expected_hover_contains:
            assert token in hover_text


def test_lsp_service_checker_refresh_only_on_save(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "checker_refresh.py"
    target.write_text(VALIDATOR_CASES[0].source, encoding="utf-8")

    calls: list[tuple[tuple[Path, ...], Path]] = []

    def fake_run_checker_adapters(*, targets, checker_adapters, project_root):
        calls.append((targets, project_root))
        return (), {}

    monkeypatch.setattr(
        "einf.analysis.lsp.service.run_checker_adapters",
        fake_run_checker_adapters,
    )

    service = LspService(LspConfig(checkers=("basedpyright",)))
    uri = target.resolve().as_uri()

    service.open_document(uri=uri, source=VALIDATOR_CASES[0].source, version=1)
    service.change_document(uri=uri, source=VALIDATOR_CASES[0].source, version=2)
    saved = service.save_document(uri=uri, source=VALIDATOR_CASES[0].source, version=3)

    assert calls == [((target.resolve(),), target.resolve().parent)]
    assert saved.checker_fresh is True
