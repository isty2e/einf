from pathlib import Path

import pytest

lsprotocol = pytest.importorskip("lsprotocol")
_ = lsprotocol

from einf.analysis.lsp import LspService
from einf.analysis.lsp.hover import build_hover
from einf.analysis.lsp.inlay_hints import build_inlay_hints
from einf.analysis.model import TextPosition
from tests.analysis.conformance_analysis_cases import LSP_CASES


@pytest.mark.parametrize("case", LSP_CASES, ids=lambda case: case.name)
def test_lsp_presentation_conformance_cases(tmp_path: Path, case) -> None:
    target = tmp_path / f"{case.name}.py"
    state = LspService().open_document(
        uri=target.resolve().as_uri(),
        source=case.source,
        version=1,
    )

    if case.expected_inlay_labels:
        assert (
            tuple(
                str(hint.label)
                for hint in build_inlay_hints(
                    axis_tokens=state.semantic_report.axis_tokens,
                    visible_range=None,
                )
            )
            == case.expected_inlay_labels
        )
    if case.hover_line is not None and case.hover_column is not None:
        hover = build_hover(
            axis_tokens=state.semantic_report.axis_tokens,
            position=TextPosition(line=case.hover_line, column=case.hover_column),
        )
        assert hover is not None
        hover_text = str(getattr(hover.contents, "value", hover.contents))
        for token in case.expected_hover_contains:
            assert token in hover_text
