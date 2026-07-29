from pathlib import Path

import pytest

lsprotocol = pytest.importorskip("lsprotocol")
pygls = pytest.importorskip("pygls")
_ = lsprotocol, pygls

from benchmarks.profile.lsp_latency import (
    TimingSummary,
    _build_report,
    _default_cases,
    _summarize,
    _to_json,
    _to_markdown,
    _write_text_output,
)


def test_lsp_latency_default_cases_include_token_heavy_source() -> None:
    cases = _default_cases()

    assert any(case.name == "token_heavy_contracts" for case in cases)
    assert any("contract(" in case.source for case in cases)
    assert any(case.name == "irrelevant_python" for case in cases)


def test_lsp_latency_summary_uses_percentile_fields() -> None:
    summary = _summarize([3.0, 1.0, 2.0, 4.0])

    assert summary == TimingSummary(
        count=4,
        median_ms=2.5,
        p25_ms=1.75,
        p75_ms=3.25,
        p95_ms=3.8499999999999996,
        max_ms=4.0,
    )


def test_lsp_latency_report_renders_markdown_and_json(tmp_path: Path) -> None:
    report = _build_report(
        parser="ast",
        repeats=1,
        checkers=(),
        work_dir=tmp_path,
    )

    payload = _to_json(report)
    markdown = _to_markdown(report)

    assert isinstance(payload, dict)
    assert payload["parser"] == "ast"
    assert "LSP Latency Smoke" in markdown
    assert "token_heavy_contracts" in markdown
    assert "Save + checkers" in markdown


def test_lsp_latency_output_writer_creates_parent_directories(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "lsp-latency.md"

    _write_text_output(output, "report\n")

    assert output.read_text(encoding="utf-8") == "report\n"
