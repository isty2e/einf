from pathlib import Path

from einf.analysis.checkers import CheckerDiagnostic, CheckerFailure
from einf.analysis.lsp import (
    LspConfig,
    LspService,
    encode_semantic_tokens,
    path_from_uri,
)
from einf.analysis.model import AxisToken, TextPosition, TextSpan

VALID_SOURCE = """from einf import ax, axes, rearrange\nb = axes(\"b\")[0]\nrearrange(ax[b], ax[b])\n"""
INVALID_SOURCE = """from einf import ax, axes, reduce\nb, n, z = axes(\"b\", \"n\", \"z\")\nreduce(ax[b, n], ax[b, z])\n"""


def test_lsp_config_from_initialize_options_deduplicates_and_filters() -> None:
    config = LspConfig.from_initialize_options(
        {
            "parser": "libcst",
            "checkers": ["basedpyright", "ty", "basedpyright", "bogus"],
        }
    )

    assert config.parser == "libcst"
    assert config.checkers == ("basedpyright", "ty")


def test_lsp_config_defaults_for_invalid_initialize_options() -> None:
    config = LspConfig.from_initialize_options({"parser": "bogus", "checkers": "ty"})

    assert config.parser == "ast"
    assert config.checkers == ()


def test_path_from_uri_resolves_file_uri(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    expected = target.resolve()

    assert path_from_uri(expected.as_uri()) == expected
    assert path_from_uri("untitled:sample") is None


def test_lsp_service_open_and_change_analyze_in_memory_document(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    service = LspService(LspConfig())

    opened = service.open_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=1,
    )

    assert opened.report.diagnostics == ()
    assert opened.report.failures == ()
    assert opened.report.checker_diagnostics == ()
    assert opened.checker_failures == ()
    assert opened.checker_fresh is False
    assert opened.report.axis_tokens

    changed = service.change_document(
        uri=target.resolve().as_uri(),
        source=INVALID_SOURCE,
        version=2,
    )

    assert len(changed.report.diagnostics) == 1
    assert changed.report.checker_diagnostics == ()
    assert changed.checker_failures == ()
    assert changed.checker_fresh is False
    assert service.get_document_state(uri=target.resolve().as_uri()) == changed


def test_lsp_service_save_refreshes_checkers(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text(VALID_SOURCE, encoding="utf-8")

    checker_diagnostic = CheckerDiagnostic(
        tool="basedpyright",
        path=target.resolve(),
        code="reportCallIssue",
        message="bad call",
        severity="error",
        span=TextSpan(
            start=TextPosition(line=2, column=0),
            end=TextPosition(line=2, column=3),
        ),
    )
    checker_failure = CheckerFailure(
        tool="basedpyright",
        kind="execution_error",
        message="checker failed",
    )

    def fake_run_checker_adapters(*, targets, checker_adapters, project_root):
        assert targets == (target.resolve(),)
        assert project_root == target.resolve().parent
        return (checker_failure,), {target.resolve(): (checker_diagnostic,)}

    monkeypatch.setattr(
        "einf.analysis.lsp.service.run_checker_adapters",
        fake_run_checker_adapters,
    )

    service = LspService(LspConfig(checkers=("basedpyright",)))
    state = service.save_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=3,
    )

    assert state.checker_fresh is True
    assert state.checker_failures == (checker_failure,)
    assert state.report.checker_diagnostics == (checker_diagnostic,)


def test_encode_semantic_tokens_uses_fixed_palette_and_modifiers() -> None:
    tokens = (
        AxisToken(
            name="b",
            span=TextSpan(
                start=TextPosition(line=2, column=4),
                end=TextPosition(line=2, column=5),
            ),
            group=0,
            roles=("introduced",),
        ),
        AxisToken(
            name="n",
            span=TextSpan(
                start=TextPosition(line=2, column=7),
                end=TextPosition(line=2, column=8),
            ),
            group=9,
            roles=("contracted", "pack"),
        ),
    )

    assert encode_semantic_tokens(tokens) == [
        1,
        4,
        1,
        0,
        1,
        0,
        3,
        1,
        1,
        (1 << 2) | (1 << 3),
    ]
