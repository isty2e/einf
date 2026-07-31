from pathlib import Path

from einf.analysis.checkers import (
    CheckerDiagnostic,
    CheckerExecutionPolicy,
    CheckerFailure,
    CheckerResult,
)
from einf.analysis.lsp import (
    LspConfig,
    LspService,
    path_from_uri,
)
from einf.analysis.model import TextPosition, TextSpan

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
    assert config.checker_execution_policy == CheckerExecutionPolicy()


def test_lsp_config_defaults_for_invalid_initialize_options() -> None:
    config = LspConfig.from_initialize_options({"parser": "bogus", "checkers": "ty"})

    assert config.parser == "ast"
    assert config.checkers == ()


def test_lsp_config_normalizes_checker_execution_policy() -> None:
    config = LspConfig.from_initialize_options(
        {
            "checkerTimeoutSeconds": 2.5,
            "checkerCleanupTimeoutSeconds": 0.5,
            "checkerMaxConcurrency": 3,
        }
    )

    assert config.checker_execution_policy == CheckerExecutionPolicy(
        timeout_seconds=2.5,
        cleanup_timeout_seconds=0.5,
        max_concurrency=3,
    )

    invalid = LspConfig.from_initialize_options(
        {
            "checkerTimeoutSeconds": float("inf"),
            "checkerCleanupTimeoutSeconds": 0,
            "checkerMaxConcurrency": 0,
        }
    )

    assert invalid.checker_execution_policy == CheckerExecutionPolicy()

    booleans = LspConfig.from_initialize_options(
        {
            "checkerTimeoutSeconds": True,
            "checkerCleanupTimeoutSeconds": True,
            "checkerMaxConcurrency": True,
        }
    )

    assert booleans.checker_execution_policy == CheckerExecutionPolicy()


def test_path_from_uri_resolves_file_uri(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    expected = target.resolve()

    assert path_from_uri(expected.as_uri()) == expected
    assert path_from_uri("untitled:sample") is None


def test_lsp_service_open_and_change_analyze_in_memory_document(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    service = LspService()

    opened = service.open_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=1,
    )

    assert opened.report is opened.semantic_report
    assert opened.report.diagnostics == ()
    assert opened.report.failures == ()
    assert opened.report.checker_diagnostics == ()
    assert opened.checker_result is None
    assert opened.report.axis_tokens
    assert opened.source == VALID_SOURCE

    changed = service.change_document(
        uri=target.resolve().as_uri(),
        source=INVALID_SOURCE,
        version=2,
    )

    assert len(changed.report.diagnostics) == 1
    assert changed.source == INVALID_SOURCE
    assert changed.report.checker_diagnostics == ()
    assert changed.checker_result is None
    assert service.get_document_state(uri=target.resolve().as_uri()) == changed


def test_lsp_service_analysis_requires_explicit_state_commit(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    uri = target.resolve().as_uri()
    service = LspService()

    state = service.analyze_document(
        uri=uri,
        source=VALID_SOURCE,
        version=1,
    )

    assert service.get_document_state(uri=uri) is None

    service.commit_document_state(state)

    assert service.get_document_state(uri=uri) == state


def test_lsp_service_skips_deep_analysis_for_irrelevant_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    service = LspService()

    def fail_analyze_source(*, source, path, parser_backend):
        _ = source, path, parser_backend
        raise AssertionError("irrelevant source should not reach deep analysis")

    monkeypatch.setattr("einf.analysis.lsp.service.analyze_source", fail_analyze_source)

    state = service.open_document(
        uri=target.resolve().as_uri(),
        source="import math\nmath.sqrt(4)\n",
        version=1,
    )

    assert state.report.diagnostics == ()
    assert state.report.axis_tokens == ()
    assert state.report.failures == ()


def test_lsp_service_prefilter_keeps_supported_einf_alias_calls(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    service = LspService()

    state = service.open_document(
        uri=target.resolve().as_uri(),
        source=(
            "from einf import ax, axes, rearrange as r\n"
            'b = axes("b")[0]\n'
            "r(ax[b], ax[b])\n"
        ),
        version=1,
    )

    assert state.report.axis_tokens
    assert state.report.failures == ()


def test_lsp_service_prefilter_keeps_supported_einf_module_chains(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    service = LspService()

    state = service.open_document(
        uri=target.resolve().as_uri(),
        source=(
            "import einf\n"
            'b = einf.axes("b")[0]\n'
            "einf.operations.rearrange(einf.ax[b], einf.ax[b])\n"
        ),
        version=1,
    )

    assert state.report.axis_tokens
    assert state.report.failures == ()


def test_lsp_document_state_projects_checker_result(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
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
    other_diagnostic = CheckerDiagnostic(
        tool="basedpyright",
        path=(tmp_path / "other.py").resolve(),
        code="reportCallIssue",
        message="other file",
        severity="error",
        span=None,
    )
    checker_failure = CheckerFailure(
        tool="basedpyright",
        kind="execution_error",
        message="checker failed",
    )
    service = LspService()
    semantic_state = service.open_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=3,
    )
    checked_state = semantic_state.with_checker_result(
        CheckerResult(
            diagnostics=(checker_diagnostic, other_diagnostic),
            failures=(checker_failure,),
        )
    )

    assert semantic_state.checker_result is None
    assert semantic_state.report is semantic_state.semantic_report
    assert semantic_state.report.checker_diagnostics == ()
    assert checked_state.checker_result is not None
    assert checked_state.report is not checked_state.semantic_report
    assert checked_state.checker_result.failures == (checker_failure,)
    assert checked_state.report.checker_diagnostics == (checker_diagnostic,)


def test_lsp_service_change_clears_stale_checker_diagnostics(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    checker_diagnostic = CheckerDiagnostic(
        tool="basedpyright",
        path=target.resolve(),
        code="reportCallIssue",
        message="bad call",
        severity="error",
        span=None,
    )
    service = LspService()
    opened = service.open_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=1,
    )
    checked = opened.with_checker_result(
        CheckerResult(diagnostics=(checker_diagnostic,), failures=())
    )
    service.commit_document_state(checked)
    changed = service.change_document(
        uri=target.resolve().as_uri(),
        source=VALID_SOURCE,
        version=2,
    )

    assert checked.report.checker_diagnostics == (checker_diagnostic,)
    assert changed.checker_result is None
    assert changed.report.checker_diagnostics == ()
