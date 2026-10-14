import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("numpy")

import benchmarks.audit.expression_layout as expression_layout_module
from benchmarks.audit.expression_layout import LayoutAuditReport
from benchmarks.audit.expression_layout import main as expression_layout_main
from benchmarks.compare import einf_einops_einx as fixed_module
from benchmarks.compare import einf_einops_einx_dynamic as dynamic_module
from benchmarks.compare import expression_parity as expression_parity_module
from benchmarks.compare.einf_einops_einx import main as fixed_main
from benchmarks.compare.einf_einops_einx_dynamic import main as dynamic_main
from benchmarks.compare.expression_parity import main as expression_parity_main
from benchmarks.profile.overhead_breakdown import main as overhead_breakdown_main
from benchmarks.profile.warm_calltree import main as warm_calltree_main


def test_expression_layout_cli_publishes_receipt_and_prints_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = LayoutAuditReport(
        title="# Layout",
        configuration=[],
        methodology=[],
        strategies=(),
        notes=[],
    )
    monkeypatch.setattr(
        expression_layout_module,
        "_find_gap_case",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        expression_layout_module,
        "_audit_case",
        lambda **kwargs: report,
    )
    receipt = tmp_path / "layout.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["expression_layout.py", "--receipt", str(receipt)],
    )

    assert expression_layout_main() == 0

    output = capsys.readouterr()
    assert output.out.startswith("# Layout\n")
    assert output.err == f"Wrote receipt: {receipt}\n"
    assert json.loads(receipt.read_text(encoding="utf-8"))["title"] == "# Layout"


@pytest.mark.parametrize(
    ("program", "entrypoint"),
    [
        ("einf_einops_einx.py", fixed_main),
        ("einf_einops_einx_dynamic.py", dynamic_main),
        ("expression_parity.py", expression_parity_main),
        ("expression_layout.py", expression_layout_main),
        ("overhead_breakdown.py", overhead_breakdown_main),
        ("warm_calltree.py", warm_calltree_main),
    ],
)
def test_receipt_producers_share_one_output_contract(
    program: str,
    entrypoint: Callable[[], int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [program, "--help"])

    with pytest.raises(SystemExit) as raised:
        entrypoint()

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--receipt" in help_text
    assert "--raw-output" not in help_text
    assert "--output" not in help_text


@pytest.mark.parametrize(
    ("program", "module"),
    [
        ("einf_einops_einx.py", fixed_module),
        ("einf_einops_einx_dynamic.py", dynamic_module),
        ("expression_parity.py", expression_parity_module),
    ],
)
def test_compare_source_expectation_is_checked_before_benchmarking(
    program: str,
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [program, "--expect-einf-source-root", str(tmp_path)],
    )

    def reject_source(checkout_root: Path) -> None:
        assert checkout_root == tmp_path
        raise RuntimeError("wrong source")

    monkeypatch.setattr(module, "require_einf_source_root", reject_source)

    with pytest.raises(RuntimeError, match="wrong source"):
        module.main()


def _source_metadata() -> dict[str, str | bool | None]:
    return {
        "kind": "git_checkout",
        "distribution_version": "0.2.0",
        "git_revision": "a" * 40,
        "git_dirty": False,
        "content_sha256": "1" * 64,
    }


@pytest.mark.parametrize(
    ("program", "module", "case_builder", "render_method"),
    [
        (
            "einf_einops_einx.py",
            fixed_module,
            "_build_case_specs",
            "render_fixed",
        ),
        (
            "einf_einops_einx_dynamic.py",
            dynamic_module,
            "_build_case_specs",
            "render_dynamic",
        ),
    ],
)
def test_library_compare_rechecks_source_content_before_receipt_publication(
    program: str,
    module: ModuleType,
    case_builder: str,
    render_method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [program, "--receipt", str(receipt)])
    monkeypatch.setattr(module, "einf_source_receipt_metadata", _source_metadata)
    monkeypatch.setattr(module, case_builder, lambda **kwargs: [])

    class Printer:
        def __getattr__(self, name: str) -> Callable[..., str]:
            assert name == render_method
            return lambda *args, **kwargs: "# report\n"

    monkeypatch.setattr(module, "MarkdownPrinter", Printer)

    def reject_changed_source(expected_sha256: str) -> None:
        assert expected_sha256 == "1" * 64
        raise RuntimeError("imported einf source changed during measurement")

    monkeypatch.setattr(
        module,
        "require_stable_einf_source_content",
        reject_changed_source,
    )

    with pytest.raises(RuntimeError, match="changed during measurement"):
        module.main()
    assert not receipt.exists()


def test_expression_compare_rechecks_source_content_before_receipt_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["expression_parity.py", "--receipt", str(receipt)],
    )
    monkeypatch.setattr(
        expression_parity_module,
        "einf_source_receipt_metadata",
        _source_metadata,
    )

    class Backend:
        name = "torch"
        requested_device = "cpu"
        resolved_device = "cpu"

    monkeypatch.setattr(
        expression_parity_module,
        "BackendSpec",
        lambda **kwargs: Backend(),
    )
    monkeypatch.setattr(
        expression_parity_module,
        "Profiler",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        expression_parity_module,
        "_find_case",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        expression_parity_module,
        "_run_dynamic_case",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        expression_parity_module,
        "_render_markdown",
        lambda report: "# report\n",
    )

    def reject_changed_source(expected_sha256: str) -> None:
        assert expected_sha256 == "1" * 64
        raise RuntimeError("imported einf source changed during measurement")

    monkeypatch.setattr(
        expression_parity_module,
        "require_stable_einf_source_content",
        reject_changed_source,
    )

    with pytest.raises(RuntimeError, match="changed during measurement"):
        expression_parity_module.main()
    assert not receipt.exists()
