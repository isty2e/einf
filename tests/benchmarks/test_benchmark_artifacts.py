import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TextIO

import pytest

import benchmarks.audit.expression_layout as expression_layout_module
import benchmarks.shared.artifacts as artifact_module
from benchmarks.audit.expression_layout import LayoutAuditReport
from benchmarks.audit.expression_layout import main as expression_layout_main
from benchmarks.compare.einf_einops_einx import main as fixed_main
from benchmarks.compare.einf_einops_einx_dynamic import main as dynamic_main
from benchmarks.compare.expression_parity import main as expression_parity_main
from benchmarks.profile.overhead_breakdown import main as overhead_breakdown_main
from benchmarks.profile.warm_calltree import main as warm_calltree_main
from benchmarks.shared.artifacts import publish_receipt


def test_publish_receipt_replaces_existing_artifact(tmp_path: Path) -> None:
    receipt = tmp_path / "nested" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text('{"run": "old"}\n', encoding="utf-8")

    publish_receipt(receipt, {"run": "new", "samples": [1, 2, 3]})

    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "run": "new",
        "samples": [1, 2, 3],
    }
    assert receipt.read_text(encoding="utf-8").endswith("\n")
    assert list(receipt.parent.glob(".einf-receipt-*.tmp")) == []


def test_publish_receipt_preserves_target_when_serialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    original = '{"run": "old"}\n'
    receipt.write_text(original, encoding="utf-8")

    def fail_serialization(payload: object, stream: TextIO, *, indent: int) -> None:
        _ = payload, indent
        stream.write('{"run":')
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(artifact_module.json, "dump", fail_serialization)

    with pytest.raises(RuntimeError, match="serialization failed"):
        publish_receipt(receipt, {"run": "new"})

    assert receipt.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


def test_failed_publication_does_not_create_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"

    def fail_serialization(payload: object, stream: TextIO, *, indent: int) -> None:
        _ = payload, stream, indent
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(artifact_module.json, "dump", fail_serialization)

    with pytest.raises(RuntimeError, match="serialization failed"):
        publish_receipt(receipt, {"run": "new"})

    assert not receipt.exists()
    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


def test_publish_receipt_preserves_target_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    original = '{"run": "old"}\n'
    receipt.write_text(original, encoding="utf-8")

    def fail_commit(source: Path, destination: Path) -> None:
        _ = source, destination
        raise OSError("commit failed")

    monkeypatch.setattr(artifact_module.os, "replace", fail_commit)

    with pytest.raises(OSError, match="commit failed"):
        publish_receipt(receipt, {"run": "new"})

    assert receipt.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


def test_concurrent_publishers_leave_one_complete_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    initial_payload = {"writer": -1, "samples": ["initial"]}
    payloads = [
        {"writer": writer, "samples": [str(writer)] * 10_000} for writer in range(8)
    ]
    publish_receipt(receipt, initial_payload)
    start = Barrier(len(payloads) + 1)

    def publish(payload: dict[str, object]) -> None:
        start.wait()
        publish_receipt(receipt, payload)

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [executor.submit(publish, payload) for payload in payloads]
        start.wait()
        while not all(future.done() for future in futures):
            observed = json.loads(receipt.read_text(encoding="utf-8"))
            assert observed == initial_payload or observed in payloads
        for future in futures:
            future.result()

    assert json.loads(receipt.read_text(encoding="utf-8")) in payloads
    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


@pytest.mark.skipif(not hasattr(os, "pathconf"), reason="requires POSIX NAME_MAX")
@pytest.mark.parametrize("character", ["a", "界"])
def test_publish_receipt_accepts_name_max_target(
    tmp_path: Path,
    character: str,
) -> None:
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    suffix = ".json"
    character_size = len(os.fsencode(character))
    count, remainder = divmod(name_max - len(os.fsencode(suffix)), character_size)
    name = (character * count) + ("a" * remainder) + suffix
    assert len(os.fsencode(name)) == name_max
    receipt = tmp_path / name

    publish_receipt(receipt, {"name_bytes": name_max})

    assert json.loads(receipt.read_text(encoding="utf-8")) == {"name_bytes": name_max}


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
