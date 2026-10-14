import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TextIO

import pytest

import benchmarks.shared.artifacts as artifact_module
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


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
def test_publish_receipt_updates_existing_symlink_target(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "receipt.json"
    target.write_text('{"run": "old"}\n', encoding="utf-8")
    target.chmod(0o640)
    receipt = tmp_path / "current.json"
    receipt.symlink_to(target)

    publish_receipt(receipt, {"run": "new"})

    assert receipt.is_symlink()
    assert receipt.resolve() == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"run": "new"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(target_dir.glob(".einf-receipt-*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
def test_publish_receipt_creates_dangling_symlink_target(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    receipt = tmp_path / "current.json"
    receipt.symlink_to(target.name)

    publish_receipt(receipt, {"run": "new"})

    assert receipt.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"run": "new"}


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
def test_failed_symlink_publication_preserves_link_and_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "receipt.json"
    original = '{"run": "old"}\n'
    target.write_text(original, encoding="utf-8")
    receipt = tmp_path / "current.json"
    receipt.symlink_to(target.name)

    def fail_serialization(payload: object, stream: TextIO, *, indent: int) -> None:
        _ = payload, indent
        stream.write('{"run":')
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(artifact_module.json, "dump", fail_serialization)

    with pytest.raises(RuntimeError, match="serialization failed"):
        publish_receipt(receipt, {"run": "new"})

    assert receipt.is_symlink()
    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permission bits")
def test_publish_receipt_preserves_existing_file_mode(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"run": "old"}\n', encoding="utf-8")
    receipt.chmod(0o640)

    publish_receipt(receipt, {"run": "new"})

    assert stat.S_IMODE(receipt.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permission bits")
def test_new_receipt_uses_normal_creation_mode(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"

    publish_receipt(receipt, {"run": "new"})

    assert stat.S_IMODE(receipt.stat().st_mode) == stat.S_IMODE(
        reference.stat().st_mode
    )
    assert list(tmp_path.glob(".einf-mode-probe-*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permission bits")
def test_temporary_receipt_requests_private_creation_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = artifact_module.os.open
    requested_modes: list[int] = []

    def observe_creation_mode(
        path: os.PathLike[str],
        flags: int,
        mode: int,
    ) -> int:
        if Path(path).name.startswith(".einf-receipt-"):
            requested_modes.append(mode)
        return original_open(path, flags, mode)

    monkeypatch.setattr(artifact_module.os, "open", observe_creation_mode)

    publish_receipt(tmp_path / "receipt.json", {"run": "new"})

    assert requested_modes == [0o600]


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permission bits")
def test_temporary_receipt_is_private_during_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dump = artifact_module.json.dump
    observed_modes: list[int] = []

    def observe_mode(payload: object, stream: TextIO, *, indent: int) -> None:
        observed_modes.append(stat.S_IMODE(os.fstat(stream.fileno()).st_mode))
        original_dump(payload, stream, indent=indent)

    monkeypatch.setattr(artifact_module.json, "dump", observe_mode)

    publish_receipt(tmp_path / "receipt.json", {"run": "new"})

    assert len(observed_modes) == 1
    assert observed_modes[0] & ~0o600 == 0


def test_fdopen_failure_preserves_primary_error_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_close = artifact_module.os.close

    def fail_stream_open(
        descriptor: int,
        mode: str,
        *,
        encoding: str,
        newline: str,
    ) -> TextIO:
        _ = descriptor, mode, encoding, newline
        raise RuntimeError("fdopen failed")

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError("close failed")

    monkeypatch.setattr(artifact_module.os, "fdopen", fail_stream_open)
    monkeypatch.setattr(artifact_module.os, "close", close_then_fail)

    with pytest.raises(RuntimeError, match="fdopen failed"):
        publish_receipt(tmp_path / "receipt.json", {"run": "new"})

    assert list(tmp_path.glob(".einf-receipt-*.tmp")) == []


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


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permission bits")
def test_failed_publication_preserves_existing_file_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"run": "old"}\n', encoding="utf-8")
    receipt.chmod(0o640)

    def fail_commit(source: Path, destination: Path) -> None:
        _ = source, destination
        raise OSError("commit failed")

    monkeypatch.setattr(artifact_module.os, "replace", fail_commit)

    with pytest.raises(OSError, match="commit failed"):
        publish_receipt(receipt, {"run": "new"})

    assert stat.S_IMODE(receipt.stat().st_mode) == 0o640


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
