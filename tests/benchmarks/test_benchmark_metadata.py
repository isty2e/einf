from pathlib import Path
from types import ModuleType

import pytest

import benchmarks.shared.metadata as metadata_module


def _einf_module(path: str) -> ModuleType:
    module = ModuleType("einf")
    module.__file__ = path
    return module


def test_einf_source_metadata_records_tracked_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metadata_module,
        "import_module",
        lambda name: _einf_module("/repo/src/einf/__init__.py"),
    )
    monkeypatch.setattr(
        metadata_module,
        "version_or_missing",
        lambda name: "0.2.0",
    )

    def git_output(*args: str, cwd: Path) -> str | None:
        if args == ("rev-parse", "--show-toplevel"):
            return "/repo"
        if args[:2] == ("ls-files", "--error-unmatch"):
            return "src/einf/__init__.py"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[:2] == ("status", "--porcelain=v1"):
            return " M src/einf/__init__.py"
        raise AssertionError((args, cwd))

    monkeypatch.setattr(metadata_module, "_git_output", git_output)

    assert metadata_module.einf_source_metadata() == {
        "kind": "git_checkout",
        "distribution_version": "0.2.0",
        "git_revision": "a" * 40,
        "git_dirty": True,
    }


def test_einf_source_metadata_does_not_misidentify_untracked_venv_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metadata_module,
        "import_module",
        lambda name: _einf_module(
            "/repo/.venv/lib/python3.13/site-packages/einf/__init__.py"
        ),
    )
    monkeypatch.setattr(
        metadata_module,
        "version_or_missing",
        lambda name: "0.2.0",
    )

    def git_output(*args: str, cwd: Path) -> str | None:
        if args == ("rev-parse", "--show-toplevel"):
            return "/repo"
        if args[:2] == ("ls-files", "--error-unmatch"):
            return None
        raise AssertionError((args, cwd))

    monkeypatch.setattr(metadata_module, "_git_output", git_output)

    assert metadata_module.einf_source_metadata() == {
        "kind": "installed_distribution",
        "distribution_version": "0.2.0",
        "git_revision": None,
        "git_dirty": None,
    }
