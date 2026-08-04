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


def test_einf_source_content_digest_tracks_package_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "einf"
    package.mkdir()
    init_file = package / "__init__.py"
    init_file.write_text("VALUE = 1\n")
    (package / "api.pyi").write_text("VALUE: int\n")
    (package / "py.typed").write_text("")
    ignored_cache = package / "cached.pyc"
    ignored_cache.write_bytes(b"first")
    monkeypatch.setattr(
        metadata_module,
        "import_module",
        lambda name: _einf_module(str(init_file)),
    )

    first = metadata_module.einf_source_content_sha256()
    ignored_cache.write_bytes(b"second")
    assert metadata_module.einf_source_content_sha256() == first

    init_file.write_text("VALUE = 2\n")
    assert metadata_module.einf_source_content_sha256() != first


def test_einf_source_receipt_metadata_includes_content_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source: dict[str, str | bool | None] = {
        "kind": "git_checkout",
        "distribution_version": "0.2.0",
        "git_revision": "a" * 40,
        "git_dirty": False,
    }
    monkeypatch.setattr(metadata_module, "einf_source_metadata", lambda: dict(source))
    monkeypatch.setattr(
        metadata_module,
        "einf_source_content_sha256",
        lambda: "1" * 64,
    )

    assert metadata_module.einf_source_receipt_metadata() == {
        **source,
        "content_sha256": "1" * 64,
    }


def test_require_einf_source_root_rejects_another_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_root = tmp_path / "selected"
    package_file = selected_root / "src" / "einf" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.touch()
    monkeypatch.setattr(
        metadata_module,
        "import_module",
        lambda name: _einf_module(str(package_file)),
    )

    metadata_module.require_einf_source_root(selected_root)
    with pytest.raises(RuntimeError, match="does not belong"):
        metadata_module.require_einf_source_root(tmp_path / "other")


def test_require_stable_einf_source_content_rejects_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metadata_module,
        "einf_source_content_sha256",
        lambda: "after",
    )

    metadata_module.require_stable_einf_source_content("after")
    with pytest.raises(RuntimeError, match="changed during measurement"):
        metadata_module.require_stable_einf_source_content("before")
