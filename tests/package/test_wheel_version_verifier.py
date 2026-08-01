import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

import tests.package.verify_wheel_version as verifier


def _write_wheel(tmp_path: Path, *, version: str) -> Path:
    wheel = tmp_path / f"einf-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.4\nName: einf\nVersion: {version}\n\n"
    with ZipFile(wheel, mode="w") as archive:
        archive.writestr(f"einf-{version}.dist-info/METADATA", metadata)
    return wheel


def test_wheel_version_verifier_accepts_matching_generated_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(tmp_path, version="1.2.3")
    generated_version = tmp_path / "_version.py"
    generated_version.write_text(
        "__version__ = version = '1.2.3'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify-wheel-version", str(wheel), str(generated_version)],
    )

    verifier.main()


def test_wheel_version_verifier_rejects_mismatched_generated_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(tmp_path, version="1.2.3")
    generated_version = tmp_path / "_version.py"
    generated_version.write_text("__version__ = '1.2.4'\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify-wheel-version", str(wheel), str(generated_version)],
    )

    with pytest.raises(RuntimeError, match="wheel version does not match"):
        verifier.main()
