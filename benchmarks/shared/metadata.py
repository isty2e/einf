"""Shared benchmark metadata helpers."""

import hashlib
import subprocess
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType


def version_or_missing(package_name: str) -> str:
    """Return installed package version or ``not-installed``."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _git_output(*args: str, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def einf_source_metadata() -> dict[str, str | bool | None]:
    """Describe the imported einf source without exposing local paths."""
    distribution_version = version_or_missing("einf")
    module_file = getattr(import_module("einf"), "__file__", None)
    fallback: dict[str, str | bool | None] = {
        "kind": "installed_distribution",
        "distribution_version": distribution_version,
        "git_revision": None,
        "git_dirty": None,
    }
    if module_file is None:
        return fallback

    source_file = Path(module_file).resolve()
    root_value = _git_output("rev-parse", "--show-toplevel", cwd=source_file.parent)
    if root_value is None:
        return fallback
    repository_root = Path(root_value)
    try:
        relative_source = source_file.relative_to(repository_root)
    except ValueError:
        return fallback
    tracked_source = _git_output(
        "ls-files",
        "--error-unmatch",
        "--",
        str(relative_source),
        cwd=repository_root,
    )
    if tracked_source is None:
        return fallback

    revision = _git_output("rev-parse", "HEAD", cwd=repository_root)
    status = _git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        cwd=repository_root,
    )
    if revision is None or status is None:
        return fallback
    return {
        "kind": "git_checkout",
        "distribution_version": distribution_version,
        "git_revision": revision,
        "git_dirty": bool(status),
    }


def einf_source_content_sha256() -> str:
    """Hash the imported ``einf`` package sources used by a benchmark run."""
    module_file = getattr(import_module("einf"), "__file__", None)
    if module_file is None:
        raise RuntimeError("cannot fingerprint imported einf source without __file__")

    package_root = Path(module_file).resolve().parent
    source_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and (path.suffix in {".py", ".pyi"} or path.name == "py.typed")
    )
    if not source_files:
        raise RuntimeError(
            "cannot fingerprint imported einf source without source files"
        )

    digest = hashlib.sha256()
    for source_file in source_files:
        relative_path = source_file.relative_to(package_root).as_posix().encode()
        content = source_file.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def einf_source_receipt_metadata() -> dict[str, str | bool | None]:
    """Describe and fingerprint the imported package for a benchmark receipt."""
    source = einf_source_metadata()
    source["content_sha256"] = einf_source_content_sha256()
    return source


def require_einf_source_root(checkout_root: Path) -> None:
    """Fail unless the imported package belongs to the selected checkout."""
    module_file = getattr(import_module("einf"), "__file__", None)
    if module_file is None:
        raise RuntimeError("imported einf package has no __file__")
    expected_package_root = (checkout_root / "src" / "einf").resolve()
    imported_source = Path(module_file).resolve()
    if not imported_source.is_relative_to(expected_package_root):
        raise RuntimeError(
            "imported einf source does not belong to the expected checkout: "
            f"expected {expected_package_root}, got {imported_source}"
        )


def require_stable_einf_source_content(expected_sha256: str) -> None:
    """Fail when imported package sources change during a benchmark run."""
    if einf_source_content_sha256() != expected_sha256:
        raise RuntimeError("imported einf source changed during measurement")


def available_libraries(
    *,
    einops_module: ModuleType | None,
    einx_module: ModuleType | None,
) -> dict[str, tuple[bool, str]]:
    """Report benchmark library availability with stable reason strings."""
    return {
        "einf": (True, "available"),
        "einops": (
            einops_module is not None,
            "not installed" if einops_module is None else "available",
        ),
        "einx": (
            einx_module is not None,
            "not installed" if einx_module is None else "available",
        ),
    }
