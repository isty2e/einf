"""Shared benchmark metadata helpers."""

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
