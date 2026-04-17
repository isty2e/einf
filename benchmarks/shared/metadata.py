"""Shared benchmark metadata helpers."""

from importlib import metadata
from types import ModuleType


def version_or_missing(package_name: str) -> str:
    """Return installed package version or ``not-installed``."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


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
