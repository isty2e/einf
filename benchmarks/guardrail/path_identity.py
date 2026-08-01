"""Existing report-file identity checks for benchmark guardrails."""

from pathlib import Path


def existing_report_paths_alias(first: Path, second: Path) -> bool:
    """Return whether two existing report paths identify the same file."""
    try:
        resolved_first = first.expanduser().resolve(strict=True)
        resolved_second = second.expanduser().resolve(strict=True)
    except FileNotFoundError:
        return False
    return resolved_first.samefile(resolved_second)


__all__ = ["existing_report_paths_alias"]
