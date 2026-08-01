"""Existing report-file identity checks for benchmark guardrails."""

from pathlib import Path, PurePath


def _paths_have_identical_spellings(first: PurePath, second: PurePath) -> bool:
    # Windows path equality case-folds even for case-sensitive directories.
    return str(first) == str(second)


def existing_report_paths_alias(first: Path, second: Path) -> bool:
    """Return whether two existing report paths identify the same file."""
    resolved_first = first.expanduser().resolve(strict=False)
    resolved_second = second.expanduser().resolve(strict=False)
    if _paths_have_identical_spellings(resolved_first, resolved_second):
        return True
    try:
        return resolved_first.samefile(resolved_second)
    except OSError:
        return False


__all__ = ["existing_report_paths_alias"]
