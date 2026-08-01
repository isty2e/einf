"""Atomic publication for canonical benchmark artifacts."""

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

_TEMPORARY_NAME_ATTEMPTS = 128


def _open_exclusive_temporary(
    parent: Path,
    *,
    name_prefix: str,
    mode: int,
) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        path = parent / f"{name_prefix}{secrets.token_hex(8)}.tmp"
        try:
            return os.open(path, flags, mode), path
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique benchmark temporary file")


def _discard_temporary(path: Path, *, descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _normal_creation_mode(parent: Path) -> int:
    # An empty sibling captures the effective umask without mutating
    # process-global state or exposing receipt contents.
    descriptor, probe_path = _open_exclusive_temporary(
        parent,
        name_prefix=".einf-mode-probe-",
        mode=0o666,
    )
    open_descriptor: int | None = descriptor

    try:
        creation_mode = stat.S_IMODE(os.fstat(open_descriptor).st_mode) & 0o777
        os.close(open_descriptor)
        open_descriptor = None
        probe_path.unlink()
    except BaseException:
        _discard_temporary(probe_path, descriptor=open_descriptor)
        raise
    return creation_mode


def _regular_file_mode(path: Path) -> int | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return stat.S_IMODE(metadata.st_mode) & 0o777


def publish_receipt(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish JSON while preserving ordinary file permissions.

    New receipts use ``0o666`` filtered by the process umask. Replacing an
    existing regular file preserves its access permission bits. Existing
    symlinks remain in place while their resolved targets are replaced. Other
    inode metadata is not part of this boundary's preservation contract.
    """
    requested_destination = path.expanduser()
    requested_destination.parent.mkdir(parents=True, exist_ok=True)
    destination = requested_destination.resolve(strict=False)
    descriptor, temporary_path = _open_exclusive_temporary(
        destination.parent,
        name_prefix=".einf-receipt-",
        mode=0o600,
    )
    open_descriptor: int | None = descriptor

    try:
        stream = os.fdopen(open_descriptor, "w", encoding="utf-8", newline="\n")
        open_descriptor = None
        with stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        existing_mode = _regular_file_mode(destination)
        final_mode = (
            _normal_creation_mode(destination.parent)
            if existing_mode is None
            else existing_mode
        )
        temporary_path.chmod(final_mode)
        os.replace(temporary_path, destination)
    except BaseException:
        _discard_temporary(temporary_path, descriptor=open_descriptor)
        raise
