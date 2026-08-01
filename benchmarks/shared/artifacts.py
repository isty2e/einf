"""Atomic publication for canonical benchmark artifacts."""

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

_TEMPORARY_NAME_ATTEMPTS = 128


def _open_temporary_receipt(parent: Path) -> tuple[int, Path, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        path = parent / f".einf-receipt-{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            continue
        try:
            creation_mode = stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o777
            path.chmod(0o600)
        except BaseException:
            os.close(descriptor)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return descriptor, path, creation_mode
    raise FileExistsError("could not allocate a unique receipt temporary file")


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
    existing regular file preserves its access permission bits. Other inode
    metadata is not part of this boundary's preservation contract.
    """
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path, creation_mode = _open_temporary_receipt(
        destination.parent
    )
    open_descriptor: int | None = descriptor

    try:
        stream = os.fdopen(open_descriptor, "w", encoding="utf-8", newline="\n")
        open_descriptor = None
        with stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        existing_mode = _regular_file_mode(destination)
        temporary_path.chmod(creation_mode if existing_mode is None else existing_mode)
        os.replace(temporary_path, destination)
    except BaseException:
        if open_descriptor is not None:
            os.close(open_descriptor)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
