"""Atomic publication for canonical benchmark artifacts."""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def publish_receipt(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one JSON benchmark receipt."""
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".einf-receipt-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    open_descriptor: int | None = descriptor

    try:
        stream = os.fdopen(open_descriptor, "w", encoding="utf-8", newline="\n")
        open_descriptor = None
        with stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, destination)
    except BaseException:
        if open_descriptor is not None:
            os.close(open_descriptor)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
