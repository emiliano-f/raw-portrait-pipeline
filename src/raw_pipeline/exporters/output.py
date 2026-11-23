"""Writing output files without ever overwriting silently."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from raw_pipeline.core.errors import ExportError, OutputExistsError


def write_output(path: Path, write: Callable[[BinaryIO], None], *, overwrite: bool) -> Path:
    """Create ``path`` and fill it with ``write``.

    Without ``overwrite`` the file is opened exclusively, so an existing file is
    never replaced by accident (not even by a concurrent writer). With
    ``overwrite`` the data goes to a temporary sibling that atomically replaces
    ``path`` once complete, so a failed write keeps the previous file. A
    partially written file is always removed when ``write`` fails.

    Raises:
        OutputExistsError: ``path`` exists and ``overwrite`` is false.
        ExportError: The file could not be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create {path.parent}: {exc}"
        raise ExportError(msg) from exc
    # The handle from the exclusive create is the one written: reopening by name could follow a swapped-in symlink.
    target, handle = _open_temporary(path) if overwrite else _open_exclusive(path)
    try:
        with handle:
            write(handle)
        if target != path:
            target.replace(path)
    except Exception as exc:
        target.unlink(missing_ok=True)
        if isinstance(exc, ExportError):
            raise
        msg = f"cannot write {path}: {exc}"
        raise ExportError(msg) from exc
    return path


def _open_exclusive(path: Path) -> tuple[Path, BinaryIO]:
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        msg = f"{path} already exists; pass --overwrite to replace it"
        raise OutputExistsError(msg) from exc
    except OSError as exc:
        msg = f"cannot create {path}: {exc}"
        raise ExportError(msg) from exc
    return path, handle


def _open_temporary(path: Path) -> tuple[Path, BinaryIO]:
    # Same directory (atomic replace), unique per writer, created with the umask's default mode.
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.partial")
    try:
        handle = temporary.open("xb")  # exclusive: never follows a planted symlink
    except OSError as exc:
        msg = f"cannot create {path}: {exc}"
        raise ExportError(msg) from exc
    return temporary, handle
