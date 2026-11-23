"""Writing output files without ever overwriting silently."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from raw_pipeline.core.errors import ExportError, OutputExistsError


def write_output(path: Path, write: Callable[[BinaryIO], None], *, overwrite: bool) -> Path:
    """Create ``path`` and fill it with ``write``.

    The file is opened exclusively unless ``overwrite`` is true, so an existing
    file is never replaced by accident (not even by a concurrent writer). A
    partially written file is removed when ``write`` fails.

    Raises:
        OutputExistsError: ``path`` exists and ``overwrite`` is false.
        ExportError: The file could not be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create {path.parent}: {exc}"
        raise ExportError(msg) from exc
    try:
        handle = path.open("wb" if overwrite else "xb")
    except FileExistsError as exc:
        msg = f"{path} already exists; pass --overwrite to replace it"
        raise OutputExistsError(msg) from exc
    except OSError as exc:
        msg = f"cannot create {path}: {exc}"
        raise ExportError(msg) from exc
    try:
        with handle:
            write(handle)
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc, ExportError):
            raise
        msg = f"cannot write {path}: {exc}"
        raise ExportError(msg) from exc
    return path
