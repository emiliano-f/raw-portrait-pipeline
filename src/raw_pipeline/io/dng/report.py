"""CSV report of ``fix-raw`` (one row per input)."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from raw_pipeline.core.errors import OutputExistsError


def write_fix_raw_report(rows: Sequence[Mapping[str, object]], path: Path, *, overwrite: bool) -> Path:
    """Write ``rows`` (same keys, in order) to ``path`` atomically; refuse to replace it unless ``overwrite``."""
    if path.exists() and not overwrite:
        msg = f"{path} already exists; pass --overwrite to replace it"
        raise OutputExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
