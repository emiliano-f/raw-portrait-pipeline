"""Guards for the batch report files written next to the exported images.

Exported images are named ``<input stem><export.filename_suffix>.<ext>`` and the
reports use the ``report.*_filename`` names; the contact sheet adds pages named
``<stem>-2<ext>``, ``<stem>-3<ext>``, ... when the batch does not fit one page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from raw_pipeline.config.models import AppConfig

FILENAME_FIELD_SUFFIX = "_filename"
"""``ReportConfig`` fields ending in this name are report files in the output folder."""


def report_filenames(config: AppConfig) -> tuple[str, ...]:
    """Every report file name configured in the ``report`` section."""
    report = config.report
    names = (getattr(report, field) for field in type(report).model_fields if field.endswith(FILENAME_FIELD_SUFFIX))
    return tuple(name for name in names if isinstance(name, str))


def report_name_collisions(sources: Sequence[Path], config: AppConfig) -> list[str]:
    """Describe each input whose output name would clash with a report file.

    Names are compared by stem, case-insensitively (outputs and reports differ
    only by extension, and macOS/Windows file systems ignore case).
    """
    reserved = {Path(name).stem.casefold(): name for name in report_filenames(config)}
    pages = _page_pattern(config.report.contact_sheet_filename, re.IGNORECASE)
    clashes = []
    for source in sources:
        stem = f"{source.stem}{config.export.filename_suffix}"
        if stem.casefold() in reserved:
            clashes.append(f"{source.name} -> {reserved[stem.casefold()]}")
        elif pages.fullmatch(f"{stem}{Path(config.report.contact_sheet_filename).suffix}"):
            clashes.append(f"{source.name} -> contact sheet page {stem}")
    return clashes


def stale_contact_sheet_pages(output_dir: Path, config: AppConfig, written: Iterable[Path]) -> list[Path]:
    """Contact sheet pages in ``output_dir`` left from an earlier run (not in ``written``)."""
    pattern = _page_pattern(config.report.contact_sheet_filename)
    current = {path.name for path in written}
    return sorted(path for path in output_dir.iterdir() if pattern.fullmatch(path.name) and path.name not in current)


def _page_pattern(filename: str, flags: int = 0) -> re.Pattern[str]:
    base = Path(filename)
    return re.compile(rf"{re.escape(base.stem)}-[0-9]+{re.escape(base.suffix)}", flags)
