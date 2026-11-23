"""Guards for the batch report files written next to the exported images.

Exported images are named ``<input stem><export.filename_suffix>.<ext>``. A
reporter named ``<name>`` in ``report.reporters`` writes ``report.<name>_filename``
(``csv`` -> ``csv_filename``); the contact sheet adds pages named ``<stem>-2<ext>``,
``<stem>-3<ext>``, ... when the batch does not fit one page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from raw_pipeline.config.models import AppConfig

FILENAME_FIELD_SUFFIX = "_filename"
"""``report.<reporter name><FILENAME_FIELD_SUFFIX>`` is the file that reporter writes."""

CONTACT_SHEET_REPORTER = "contact_sheet"
"""Reporter whose output is split into numbered pages."""


def report_filenames(config: AppConfig) -> tuple[str, ...]:
    """The report file names of the reporters enabled in ``report.reporters``."""
    report = config.report
    fields = (f"{name}{FILENAME_FIELD_SUFFIX}" for name in report.reporters)
    names = (getattr(report, field, None) for field in fields)
    return tuple(name for name in names if isinstance(name, str))


def report_name_collisions(sources: Sequence[Path], config: AppConfig) -> list[str]:
    """Describe each input whose output name would clash with a report file.

    Names are compared by stem, case-insensitively (outputs and reports differ
    only by extension, and macOS/Windows file systems ignore case).
    """
    reserved = {Path(name).stem.casefold(): name for name in report_filenames(config)}
    paged = CONTACT_SHEET_REPORTER in config.report.reporters
    pages = _page_pattern(config.report.contact_sheet_filename, re.IGNORECASE)
    clashes = []
    for source in sources:
        stem = f"{source.stem}{config.export.filename_suffix}"
        if stem.casefold() in reserved:
            clashes.append(f"{source.name} -> {reserved[stem.casefold()]}")
        elif paged and pages.fullmatch(f"{stem}{Path(config.report.contact_sheet_filename).suffix}"):
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


def duplicate_output_stems(sources: Sequence[Path]) -> list[str]:
    """Describe inputs whose output names would clash (same stem, ignoring case).

    ``IMG_1.CR2`` and ``img_1.cr2`` would both export to ``IMG_1.jpg`` on a
    case-insensitive file system, and one export would silently replace the other.
    """
    by_stem: dict[str, list[str]] = {}
    for source in sources:
        by_stem.setdefault(source.stem.casefold(), []).append(source.name)
    return [" / ".join(sorted(names)) for _, names in sorted(by_stem.items()) if len(names) > 1]
