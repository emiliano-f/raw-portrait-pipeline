"""CSV report: one row per image with every parameter applied."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO, ClassVar, Final, Self

from raw_pipeline.config import app_config
from raw_pipeline.core.protocols import ImageOutcome
from raw_pipeline.core.registries import reporter_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import Scalar
from raw_pipeline.exporters.output import write_output

LEADING_COLUMNS: Final = ("file", "status", "error_type", "error", "outputs")
"""Columns before the flattened ``stage.param`` metadata columns."""
STATUS_OK: Final = "ok"
STATUS_FAILED: Final = "failed"
OUTPUT_SEPARATOR: Final = ";"
FORMULA_PREFIXES: Final = ("=", "+", "-", "@", "\t", "\r")
"""Text cells starting with these are prefixed with ``'`` so spreadsheets do not evaluate them."""


def text_cell(text: str) -> str:
    """Neutralise spreadsheet formulas in a free-text cell (file names, errors)."""
    return f"'{text}" if text.startswith(FORMULA_PREFIXES) else text


def format_value(value: Scalar) -> str:
    """Render a metadata value for a CSV cell.

    ``None`` is empty, booleans are ``true``/``false`` and floats use the
    shortest representation that round-trips. Strings go through
    :func:`text_cell`; numbers are never prefixed, so negative values stay numeric.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        return text_cell(value)
    return str(value)


def _row(outcome: ImageOutcome) -> dict[str, str]:
    row = {
        "file": text_cell(outcome.source_path.name),
        "status": STATUS_OK if outcome.ok else STATUS_FAILED,
        "error_type": text_cell(outcome.error_type or ""),
        "error": text_cell(outcome.error or ""),
        "outputs": text_cell(OUTPUT_SEPARATOR.join(path.name for path in outcome.outputs)),
    }
    row.update({key: format_value(value) for key, value in outcome.metadata.flatten().items()})
    return row


def metadata_columns(rows: list[dict[str, str]]) -> list[str]:
    """Order metadata columns by stage (first appearance), then alphabetically within a stage.

    ``exif.*`` comes first because :meth:`ImageMetadata.flatten` lists it first;
    stages follow in pipeline order. Rows must already be sorted for the result
    to be deterministic.
    """
    groups: dict[str, set[str]] = {}
    for row in rows:
        for key in row:
            if key not in LEADING_COLUMNS:
                groups.setdefault(key.split(".", 1)[0], set()).add(key)
    return [key for keys in groups.values() for key in sorted(keys)]


@reporter_registry.register("csv")
class CsvReporter:
    """Collects one row per image (successful or failed) and writes a UTF-8 CSV.

    Rows are sorted by file name so parallel processing order never changes the
    file. Missing values (e.g. stages that did not run for a failed image) are
    empty cells.

    Args:
        filename: File name inside the output directory.
        overwrite: Replace an existing report instead of raising ``OutputExistsError``.
    """

    name: ClassVar[str] = "csv"

    def __init__(self, filename: str, *, overwrite: bool) -> None:
        self.filename = filename
        self.overwrite = overwrite
        self._rows: list[dict[str, str]] = []

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from ``report.csv_filename`` and the run's overwrite flag."""
        return cls(app_config(env).report.csv_filename, overwrite=env.overwrite)

    def add(self, outcome: ImageOutcome) -> None:
        """Record one image outcome."""
        self._rows.append(_row(outcome))

    def render(self) -> str:
        """Return the CSV text for the outcomes added so far."""
        rows = sorted(self._rows, key=lambda row: row["file"])
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=[*LEADING_COLUMNS, *metadata_columns(rows)], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    def finalize(self, output_dir: Path) -> list[Path]:
        """Write the CSV into ``output_dir`` and return its path."""
        data = self.render().encode("utf-8")

        def write(handle: BinaryIO) -> None:
            handle.write(data)

        return [write_output(output_dir / self.filename, write, overwrite=self.overwrite)]
