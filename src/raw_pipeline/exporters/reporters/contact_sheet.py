"""Contact sheet: before/after thumbnails per image plus 100 % defect crops."""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from raw_pipeline.config import app_config
from raw_pipeline.core.protocols import ImageOutcome
from raw_pipeline.core.registries import reporter_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import Scalar
from raw_pipeline.exporters.encoding import srgb_icc_profile
from raw_pipeline.exporters.output import write_output

THUMBNAIL_COLUMNS: Final = ("before", "after")
DEFECT_COLUMNS: Final = ("defect_before", "defect_after")
COLUMN_TITLES: Final = {
    "before": "Before (camera WB)",
    "after": "After",
    "defect_before": "Defect before (100 %)",
    "defect_after": "Defect after (100 %)",
}
DEFAULT_CAPTION_KEYS: Final = (
    "meter.method",
    "meter.correction_ev",
    "meter.faces_used",
    "retouch.method",
    "retouch.strength",
    "defects.method",
    "defects.corrected_pixel_count",
    "qa.flags",
)
"""Metadata keys shown under each row when present (others are in the CSV)."""

GUTTER_PX: Final = 8
FONT_SIZE_PX: Final = 13
LINE_HEIGHT_PX: Final = 18
CAPTION_MAX_LINES: Final = 3
MAX_ROWS_PER_PAGE: Final = 40
"""Rows per page; fewer when large thumbnails would exceed :data:`JPEG_MAX_SIDE_PX`."""
JPEG_MAX_SIDE_PX: Final = 65_535
SHEET_JPEG_QUALITY: Final = 90
CHROMA_SUBSAMPLING_444: Final = 0
BACKGROUND_RGB: Final = (24, 24, 24)
PLACEHOLDER_RGB: Final = (56, 56, 56)
TEXT_RGB: Final = (232, 232, 232)
ERROR_RGB: Final = (255, 120, 110)


@dataclass(frozen=True)
class _Canvas:
    sheet: Image.Image
    draw: ImageDraw.ImageDraw
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    columns: Sequence[tuple[str, int]]
    cell_height: int


@dataclass(frozen=True)
class _SheetRow:
    name: str
    images: dict[str, NDArray[np.uint8]]
    caption: str
    error: str | None


def _format(value: Scalar) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def _wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width_px: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current}  {word}" if current else word
        if current and font.getlength(candidate) > width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:CAPTION_MAX_LINES]


@reporter_registry.register("contact_sheet")
class ContactSheetReporter:
    """Renders a deterministic JPEG grid: one row per image, sorted by file name.

    Columns: before, after (fitted into ``thumbnail_long_edge_px`` cells) and,
    when any image has them, the before/after defect crops shown at 100 %
    (one sheet pixel per image pixel). Each row has a caption with the file
    name and key parameters; failed images show the error instead. Large
    batches are split into pages (``name.jpg``, ``name-2.jpg``, ...).

    Args:
        filename: Contact sheet file name inside the output directory.
        thumbnail_long_edge_px: Edge of the square before/after cells, in pixels.
        overwrite: Replace existing sheets instead of raising ``OutputExistsError``.
        caption_keys: Metadata keys (``stage.param``) printed under each row.
    """

    name: ClassVar[str] = "contact_sheet"

    def __init__(
        self,
        filename: str,
        thumbnail_long_edge_px: int,
        *,
        overwrite: bool,
        caption_keys: Sequence[str] = DEFAULT_CAPTION_KEYS,
    ) -> None:
        self.filename = filename
        self.thumbnail_long_edge_px = thumbnail_long_edge_px
        self.overwrite = overwrite
        self.caption_keys = tuple(caption_keys)
        self._rows: list[_SheetRow] = []

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``report`` config section and the run's overwrite flag."""
        cfg = app_config(env).report
        return cls(cfg.contact_sheet_filename, cfg.thumbnail_long_edge_px, overwrite=env.overwrite)

    def add(self, outcome: ImageOutcome) -> None:
        """Keep the outcome's thumbnails (already small) and caption."""
        flat = outcome.metadata.flatten()
        caption = "  ".join(f"{key}={_format(flat[key])}" for key in self.caption_keys if key in flat)
        error = f"FAILED {outcome.error_type or 'error'}: {outcome.error}" if not outcome.ok else None
        self._rows.append(_SheetRow(outcome.source_path.name, dict(outcome.thumbnails), caption, error))

    def render(self) -> Iterator[Image.Image]:
        """Yield the sheet pages for the outcomes added so far, one at a time."""
        rows = sorted(self._rows, key=lambda row: row.name)
        if not rows:
            yield self._render_page([])
            return
        per_page = self._rows_per_page(rows)
        for start in range(0, len(rows), per_page):
            yield self._render_page(rows[start : start + per_page])

    def finalize(self, output_dir: Path) -> list[Path]:
        """Write the sheet page(s) into ``output_dir`` and return their paths."""
        base = Path(self.filename)
        written: list[Path] = []
        for index, page in enumerate(self.render()):
            data = self._encode(page)
            name = base.name if index == 0 else f"{base.stem}-{index + 1}{base.suffix}"

            def write(handle: BinaryIO, data: bytes = data) -> None:
                handle.write(data)

            written.append(write_output(output_dir / base.parent / name, write, overwrite=self.overwrite))
        return written

    def _rows_per_page(self, rows: Sequence[_SheetRow]) -> int:
        available = JPEG_MAX_SIDE_PX - GUTTER_PX - LINE_HEIGHT_PX
        return max(1, min(MAX_ROWS_PER_PAGE, available // self._row_height(self._columns(rows))))

    @staticmethod
    def _row_height(columns: Sequence[tuple[str, int]]) -> int:
        return max(edge for _, edge in columns) + CAPTION_MAX_LINES * LINE_HEIGHT_PX + GUTTER_PX

    def _columns(self, rows: Sequence[_SheetRow]) -> list[tuple[str, int]]:
        columns = [(key, self.thumbnail_long_edge_px) for key in THUMBNAIL_COLUMNS]
        crop_edge = max(
            (max(row.images[key].shape[:2]) for row in rows for key in DEFECT_COLUMNS if key in row.images),
            default=0,
        )
        if crop_edge:
            columns += [(key, crop_edge) for key in DEFECT_COLUMNS]
        return columns

    def _render_page(self, rows: Sequence[_SheetRow]) -> Image.Image:
        font = ImageFont.load_default(size=FONT_SIZE_PX)
        columns = self._columns(rows)
        width = GUTTER_PX + sum(edge + GUTTER_PX for _, edge in columns)
        cell_height = max(edge for _, edge in columns)
        row_height = self._row_height(columns)
        height = GUTTER_PX + LINE_HEIGHT_PX + len(rows) * row_height
        sheet = Image.new("RGB", (width, height), BACKGROUND_RGB)
        draw = ImageDraw.Draw(sheet)
        x = GUTTER_PX
        for key, edge in columns:
            draw.text((x, GUTTER_PX), COLUMN_TITLES[key], fill=TEXT_RGB, font=font)
            x += edge + GUTTER_PX
        canvas = _Canvas(sheet, draw, font, columns, cell_height)
        for index, row in enumerate(rows):
            self._draw_row(canvas, row, top=GUTTER_PX + LINE_HEIGHT_PX + index * row_height)
        return sheet

    @staticmethod
    def _draw_row(canvas: _Canvas, row: _SheetRow, *, top: int) -> None:
        x = GUTTER_PX
        for key, edge in canvas.columns:
            image = row.images.get(key)
            if image is None:
                canvas.draw.rectangle((x, top, x + edge - 1, top + edge - 1), fill=PLACEHOLDER_RGB)
            else:
                tile = Image.fromarray(image[:edge, :edge], mode="RGB")
                canvas.sheet.paste(tile, (x + (edge - tile.width) // 2, top + (edge - tile.height) // 2))
            x += edge + GUTTER_PX
        text = f"{row.name}  {row.error or row.caption}"
        colour = ERROR_RGB if row.error else TEXT_RGB
        text_top = top + canvas.cell_height
        for i, line in enumerate(_wrap(text, canvas.font, canvas.sheet.width - 2 * GUTTER_PX)):
            canvas.draw.text((GUTTER_PX, text_top + i * LINE_HEIGHT_PX), line, fill=colour, font=canvas.font)

    @staticmethod
    def _encode(page: Image.Image) -> bytes:
        buffer = io.BytesIO()
        page.save(
            buffer,
            format="JPEG",
            quality=SHEET_JPEG_QUALITY,
            subsampling=CHROMA_SUBSAMPLING_444,
            optimize=False,
            icc_profile=srgb_icc_profile(),
        )
        return buffer.getvalue()
