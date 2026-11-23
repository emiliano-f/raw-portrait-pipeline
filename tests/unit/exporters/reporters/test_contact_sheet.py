from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raw_pipeline.core.errors import OutputExistsError
from raw_pipeline.core.registries import reporter_registry
from raw_pipeline.exporters.encoding import srgb_icc_profile
from raw_pipeline.exporters.reporters import contact_sheet
from raw_pipeline.exporters.reporters.contact_sheet import (
    CAPTION_MAX_LINES,
    GUTTER_PX,
    LINE_HEIGHT_PX,
    ContactSheetReporter,
)
from tests.unit.exporters.helpers import build_env
from tests.unit.exporters.reporters.fakes import failed_outcome, ok_outcome

THUMB = 64


def _sheet(*outcomes, **kwargs) -> ContactSheetReporter:
    reporter = ContactSheetReporter("sheet.jpg", THUMB, overwrite=False, **kwargs)
    for outcome in outcomes:
        reporter.add(outcome)
    return reporter


def _expected_size(rows: int, columns: list[int]) -> tuple[int, int]:
    width = GUTTER_PX + sum(edge + GUTTER_PX for edge in columns)
    row_height = max(columns) + CAPTION_MAX_LINES * LINE_HEIGHT_PX + GUTTER_PX
    return width, GUTTER_PX + LINE_HEIGHT_PX + rows * row_height


def test_dimensions_without_defect_crops(tmp_path: Path) -> None:
    [path] = _sheet(ok_outcome("A.CR2"), failed_outcome("B.CR2")).finalize(tmp_path)
    with Image.open(path) as img:
        assert img.size == _expected_size(2, [THUMB, THUMB])
        assert img.info["icc_profile"] == srgb_icc_profile()


def test_defect_crops_add_full_size_columns(tmp_path: Path) -> None:
    [path] = _sheet(ok_outcome("A.CR2", crops=True), ok_outcome("B.CR2")).finalize(tmp_path)
    with Image.open(path) as img:
        assert img.size == _expected_size(2, [THUMB, THUMB, 20, 20])
        pixels = np.asarray(img)
    top = GUTTER_PX + LINE_HEIGHT_PX  # crops are top-aligned in their 1:1 cell
    left = GUTTER_PX + 2 * (THUMB + GUTTER_PX)
    crop_before = pixels[top + 2 : top + 18, left + 2 : left + 18]
    crop_after = pixels[top + 2 : top + 18, left + 20 + GUTTER_PX + 2 : left + 20 + GUTTER_PX + 18]
    assert crop_before.min() > 230
    assert crop_after.max() < 25


def test_deterministic_regardless_of_add_order(tmp_path: Path) -> None:
    outcomes = [ok_outcome("B.CR2", crops=True, seed=1), failed_outcome("A.CR2"), ok_outcome("C.CR2", seed=2)]
    [a] = _sheet(*outcomes).finalize(tmp_path / "a")
    [b] = _sheet(*reversed(outcomes)).finalize(tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_caption_lists_present_keys_only() -> None:
    reporter = _sheet(ok_outcome("A.CR2", ev=0.456), caption_keys=("meter.correction_ev", "tone.missing"))
    assert reporter._rows[0].caption == "meter.correction_ev=0.46"
    assert _sheet(failed_outcome("B.CR2"))._rows[0].error == "FAILED DecodeError: truncated file"


def test_large_batches_are_paginated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contact_sheet, "MAX_ROWS_PER_PAGE", 2)
    paths = _sheet(*(ok_outcome(f"IMG_{i}.CR2") for i in range(5))).finalize(tmp_path)
    assert [p.name for p in paths] == ["sheet.jpg", "sheet-2.jpg", "sheet-3.jpg"]
    with Image.open(paths[-1]) as img:
        assert img.size == _expected_size(1, [THUMB, THUMB])


def test_empty_batch_and_no_silent_overwrite(tmp_path: Path) -> None:
    [path] = _sheet().finalize(tmp_path)
    assert path.exists()
    with pytest.raises(OutputExistsError):
        _sheet().finalize(tmp_path)


def test_built_from_config(tmp_path: Path) -> None:
    env = build_env(tmp_path, report={"contact_sheet_filename": "cs.jpg", "thumbnail_long_edge_px": 300})
    reporter = reporter_registry.create("contact_sheet", env)
    assert isinstance(reporter, ContactSheetReporter)
    assert (reporter.filename, reporter.thumbnail_long_edge_px, reporter.overwrite) == ("cs.jpg", 300, False)


def test_pages_stay_within_jpeg_side_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limit = _expected_size(2, [THUMB, THUMB])[1] + 5
    monkeypatch.setattr(contact_sheet, "JPEG_MAX_SIDE_PX", limit)
    paths = _sheet(*(ok_outcome(f"IMG_{i}.CR2") for i in range(5))).finalize(tmp_path)
    assert len(paths) == 3
    for path in paths:
        with Image.open(path) as img:
            assert img.height <= limit
