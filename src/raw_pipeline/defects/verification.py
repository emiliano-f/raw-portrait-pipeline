"""Check that configured defects are gone from a corrected CFA, with the existing detector scoring."""

from __future__ import annotations

from typing import Final

import numpy as np

from raw_pipeline.core.raw import CFA_TILE_SIZE, CFAFrame, ColumnDefect, Defect, DefectMap, DetectedDefect, RowDefect
from raw_pipeline.defects.detection import StatisticalDefectDetector

WINDOW_MARGIN_PX: Final = 32
"""Photosites kept on each side of a defect when re-scoring it (a multiple of the CFA tile)."""


def _even_floor(value: int) -> int:
    return value - value % CFA_TILE_SIZE


def _window(defect: Defect, height: int, width: int) -> tuple[slice, slice]:
    mask = DefectMap((defect,)).mask(height, width)
    ys, xs = np.nonzero(mask)
    if isinstance(defect, ColumnDefect):
        rows = slice(0, height)
    else:
        rows = slice(
            _even_floor(max(0, int(ys.min()) - WINDOW_MARGIN_PX)), min(height, int(ys.max()) + 1 + WINDOW_MARGIN_PX)
        )
    if isinstance(defect, RowDefect):
        cols = slice(0, width)
    else:
        cols = slice(
            _even_floor(max(0, int(xs.min()) - WINDOW_MARGIN_PX)), min(width, int(xs.max()) + 1 + WINDOW_MARGIN_PX)
        )
    return rows, cols


def remaining_defects(
    frame: CFAFrame, defects: DefectMap, detector: StatisticalDefectDetector
) -> list[tuple[Defect, DetectedDefect]]:
    """Configured defects the detector still finds in ``frame``, with the detection that overlaps each.

    Each defect is re-scored in a window around it (windows start on even
    rows/columns, so the CFA pattern is unchanged); only detections that
    overlap the defect's own photosites *and* have the same kind (a column
    for a column, ...) count, so scene edges crossing the window and single
    textured photosites along a corrected line are ignored.
    """
    height, width = frame.data.shape
    remaining: list[tuple[Defect, DetectedDefect]] = []
    for defect in defects.defects:
        rows, cols = _window(defect, height, width)
        if rows.start >= rows.stop or cols.start >= cols.stop:
            continue
        window = frame.with_data(np.ascontiguousarray(frame.data[rows, cols]))
        own = DefectMap((defect,)).mask(height, width)[rows, cols]
        win_h, win_w = own.shape
        for found in detector.detect([window]):
            same_kind = type(found.defect) is type(defect)
            if same_kind and (DefectMap((found.defect,)).mask(win_h, win_w) & own).any():
                remaining.append((defect, found))
                break
    return remaining
