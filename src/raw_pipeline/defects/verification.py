"""Check that configured defects are gone (or much weaker) in a corrected CFA, with the existing detector scoring."""

from __future__ import annotations

from typing import Final

import numpy as np

from raw_pipeline.core.raw import CFA_TILE_SIZE, CFAFrame, ColumnDefect, Defect, DefectMap, DefectResidual, RowDefect
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


def _score(frame: CFAFrame, defect: Defect, detector: StatisticalDefectDetector) -> float:
    """Detector score of ``defect`` in ``frame``, or 0 when the detector does not find it."""
    height, width = frame.data.shape
    rows, cols = _window(defect, height, width)
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return 0.0
    window = frame.with_data(np.ascontiguousarray(frame.data[rows, cols]))
    own = DefectMap((defect,)).mask(height, width)[rows, cols]
    win_h, win_w = own.shape
    scores = [
        found.score
        for found in detector.detect([window])
        if type(found.defect) is type(defect) and (DefectMap((found.defect,)).mask(win_h, win_w) & own).any()
    ]
    return max(scores, default=0.0)


def residual_defects(
    before: CFAFrame,
    after: CFAFrame,
    defects: DefectMap,
    detector: StatisticalDefectDetector,
) -> list[DefectResidual]:
    """Configured defects the detector still finds in ``after``, scored against ``before``.

    Each defect is re-scored in a window around it (windows start on even
    rows/columns, so the CFA pattern is unchanged); only detections that
    overlap the defect's own photosites *and* have the same kind (a column
    for a column, ...) count, so scene edges crossing the window are ignored.

    A corrected line next to a thin scene stripe (a hair, a cable) takes part
    of the stripe's level from its same-colour neighbours (all its vertical
    neighbours are defective too) and can still score as a line, at up to
    about 0.9 of the uncorrected score. So the check is relative: a residual
    is accepted (:attr:`DefectResidual.improved`) when its score is lower
    than in the uncorrected frame; an unchanged or newly appeared line is not.

    The "before" score puts only this defect's photosites back to their
    uncorrected values (everything else as corrected), so a stronger adjacent
    line that the detector merged into one band cannot lend it its score.
    """
    residuals: list[DefectResidual] = []
    height, width = after.data.shape
    for defect in defects.defects:
        after_score = _score(after, defect, detector)
        if after_score > 0:
            own = DefectMap((defect,)).mask(height, width)
            isolated = after.with_data(np.where(own, before.data, after.data))
            residuals.append(DefectResidual(defect, _score(isolated, defect, detector), after_score))
    return residuals
