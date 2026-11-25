"""Check that configured defects are gone (or much weaker) in a corrected CFA, with the existing detector scoring."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

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


def _detected(frame: CFAFrame, defect: Defect, detector: StatisticalDefectDetector) -> NDArray[np.bool_]:
    """Photosites of ``defect`` that the detector flags in ``frame`` (all false when it does not find it)."""
    height, width = frame.data.shape
    flagged = np.zeros((height, width), dtype=np.bool_)
    rows, cols = _window(defect, height, width)
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return flagged
    window = frame.with_data(np.ascontiguousarray(frame.data[rows, cols]))
    own = DefectMap((defect,)).mask(height, width)[rows, cols]
    win_h, win_w = own.shape
    for found in detector.detect([window]):
        if type(found.defect) is type(defect):
            flagged[rows, cols] |= DefectMap((found.defect,)).mask(win_h, win_w) & own
    return flagged


def _directions(defect: Defect) -> tuple[tuple[int, int], ...]:
    """Steps ``(dy, dx)`` to the same-colour neighbours a defect photosite is compared with."""
    step = CFA_TILE_SIZE
    if isinstance(defect, ColumnDefect):
        return ((0, -step), (0, step))
    if isinstance(defect, RowDefect):
        return ((-step, 0), (step, 0))
    return tuple((dy, dx) for dy in (-step, 0, step) for dx in (-step, 0, step) if (dy, dx) != (0, 0))


def _deviation(
    frame: CFAFrame, own: NDArray[np.bool_], region: NDArray[np.bool_], directions: tuple[tuple[int, int], ...]
) -> float:
    """Median absolute difference (DN) of the ``region`` photosites from their same-colour neighbours.

    Each photosite is compared with the median of its nearest same-colour photosites outside ``own``
    in each of ``directions`` (across the line for columns and rows, the eight around a pixel, as the
    detector does).
    """
    data = frame.data.astype(np.float64)
    height, width = own.shape
    ys, xs = np.nonzero(region)
    refs: list[NDArray[np.float64]] = []
    for dy, dx in directions:
        ry, rx = ys + dy, xs + dx
        while True:
            inside = (ry >= 0) & (ry < height) & (rx >= 0) & (rx < width)
            blocked = inside.copy()
            blocked[inside] = own[ry[inside], rx[inside]]
            if not blocked.any():
                break
            ry, rx = np.where(blocked, ry + dy, ry), np.where(blocked, rx + dx, rx)
        refs.append(np.where(inside, data[np.clip(ry, 0, height - 1), np.clip(rx, 0, width - 1)], np.nan))
    stacked = np.stack(refs)
    usable = ~np.isnan(stacked).all(axis=0)
    if not usable.any():
        return 0.0
    reference = np.nanmedian(stacked[:, usable], axis=0)
    return float(np.median(np.abs(data[ys, xs][usable] - reference)))


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
    is accepted (:attr:`DefectResidual.improved`) when the photosites the
    detector still flags deviate less from their same-colour neighbours than
    before correction (only those: a repaired part elsewhere on the line
    cannot hide an unrepaired segment); an unchanged, worse or newly appeared line is not.

    The deviation, not the detector score, is compared: the detector scores a
    band of adjacent lines by its strongest member, so a scene stripe or
    another defect merged into the band could supply both scores. The
    "before" frame puts only this defect's photosites back to their
    uncorrected values, so both deviations use the same neighbours.
    """
    residuals: list[DefectResidual] = []
    height, width = after.data.shape
    for defect in defects.defects:
        region = _detected(after, defect, detector)
        if region.any():
            own = DefectMap((defect,)).mask(height, width)
            isolated = after.with_data(np.where(own, before.data, after.data))
            steps = _directions(defect)
            residuals.append(
                DefectResidual(defect, _deviation(isolated, own, region, steps), _deviation(after, own, region, steps))
            )
    return residuals
