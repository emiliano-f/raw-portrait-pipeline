"""``residual_defects``: scene edges and thin stripes aligned with a corrected line."""

from __future__ import annotations

import numpy as np
import pytest

from raw_pipeline.core.protocols import DefectCorrector
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, Defect, DefectMap, RowDefect
from raw_pipeline.defects import DirectionalCorrector, NeighborAverageCorrector
from raw_pipeline.defects.detection import StatisticalDefectDetector
from raw_pipeline.defects.verification import residual_defects
from tests.support.synthetic import RGGB, cfa_frame, inject_hot

SIZE = 256
LINE = 128
LOW, HIGH = 2000, 12000
CORRECTORS = [NeighborAverageCorrector(), DirectionalCorrector()]


def _step(*, vertical: bool, at: int) -> CFAFrame:
    """Noisy frame with a sharp step edge: bright from column (or row) ``at`` on."""
    base = cfa_frame(SIZE, SIZE, level=LOW, seed=5)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    gain = np.asarray([1.0, 0.6, 1.0, 0.6])[np.tile(np.asarray(RGGB), (SIZE // 2, SIZE // 2))]
    bright = (xx if vertical else yy) >= at
    data = base.data.astype(np.float64) + np.where(bright, (HIGH - LOW) * gain, 0.0)
    return base.with_data(np.clip(data, 0, 65535).astype(np.uint16))


def _defect(*, vertical: bool, width: int) -> Defect:
    return ColumnDefect(index=LINE, width=width) if vertical else RowDefect(index=LINE, width=width)


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
@pytest.mark.parametrize("vertical", [True, False], ids=["column", "row"])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("edge_shift", [-1, 0, 1, 2], ids=lambda s: f"edge{s:+d}")
def test_edge_along_corrected_line_passes(
    corrector: DefectCorrector, vertical: bool, width: int, edge_shift: int
) -> None:
    defects = DefectMap((_defect(vertical=vertical, width=width),))
    clean = _step(vertical=vertical, at=LINE + edge_shift)
    hot = inject_hot(clean, defects.mask(SIZE, SIZE))
    detector = StatisticalDefectDetector()
    assert residual_defects(hot, hot, defects, detector), "the hot line must be detected before correction"
    fixed = corrector.correct(hot, defects).frame
    assert residual_defects(hot, fixed, defects, detector) == []


def _stripe(*, vertical: bool, at: int, width: int) -> CFAFrame:
    """Noisy frame with a thin bright scene stripe ``width`` photosites wide starting at ``at``."""
    base = cfa_frame(SIZE, SIZE, level=LOW, seed=7)
    coord = np.mgrid[:SIZE, :SIZE][1 if vertical else 0]
    data = base.data.astype(np.float64) + np.where((coord >= at) & (coord < at + width), HIGH - LOW, 0.0)
    return base.with_data(np.clip(data, 0, 65535).astype(np.uint16))


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
@pytest.mark.parametrize("vertical", [True, False], ids=["column", "row"])
@pytest.mark.parametrize("offset", [-2, -1, 1, 2, 3])
@pytest.mark.parametrize("stripe_width", [1, 2])
def test_thin_scene_stripe_next_to_corrected_line_passes(
    corrector: DefectCorrector, vertical: bool, offset: int, stripe_width: int
) -> None:
    defects = DefectMap((_defect(vertical=vertical, width=2),))
    clean = _stripe(vertical=vertical, at=LINE + offset if offset < 0 else LINE + 1 + offset, width=stripe_width)
    hot = inject_hot(clean, defects.mask(SIZE, SIZE))
    fixed = corrector.correct(hot, defects).frame
    residuals = residual_defects(hot, fixed, defects, StatisticalDefectDetector())
    assert all(r.improved for r in residuals), residuals


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_pixels_in_fine_texture_pass(corrector: DefectCorrector) -> None:
    from raw_pipeline.core.raw import PixelDefect  # noqa: PLC0415

    base = cfa_frame(SIZE, SIZE, level=LOW, seed=9)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    texture = (HIGH - LOW) / 2 * (1 + np.sin(2 * np.pi * (xx + yy) / 6.0))
    clean = base.with_data(np.clip(base.data + texture, 0, 65535).astype(np.uint16))
    defects = DefectMap(tuple(PixelDefect(x=x, y=y) for x, y in [(40, 50), (101, 77), (128, 128), (200, 31)]))
    hot = inject_hot(clean, defects.mask(SIZE, SIZE))
    fixed = corrector.correct(hot, defects).frame
    assert residual_defects(hot, fixed, defects, StatisticalDefectDetector()) == []


@pytest.mark.parametrize("vertical", [True, False], ids=["column", "row"])
def test_uncorrected_line_is_not_improved(vertical: bool) -> None:
    defects = DefectMap((_defect(vertical=vertical, width=2),))
    hot = inject_hot(_stripe(vertical=vertical, at=LINE - 2, width=1), defects.mask(SIZE, SIZE))
    residuals = residual_defects(hot, hot, defects, StatisticalDefectDetector())
    assert residuals
    assert not any(r.improved for r in residuals)


def test_line_introduced_by_correction_is_not_improved() -> None:
    defects = DefectMap((_defect(vertical=True, width=1),))
    clean = cfa_frame(SIZE, SIZE, level=LOW, seed=11)
    broken = inject_hot(clean, defects.mask(SIZE, SIZE))
    [residual] = residual_defects(clean, broken, defects, StatisticalDefectDetector())
    assert residual.score_before == 0
    assert not residual.improved
