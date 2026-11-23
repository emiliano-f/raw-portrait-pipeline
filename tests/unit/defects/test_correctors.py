from __future__ import annotations

import numpy as np
import pytest

from raw_pipeline.core.errors import DefectCorrectionError
from raw_pipeline.core.protocols import DefectCorrector
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, Defect, DefectMap, PixelDefect, RowDefect
from raw_pipeline.defects import DirectionalCorrector, NeighborAverageCorrector, NullDefectCorrector
from tests.support.synthetic import RGGB, cfa_frame, inject_hot

SIZE = 64
NOISE_TOLERANCE = 120  # 6 sigma of the synthetic noise (20 raw units)

CORRECTORS = [NeighborAverageCorrector(), DirectionalCorrector()]
SCENARIOS: dict[str, tuple[Defect, ...]] = {
    "vertical": (ColumnDefect(index=31),),
    "horizontal": (RowDefect(index=20),),
    "partial_column": (ColumnDefect(index=10, start=8, end=40),),
    "partial_row": (RowDefect(index=33, start=5, end=50),),
    "width_2": (ColumnDefect(index=30, width=2),),
    "width_3_row": (RowDefect(index=12, width=3),),
    "border_column": (ColumnDefect(index=0), ColumnDefect(index=SIZE - 1)),
    "border_row": (RowDefect(index=0), RowDefect(index=SIZE - 1)),
    "pixels": (PixelDefect(x=5, y=7), PixelDefect(x=0, y=0), PixelDefect(x=63, y=63), PixelDefect(x=6, y=7)),
    "crossing_lines": (ColumnDefect(index=25), RowDefect(index=40), PixelDefect(x=25, y=40)),
}


def _error(result: CFAFrame, clean: CFAFrame, mask: np.ndarray) -> np.ndarray:
    return np.abs(result.data[mask].astype(np.int64) - clean.data[mask].astype(np.int64))


def _diagonal_edge_frame(low: int = 3000, high: int = 11000) -> CFAFrame:
    """Flat frame with a sharp 45-degree step edge (bright where x - y > 0)."""
    base = cfa_frame(SIZE, SIZE, level=low)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    gain = np.asarray([1.0, 0.6, 1.0, 0.6])[np.tile(np.asarray(RGGB), (SIZE // 2, SIZE // 2))]
    data = base.data.astype(np.float64) + np.where(xx - yy > 0, (high - low) * gain, 0.0)
    return base.with_data(np.clip(data, 0, 65535).astype(np.uint16))


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=str)
def test_hot_defect_is_removed_and_rest_untouched(corrector: DefectCorrector, scenario: str) -> None:
    defects = DefectMap(SCENARIOS[scenario])
    clean = cfa_frame(SIZE, SIZE, seed=3)
    mask = defects.mask(SIZE, SIZE)
    hot = inject_hot(clean, mask)
    result = corrector.correct(hot, defects)
    assert _error(result.frame, clean, mask).max() < NOISE_TOLERANCE
    np.testing.assert_array_equal(result.frame.data[~mask], hot.data[~mask])
    assert result.corrected_pixel_count == int(mask.sum())
    assert result.method == corrector.name


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_overlapping_entries_counted_once(corrector: DefectCorrector) -> None:
    defects = DefectMap((ColumnDefect(index=20, width=2), ColumnDefect(index=21), PixelDefect(x=20, y=3)))
    result = corrector.correct(inject_hot(cfa_frame(), defects.mask(SIZE, SIZE)), defects)
    assert result.corrected_pixel_count == 2 * SIZE


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_vertical_edge_crossing_a_row_stays_sharp(corrector: DefectCorrector) -> None:
    clean = cfa_frame(SIZE, SIZE, edge_at_x=32)
    defects = DefectMap((RowDefect(index=30),))
    mask = defects.mask(SIZE, SIZE)
    result = corrector.correct(inject_hot(clean, mask), defects)
    assert _error(result.frame, clean, mask).max() < NOISE_TOLERANCE


def test_directional_keeps_a_diagonal_edge_that_neighbor_average_blurs() -> None:
    clean = _diagonal_edge_frame()
    defects = DefectMap((ColumnDefect(index=32), RowDefect(index=17, start=20, end=60)))
    mask = defects.mask(SIZE, SIZE)
    hot = inject_hot(clean, mask)
    directional = _error(DirectionalCorrector().correct(hot, defects).frame, clean, mask).max()
    average = _error(NeighborAverageCorrector().correct(hot, defects).frame, clean, mask).max()
    assert directional < NOISE_TOLERANCE
    assert average > 1000
    assert directional * 10 < average


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_uint16_range_respected(corrector: DefectCorrector) -> None:
    data = np.zeros((SIZE, SIZE), dtype=np.uint16)
    data[:, ::2] = 65535  # saturated even columns, zero odd columns: same-colour planes are flat
    frame = CFAFrame(data=data, pattern=RGGB, color_desc="RGBG", black_level=(0,) * 4, white_level=65535)
    defects = DefectMap((RowDefect(index=10), PixelDefect(x=4, y=40)))
    out = corrector.correct(frame, defects).frame
    assert out.data.dtype == np.uint16
    np.testing.assert_array_equal(out.data, data)


def test_none_is_true_passthrough_even_with_invalid_map() -> None:
    frame = inject_hot(cfa_frame(), DefectMap((ColumnDefect(index=3),)).mask(SIZE, SIZE))
    result = NullDefectCorrector().correct(frame, DefectMap((ColumnDefect(index=10_000),)))
    assert result.frame is frame
    assert result.corrected_pixel_count == 0
    assert result.method == "none"


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_defect_outside_sensor_fails(corrector: DefectCorrector) -> None:
    with pytest.raises(DefectCorrectionError, match="outside"):
        corrector.correct(cfa_frame(), DefectMap((PixelDefect(x=SIZE, y=0),)))


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_band_too_wide_to_interpolate_fails(corrector: DefectCorrector) -> None:
    with pytest.raises(DefectCorrectionError, match="no valid same-colour neighbour"):
        corrector.correct(cfa_frame(), DefectMap((ColumnDefect(index=0, width=40),)))


@pytest.mark.parametrize("corrector", CORRECTORS, ids=lambda c: c.name)
def test_partially_outside_entries_are_clipped(corrector: DefectCorrector) -> None:
    defects = DefectMap((ColumnDefect(index=SIZE - 1, width=4), RowDefect(index=5, start=60, end=200)))
    result = corrector.correct(inject_hot(cfa_frame(), defects.mask(SIZE, SIZE)), defects)
    assert result.corrected_pixel_count == int(defects.mask(SIZE, SIZE).sum())
