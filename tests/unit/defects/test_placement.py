from __future__ import annotations

import numpy as np
import pytest

from raw_pipeline.core.dng import CFAOffset, DngRawImage
from raw_pipeline.core.errors import CoordinateMappingError
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, DefectMap
from raw_pipeline.defects.placement import find_offset, stored_coordinates, visible_frame
from tests.support.dng_builder import raw_frame

RGGB = ((0, 1), (3, 2))


def _raw(
    active: tuple[int, int, int, int] | None = (2, 4, 48, 64), crop: tuple[int, int] | None = None, **kw: object
) -> DngRawImage:
    return DngRawImage(raw_frame(), "RGGB", active, crop, **kw)  # type: ignore[arg-type]


def _reference(raw: DngRawImage, y: int, x: int, pattern: tuple[tuple[int, int], tuple[int, int]] = RGGB) -> CFAFrame:
    data = raw.data[y : y + 40, x : x + 50].copy()
    return CFAFrame(data, pattern, "RGBG", (0, 0, 0, 0), 16383)


@pytest.mark.parametrize(
    ("raw", "y", "x", "method"),
    [
        (_raw(), 2, 4, "active_area"),
        (_raw(crop=(2, 2)), 4, 6, "default_crop"),
        (_raw(active=None), 0, 0, "origin"),
        (_raw(), 6, 8, "search"),
    ],
)
def test_finds_exact_offset(raw: DngRawImage, y: int, x: int, method: str) -> None:
    placement = find_offset(_reference(raw, y, x), raw)
    assert placement.offset == CFAOffset(y, x)
    assert placement.method == method


def test_search_rejects_wrong_bayer_phase() -> None:
    raw = _raw()
    # Data really sits at an odd row, where the DNG phase is GBRG, but the reference claims RGGB.
    with pytest.raises(CoordinateMappingError):
        find_offset(_reference(raw, 3, 4), raw)


def test_wrong_phase_accepted_when_pattern_matches() -> None:
    raw = _raw()
    assert find_offset(_reference(raw, 3, 4, ((3, 2), (0, 1))), raw).offset == CFAOffset(3, 4)


def test_no_match_fails() -> None:
    raw = _raw()
    reference = CFAFrame(raw_frame(40, 50, seed=9), RGGB, "RGBG", (0,) * 4, 16383)
    with pytest.raises(CoordinateMappingError, match="no offset"):
        find_offset(reference, raw)


def test_ambiguous_match_fails() -> None:
    data = np.full((48, 64), 700, np.uint16)
    data[0] = 1  # the origin candidate fails, so the search runs and finds many offsets
    flat = DngRawImage(data, "RGGB", None)
    with pytest.raises(CoordinateMappingError, match="ambiguous"):
        find_offset(CFAFrame(np.full((40, 50), 700, np.uint16), RGGB, "RGBG", (0,) * 4, 16383), flat)


def test_reference_larger_than_dng_fails() -> None:
    raw = _raw()
    big = CFAFrame(raw_frame(60, 80), RGGB, "RGBG", (0,) * 4, 16383)
    with pytest.raises(CoordinateMappingError):
        find_offset(big, raw)


def test_visible_frame_uses_dng_levels() -> None:
    raw = _raw(black_level=(10, 11, 12, 13), white_level=12000)
    frame = visible_frame(_reference(raw, 2, 4), raw, CFAOffset(2, 4))
    np.testing.assert_array_equal(frame.data, raw.data[2:42, 4:54])
    assert frame.white_level == 12000
    assert frame.pattern == RGGB
    # DNG tile (R, G, G, B) -> LibRaw channels R=0, G=1, B=2, G2=3.
    assert frame.black_level == (10, 11, 13, 12)


def test_visible_frame_single_black_level() -> None:
    raw = _raw(black_level=(511,))
    assert visible_frame(_reference(raw, 2, 4), raw, CFAOffset(2, 4)).black_level == (511,) * 4


def test_stored_coordinates_shift_mask() -> None:
    mask = DefectMap((ColumnDefect(index=3),)).mask(5, 8)
    ys, xs = stored_coordinates(mask, CFAOffset(36, 264))
    assert set(xs.tolist()) == {267}
    assert ys.tolist() == [36, 37, 38, 39, 40]
