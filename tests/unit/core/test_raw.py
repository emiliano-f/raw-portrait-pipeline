from __future__ import annotations

import numpy as np
import pytest

from raw_pipeline.core.raw import CFAFrame, ColumnDefect, DefectMap, PixelDefect, RowDefect
from tests.support.synthetic import RGGB, cfa_frame


def test_cfa_validates_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        CFAFrame(np.zeros((2, 2, 1), np.uint16), RGGB, "RGBG", (0, 0, 0, 0), 100)
    with pytest.raises(ValueError, match="uint16"):
        CFAFrame(np.zeros((2, 2), np.float32), RGGB, "RGBG", (0, 0, 0, 0), 100)  # type: ignore[arg-type]


def test_color_index_map_tiles_pattern_with_odd_size() -> None:
    frame = CFAFrame(np.zeros((3, 5), np.uint16), RGGB, "RGBG", (0, 0, 0, 0), 100)
    cmap = frame.color_index_map()
    assert cmap.shape == (3, 5)
    assert cmap[0, 0] == 0
    assert cmap[0, 1] == 1
    assert cmap[1, 0] == 3
    assert cmap[1, 1] == 2
    assert cmap[2, 4] == 0
    assert (frame.height, frame.width) == (3, 5)


def test_with_data_keeps_layout() -> None:
    frame = cfa_frame(8, 8)
    new = frame.with_data(np.zeros((8, 8), np.uint16))
    assert new.pattern == frame.pattern
    assert new.white_level == frame.white_level
    assert frame.data.any()


def test_mask_full_and_partial_lines() -> None:
    dmap = DefectMap(
        (
            ColumnDefect(index=3, width=2),
            RowDefect(index=5, start=1, end=4),
            PixelDefect(x=0, y=0),
            ColumnDefect(index=7, start=2, end=4),
        )
    )
    mask = dmap.mask(8, 9)
    assert mask[:, 3:5].all()
    assert mask[5, 1:4].all()
    assert not mask[5, 0]
    assert not mask[5, 5]
    assert mask[0, 0]
    assert mask[2:4, 7].all()
    assert not mask[1, 7]
    assert not mask[4, 7]
    assert int(mask.sum()) == 8 * 2 + (3 - 1) + 1 + 2  # row segment overlaps column 3
    assert len(dmap) == 4
    assert list(dmap)[2] == PixelDefect(0, 0)


def test_mask_clips_and_rejects_outside() -> None:
    assert DefectMap((RowDefect(index=7, width=5),)).mask(8, 4)[7].all()
    with pytest.raises(ValueError, match="outside"):
        DefectMap((ColumnDefect(index=50),)).mask(8, 8)
    with pytest.raises(ValueError, match="outside"):
        DefectMap((PixelDefect(x=1, y=20),)).mask(8, 8)
