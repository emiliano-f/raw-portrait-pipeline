"""Contract every :class:`~raw_pipeline.core.protocols.DefectCorrector` must satisfy."""

from __future__ import annotations

import numpy as np

from raw_pipeline.core.protocols import DefectCorrector
from raw_pipeline.core.raw import ColumnDefect, DefectMap, PixelDefect, RowDefect
from tests.support.synthetic import cfa_frame, inject_hot

DEFECTS = DefectMap((ColumnDefect(index=21, width=1), RowDefect(index=40, start=5, end=50), PixelDefect(x=7, y=9)))


class DefectCorrectorContract:
    """Reusable suite. Override ``make_corrector``; set ``is_passthrough`` for ``none``."""

    is_passthrough: bool = False

    def make_corrector(self) -> DefectCorrector:
        raise NotImplementedError

    def test_name(self) -> None:
        assert self.make_corrector().name

    def test_does_not_mutate_input(self) -> None:
        frame = inject_hot(cfa_frame(), DEFECTS.mask(64, 64))
        before = frame.data.copy()
        self.make_corrector().correct(frame, DEFECTS)
        np.testing.assert_array_equal(frame.data, before)

    def test_preserves_layout(self) -> None:
        frame = inject_hot(cfa_frame(), DEFECTS.mask(64, 64))
        out = self.make_corrector().correct(frame, DEFECTS).frame
        assert out.data.shape == frame.data.shape
        assert out.data.dtype == np.uint16
        assert out.pattern == frame.pattern
        assert out.black_level == frame.black_level
        assert out.white_level == frame.white_level

    def test_only_masked_pixels_change(self) -> None:
        mask = DEFECTS.mask(64, 64)
        frame = inject_hot(cfa_frame(), mask)
        result = self.make_corrector().correct(frame, DEFECTS)
        np.testing.assert_array_equal(result.frame.data[~mask], frame.data[~mask])
        changed = int(np.count_nonzero(result.frame.data != frame.data))
        assert changed <= result.corrected_pixel_count <= int(mask.sum())

    def test_empty_map_is_identity(self) -> None:
        frame = cfa_frame()
        result = self.make_corrector().correct(frame, DefectMap())
        np.testing.assert_array_equal(result.frame.data, frame.data)
        assert result.corrected_pixel_count == 0

    def test_removes_hot_line(self) -> None:
        clean = cfa_frame()
        mask = DEFECTS.mask(64, 64)
        result = self.make_corrector().correct(inject_hot(clean, mask), DEFECTS)
        if self.is_passthrough:
            assert result.corrected_pixel_count == 0
            return
        error = np.abs(result.frame.data[mask].astype(np.int64) - clean.data[mask].astype(np.int64))
        assert error.max() < 200  # noise sigma is 20 raw units

    def test_deterministic(self) -> None:
        frame = inject_hot(cfa_frame(), DEFECTS.mask(64, 64))
        a = self.make_corrector().correct(frame, DEFECTS)
        b = self.make_corrector().correct(frame, DEFECTS)
        np.testing.assert_array_equal(a.frame.data, b.frame.data)
