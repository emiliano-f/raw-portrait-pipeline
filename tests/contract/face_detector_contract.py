"""Contract every :class:`~raw_pipeline.core.protocols.FaceDetector` must satisfy."""

from __future__ import annotations

import numpy as np

from raw_pipeline.core.protocols import FaceDetector


class FaceDetectorContract:
    """Reusable suite. Override ``make_detector``."""

    def make_detector(self) -> FaceDetector:
        raise NotImplementedError

    def test_blank_preview_has_no_faces(self) -> None:
        preview = np.full((240, 320, 3), 128, dtype=np.uint8)
        assert self.make_detector().detect(preview) == []

    def test_does_not_mutate_preview(self) -> None:
        rng = np.random.default_rng(0)
        preview = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        before = preview.copy()
        self.make_detector().detect(preview)
        np.testing.assert_array_equal(preview, before)

    def test_faces_inside_preview(self) -> None:
        rng = np.random.default_rng(1)
        preview = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        for face in self.make_detector().detect(preview):
            assert 0 <= face.box.x0 <= face.box.x1 <= 320
            assert 0 <= face.box.y0 <= face.box.y1 <= 240
            assert 0.0 <= face.confidence <= 1.0
