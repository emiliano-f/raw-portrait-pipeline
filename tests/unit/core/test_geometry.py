from __future__ import annotations

import numpy as np

from raw_pipeline.core.geometry import PreviewGeometry, preview_to_full
from raw_pipeline.core.values import BoundingBox, FaceRegion


def test_for_long_edge_landscape_and_portrait() -> None:
    g = PreviewGeometry.for_long_edge(6000, 4000, 1024)
    assert (g.preview_width, g.preview_height) == (1024, 683)
    p = PreviewGeometry.for_long_edge(4000, 6000, 1024)
    assert (p.preview_width, p.preview_height) == (683, 1024)


def test_never_upscales() -> None:
    g = PreviewGeometry.for_long_edge(300, 200, 1024)
    assert (g.preview_width, g.preview_height) == (300, 200)
    assert g.scale_x == 1.0


def test_preview_to_full_scales_box_and_landmarks() -> None:
    g = PreviewGeometry(full_width=2000, full_height=1000, preview_width=1000, preview_height=250)
    face = FaceRegion(
        BoundingBox(100, 50, 200, 100),
        0.9,
        "fake",
        landmarks=np.asarray([[100, 50], [150, 75]], dtype=np.float32),
        landmark_scheme="fake",
    )
    full = preview_to_full(face, g)
    assert full.box == BoundingBox(200, 200, 400, 400)
    assert full.landmarks is not None
    np.testing.assert_allclose(full.landmarks, [[200, 200], [300, 300]])
    assert full.landmarks.dtype == np.float32
    assert full.detector == "fake"


def test_preview_to_full_clips_to_frame() -> None:
    g = PreviewGeometry(full_width=200, full_height=100, preview_width=100, preview_height=50)
    full = preview_to_full(FaceRegion(BoundingBox(90, 40, 120, 60), 0.5, "fake"), g)
    assert full.box == BoundingBox(180, 80, 200, 100)
    assert full.landmarks is None
