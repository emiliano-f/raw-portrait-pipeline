from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from raw_pipeline.core.values import BoundingBox, ExifInfo, FaceRegion, ImageMetadata, freeze_record


def test_freeze_record_is_read_only_and_sorted() -> None:
    rec = freeze_record({"b": 1, "a": 2})
    assert list(rec) == ["a", "b"]
    with pytest.raises(TypeError):
        rec["c"] = 3  # type: ignore[index]


def test_metadata_with_record_returns_new_instance() -> None:
    meta = ImageMetadata()
    new = meta.with_record("tone", {"ev": 0.5})
    assert meta.records == {}
    assert new.records["tone"]["ev"] == 0.5
    assert new.with_record("tone", {"ev": 1.0}).records["tone"]["ev"] == 1.0


def test_metadata_flatten_includes_exif_and_records() -> None:
    exif = ExifInfo(camera_model="Canon EOS 800D", iso=100, captured_at=datetime(2026, 1, 2, 3, 4, 5))
    flat = ImageMetadata().with_exif(exif).with_record("meter", {"correction_ev": 0.7}).flatten()
    assert flat["exif.camera_model"] == "Canon EOS 800D"
    assert flat["exif.captured_at"] == "2026-01-02T03:04:05"
    assert flat["exif.has_gps"] is False
    assert flat["meter.correction_ev"] == 0.7


def test_exif_gps_never_in_record_values() -> None:
    rec = ExifInfo(gps={"GPSLatitude": "1"}).as_record()
    assert rec["has_gps"] is True
    assert "GPSLatitude" not in str(dict(rec))


def test_bounding_box_geometry() -> None:
    box = BoundingBox(10, 20, 30, 60)
    assert (box.width, box.height, box.area) == (20, 40, 800)
    assert box.scaled(2, 0.5) == BoundingBox(20, 10, 60, 30)
    assert BoundingBox(-5, -5, 500, 50).clipped(100, 40) == BoundingBox(0, 0, 100, 40)


def test_bounding_box_rejects_inverted() -> None:
    with pytest.raises(ValueError, match="invalid box"):
        BoundingBox(10, 0, 5, 1)


def test_face_region_defaults() -> None:
    face = FaceRegion(BoundingBox(0, 0, 1, 1), 0.9, "fake")
    assert face.landmarks is None
    assert face.landmark_scheme is None
    assert np.float32(face.confidence) == np.float32(0.9)
