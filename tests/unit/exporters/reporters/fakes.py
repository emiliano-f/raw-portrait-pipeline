"""Fake outcomes for reporter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from raw_pipeline.core.protocols import ImageOutcome
from raw_pipeline.core.values import ImageMetadata
from tests.unit.exporters.helpers import CANON_EXIF


def ok_outcome(name: str, *, ev: float = 0.5, crops: bool = False, seed: int = 0) -> ImageOutcome:
    metadata = (
        ImageMetadata(exif=CANON_EXIF)
        .with_record("decode", {"demosaic": "ahd"})
        .with_record("defects", {"method": "directional", "corrected_pixel_count": 3000})
        .with_record("meter", {"method": "face_weighted", "correction_ev": ev, "faces_used": 1})
        .with_record("qa", {"flags": "", "skipped": True})
    )
    rng = np.random.default_rng(seed)
    thumbs = {
        "before": rng.integers(0, 255, (32, 48, 3), dtype=np.uint8),
        "after": rng.integers(0, 255, (48, 32, 3), dtype=np.uint8),
    }
    if crops:
        thumbs |= {
            "defect_before": np.full((20, 20, 3), 250, np.uint8),
            "defect_after": np.zeros((20, 20, 3), np.uint8),
        }
    return ImageOutcome(Path("in") / name, metadata, (Path("out") / f"{Path(name).stem}.jpg",), thumbs)


def failed_outcome(name: str) -> ImageOutcome:
    metadata = ImageMetadata().with_record("decode", {"demosaic": "ahd"})
    return ImageOutcome(Path("in") / name, metadata, error="truncated file", error_type="DecodeError")
