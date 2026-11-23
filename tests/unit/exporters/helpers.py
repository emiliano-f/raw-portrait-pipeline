"""Shared builders for the export-report tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from raw_pipeline.config import parse_config
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import ExifInfo

CANON_EXIF = ExifInfo(
    camera_make="Canon",
    camera_model="Canon EOS 800D",
    lens_model="EF-S18-55mm f/4-5.6 IS STM",
    captured_at=datetime(2026, 3, 14, 15, 9, 26),
    iso=200,
    exposure_time_s=1 / 250,
    f_number=5.6,
    focal_length_mm=55.0,
    orientation=6,
    gps={"GPSLatitudeRef": "S", "GPSLatitude": 24.18, "GPSAltitude": 1250.5},
)


def build_env(tmp_path: Path, *, overwrite: bool = False, **sections: dict[str, Any]) -> BuildEnv:
    return BuildEnv(config=parse_config(sections), output_dir=tmp_path, overwrite=overwrite)
