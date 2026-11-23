from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image, TiffImagePlugin

from raw_pipeline.core.values import ExifInfo
from raw_pipeline.io.exif import ExifReader

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SAMPLES = sorted(FIXTURES.glob("*.CR2"))


def write_jpeg(path: Path, *, gps: bool = False, orientation: int = 6) -> Path:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Canon"
    exif[ExifTags.Base.Model] = "Canon EOS 800D"
    exif[ExifTags.Base.Orientation] = orientation
    exif[ExifTags.Base.DateTime] = "2024:01:02 03:04:05"
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    sub[ExifTags.Base.LensModel] = "EF50mm f/1.8 STM"
    sub[ExifTags.Base.DateTimeOriginal] = "2024:05:06 07:08:09"
    sub[ExifTags.Base.ISOSpeedRatings] = 400
    sub[ExifTags.Base.ExposureTime] = TiffImagePlugin.IFDRational(1, 200)
    sub[ExifTags.Base.FNumber] = TiffImagePlugin.IFDRational(18, 10)
    sub[ExifTags.Base.FocalLength] = TiffImagePlugin.IFDRational(50, 1)
    if gps:
        exif.get_ifd(ExifTags.IFD.GPSInfo)[ExifTags.GPS.GPSLatitudeRef] = "S"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, format="JPEG", exif=exif)
    return path


def test_reads_camera_metadata(tmp_path: Path) -> None:
    path = write_jpeg(tmp_path / "a.jpg")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = ExifReader().read(path)
    assert info.camera_make == "Canon"
    assert info.camera_model == "Canon EOS 800D"
    assert info.lens_model == "EF50mm f/1.8 STM"
    assert info.captured_at == datetime(2024, 5, 6, 7, 8, 9)
    assert info.iso == 400
    assert info.exposure_time_s == pytest.approx(1 / 200)
    assert info.f_number == pytest.approx(1.8)
    assert info.focal_length_mm == pytest.approx(50.0)
    assert info.orientation == 6
    assert info.gps is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_gps_presence(tmp_path: Path) -> None:
    info = ExifReader().read(write_jpeg(tmp_path / "a.jpg", gps=True))
    assert info.gps == {"GPSLatitudeRef": "S"}


def test_invalid_orientation_is_none(tmp_path: Path) -> None:
    assert ExifReader().read(write_jpeg(tmp_path / "a.jpg", orientation=42)).orientation is None


def test_unreadable_file_gives_empty_metadata(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "junk.CR2"
    path.write_bytes(b"junk")
    assert ExifReader().read(path) == ExifInfo()
    assert "cannot read EXIF" in caplog.text


def test_file_without_exif(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    assert ExifReader().read(path) == ExifInfo()


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLES, reason="no tests/fixtures/*.CR2")
def test_real_cr2() -> None:
    info = ExifReader().read(SAMPLES[0])
    assert info.camera_make == "Canon"
    assert info.iso is not None
    assert info.captured_at is not None
