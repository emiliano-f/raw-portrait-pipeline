from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
from PIL import ExifTags, Image

from raw_pipeline.core.values import ExifInfo
from raw_pipeline.exporters.exif import ascii_text, exif_fields, jpeg_exif_bytes
from tests.unit.exporters.helpers import CANON_EXIF


def _read_back(exif_bytes: bytes) -> Image.Exif:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format="JPEG", exif=exif_bytes)
    buf.seek(0)
    with Image.open(buf) as img:
        return img.getexif()


def test_descriptive_fields_round_trip() -> None:
    exif = _read_back(jpeg_exif_bytes(CANON_EXIF, keep_gps=False))
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    base = ExifTags.Base
    assert exif[base.Make] == "Canon"
    assert exif[base.Model] == "Canon EOS 800D"
    assert exif[base.DateTime] == "2026:03:14 15:09:26"
    assert sub[base.DateTimeOriginal] == "2026:03:14 15:09:26"
    assert sub[base.LensModel] == "EF-S18-55mm f/4-5.6 IS STM"
    assert sub[base.ISOSpeedRatings] == 200
    assert float(sub[base.ExposureTime]) == 1 / 250
    assert float(sub[base.FNumber]) == 5.6
    assert float(sub[base.FocalLength]) == 55.0


def test_orientation_is_upright_because_pixels_are_already_rotated() -> None:
    assert _read_back(jpeg_exif_bytes(CANON_EXIF, keep_gps=False))[ExifTags.Base.Orientation] == 1


def test_gps_is_stripped_by_default() -> None:
    exif = _read_back(jpeg_exif_bytes(CANON_EXIF, keep_gps=False))
    assert ExifTags.IFD.GPSInfo not in exif
    assert not exif.get_ifd(ExifTags.IFD.GPSInfo)


def test_gps_kept_on_request_and_unknown_tags_skipped() -> None:
    info = ExifInfo(gps={"GPSLatitudeRef": "S", "GPSAltitude": 1250.5, "NotATag": 1, "GPSSpeed": None, "2": 24.5})
    gps = _read_back(jpeg_exif_bytes(info, keep_gps=True)).get_ifd(ExifTags.IFD.GPSInfo)
    assert gps[ExifTags.GPS.GPSLatitudeRef] == "S"
    assert float(gps[ExifTags.GPS.GPSAltitude]) == 1250.5
    assert float(gps[ExifTags.GPS.GPSLatitude]) == 24.5
    assert set(gps) == {ExifTags.GPS.GPSLatitudeRef, ExifTags.GPS.GPSAltitude, ExifTags.GPS.GPSLatitude}


def test_missing_fields_are_skipped() -> None:
    fields = exif_fields(ExifInfo(iso=100))
    assert {f.tag for f in fields} == {ExifTags.Base.Orientation, ExifTags.Base.ISOSpeedRatings}


def test_rationals_and_ascii() -> None:
    exposure = next(f for f in exif_fields(CANON_EXIF) if f.tag == ExifTags.Base.ExposureTime)
    assert exposure.value == Fraction(1, 250)
    assert ascii_text("Señal") == "Se?al"


def test_gps_read_back_from_raw_metadata_survives_jpeg_export() -> None:
    info = ExifInfo(gps={"GPSLatitudeRef": "S", "GPSLatitude": "24.0,11.0,30.5", "GPSAltitude": "1250.5"})
    gps = _read_back(jpeg_exif_bytes(info, keep_gps=True)).get_ifd(ExifTags.IFD.GPSInfo)
    assert [float(v) for v in gps[ExifTags.GPS.GPSLatitude]] == [24.0, 11.0, 30.5]
    assert float(gps[ExifTags.GPS.GPSAltitude]) == 1250.5
