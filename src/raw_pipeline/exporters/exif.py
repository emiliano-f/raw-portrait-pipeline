"""EXIF fields written to outputs, derived from the source :class:`ExifInfo`.

Only descriptive camera metadata is copied (camera, lens, capture date, ISO,
shutter, aperture, focal length). Orientation is always written as 1
("top-left") because the decoder already rotated the pixels. GPS tags are
copied only on request. Nothing time-dependent (e.g. "now") is ever written,
so outputs stay deterministic.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Final

from PIL import ExifTags, Image, TiffImagePlugin

from raw_pipeline.core.values import ExifInfo, Scalar

logger = logging.getLogger(__name__)

EXIF_DATETIME_FORMAT: Final = "%Y:%m:%d %H:%M:%S"
RATIONAL_MAX_DENOMINATOR: Final = 1_000_000
UPRIGHT_ORIENTATION: Final = 1
"""EXIF orientation "top-left": pixels are stored the way they are displayed."""


class ExifIfd(Enum):
    """Directory an EXIF tag belongs to in a JPEG's APP1 segment."""

    IFD0 = "ifd0"
    EXIF = "exif"


ExifValue = str | int | Fraction


@dataclass(frozen=True)
class ExifField:
    """One tag to write.

    Attributes:
        tag: Numeric EXIF/TIFF tag id.
        ifd: Directory of the tag in JPEG files (TIFF files keep all in IFD0, TIFF/EP style).
        value: ASCII text, unsigned short, or unsigned rational.
    """

    tag: int
    ifd: ExifIfd
    value: ExifValue


def ascii_text(text: str) -> str:
    """Return ``text`` restricted to printable ASCII (EXIF ASCII tags allow nothing else)."""
    return text.encode("ascii", "replace").decode("ascii")


def _rational(value: float) -> Fraction:
    return Fraction(value).limit_denominator(RATIONAL_MAX_DENOMINATOR)


def exif_fields(info: ExifInfo) -> list[ExifField]:
    """Return the descriptive tags of ``info`` that are known (``None`` fields are skipped)."""
    base, exif = ExifTags.Base, ExifIfd.EXIF
    candidates: list[tuple[int, ExifIfd, ExifValue | None]] = [
        (base.Make, ExifIfd.IFD0, ascii_text(info.camera_make) if info.camera_make else None),
        (base.Model, ExifIfd.IFD0, ascii_text(info.camera_model) if info.camera_model else None),
        (base.Orientation, ExifIfd.IFD0, UPRIGHT_ORIENTATION),
        (base.LensModel, exif, ascii_text(info.lens_model) if info.lens_model else None),
        (base.ISOSpeedRatings, exif, info.iso),
        (base.ExposureTime, exif, _rational(info.exposure_time_s) if info.exposure_time_s else None),
        (base.FNumber, exif, _rational(info.f_number) if info.f_number else None),
        (base.FocalLength, exif, _rational(info.focal_length_mm) if info.focal_length_mm else None),
    ]
    if info.captured_at is not None:
        stamp = info.captured_at.strftime(EXIF_DATETIME_FORMAT)
        candidates += [(base.DateTime, ExifIfd.IFD0, stamp), (base.DateTimeOriginal, exif, stamp)]
    return [ExifField(tag, ifd, value) for tag, ifd, value in candidates if value is not None]


def _pillow_value(value: ExifValue | Scalar) -> object:
    if isinstance(value, Fraction):
        return TiffImagePlugin.IFDRational(value.numerator, value.denominator)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return _pillow_value(_rational(value))
    if isinstance(value, str):
        return ascii_text(value)
    return value


def _gps_tag(name: str) -> int | None:
    if name.isdigit():
        return int(name)
    member = ExifTags.GPS.__members__.get(name)
    return int(member) if member is not None else None


def _add_gps(exif: Image.Exif, gps: Mapping[str, Scalar]) -> None:
    ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    for name, value in sorted(gps.items()):
        tag = _gps_tag(name)
        if tag is None or value is None:
            logger.debug("skipping GPS tag %s", name)
            continue
        ifd[tag] = _pillow_value(value)


def jpeg_exif_bytes(info: ExifInfo, *, keep_gps: bool) -> bytes:
    """Encode the EXIF APP1 payload for a JPEG.

    Args:
        info: Source camera metadata.
        keep_gps: Copy ``info.gps`` (tag names as in :class:`PIL.ExifTags.GPS`,
            e.g. ``"GPSLatitude"``, or numeric ids) into a GPS IFD.
    """
    exif = Image.Exif()
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    for field in exif_fields(info):
        target = exif if field.ifd is ExifIfd.IFD0 else exif_ifd
        target[field.tag] = _pillow_value(field.value)
    if keep_gps and info.gps:
        _add_gps(exif, info.gps)
    return exif.tobytes()
