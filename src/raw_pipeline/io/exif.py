"""Camera metadata (EXIF) of RAW files, read with Pillow's TIFF parser.

Canon CR2 files are TIFF containers, so Pillow can read their IFDs without
decoding any pixels and without an extra dependency. The file is opened
read-only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Final

from PIL import ExifTags, Image, UnidentifiedImageError

from raw_pipeline.core.values import ExifInfo, Scalar

logger = logging.getLogger(__name__)

EXIF_DATETIME_FORMAT: Final = "%Y:%m:%d %H:%M:%S"
_GPS_VERSION_TAG: Final = 0
_ROTATION_TAGS: Final = range(1, 9)


def _first(value: object) -> object:
    if isinstance(value, tuple | list):
        return value[0] if value else None
    return value


def _as_text(value: object) -> str | None:
    value = _first(value) if isinstance(value, list) else value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    text = value.strip("\x00 ").strip()
    return text or None


def _as_float(value: object) -> float | None:
    value = _first(value)
    try:
        number = float(value)  # type: ignore[arg-type]  # IFDRational, int, float, str
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _as_datetime(value: object) -> datetime | None:
    text = _as_text(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text[:19], EXIF_DATETIME_FORMAT)  # EXIF has no time zone
    except ValueError:
        return None


def _scalar(value: object) -> Scalar:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple | list):
        return ",".join(str(_scalar(v)) for v in value)
    number = _as_float(value)
    return number if number is not None else str(value)


def _gps_record(gps: Mapping[int, object]) -> dict[str, Scalar] | None:
    """Return GPS tags by name, or ``None`` when no position is recorded (version tag only)."""
    tags = {k: v for k, v in gps.items() if k != _GPS_VERSION_TAG}
    if not tags:
        return None
    return {ExifTags.GPSTAGS.get(k, str(k)): _scalar(v) for k, v in sorted(tags.items())}


class ExifReader:
    """Reads camera metadata from a RAW (TIFF-based) or JPEG/TIFF file.

    Implements :class:`~raw_pipeline.core.protocols.MetadataReader`. Missing or
    malformed tags are ``None``; an unreadable file yields an empty
    :class:`ExifInfo` and a warning, because metadata never fails an image.
    """

    def read(self, path: Path) -> ExifInfo:
        """Return the EXIF of ``path``; missing tags are ``None``."""
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                sub = exif.get_ifd(ExifTags.IFD.Exif)
                gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
            logger.warning("%s: cannot read EXIF (%s); camera metadata left empty", path.name, exc)
            return ExifInfo()
        orientation = _as_int(exif.get(ExifTags.Base.Orientation))
        return ExifInfo(
            camera_make=_as_text(exif.get(ExifTags.Base.Make)),
            camera_model=_as_text(exif.get(ExifTags.Base.Model)),
            lens_model=_as_text(sub.get(ExifTags.Base.LensModel)),
            captured_at=_as_datetime(sub.get(ExifTags.Base.DateTimeOriginal))
            or _as_datetime(exif.get(ExifTags.Base.DateTime)),
            iso=_as_int(sub.get(ExifTags.Base.ISOSpeedRatings)),
            exposure_time_s=_as_float(sub.get(ExifTags.Base.ExposureTime)),
            f_number=_as_float(sub.get(ExifTags.Base.FNumber)),
            focal_length_mm=_as_float(sub.get(ExifTags.Base.FocalLength)),
            orientation=orientation if orientation in _ROTATION_TAGS else None,
            gps=_gps_record(gps),
        )
