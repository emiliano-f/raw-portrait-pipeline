"""Small immutable value objects exchanged between components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

Scalar = str | int | float | bool | None
"""Value type allowed in per-stage metadata records (written verbatim to the CSV)."""

ParamRecord = Mapping[str, Scalar]
"""Parameters one stage applied to one image."""


def freeze_record(params: Mapping[str, Scalar]) -> ParamRecord:
    """Return a read-only copy of ``params`` with keys in sorted order."""
    return MappingProxyType(dict(sorted(params.items())))


@dataclass(frozen=True)
class ExifInfo:
    """Camera metadata read from the RAW file.

    Attributes:
        camera_make: Manufacturer, e.g. ``"Canon"``.
        camera_model: Model, e.g. ``"Canon EOS 800D"``.
        lens_model: Lens description if the file records it.
        captured_at: Capture timestamp as recorded by the camera (no time zone).
        iso: ISO speed.
        exposure_time_s: Shutter time in seconds.
        f_number: Aperture as an f-number (e.g. ``2.8``).
        focal_length_mm: Focal length in millimetres.
        orientation: EXIF orientation tag (1-8) of the original file.
        gps: Raw GPS tags. Only written to outputs when ``export.keep_gps`` is true.
    """

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    captured_at: datetime | None = None
    iso: int | None = None
    exposure_time_s: float | None = None
    f_number: float | None = None
    focal_length_mm: float | None = None
    orientation: int | None = None
    gps: Mapping[str, Scalar] | None = None

    def as_record(self) -> ParamRecord:
        """Return the non-GPS fields as a flat record for the CSV report."""
        return freeze_record(
            {
                "camera_make": self.camera_make,
                "camera_model": self.camera_model,
                "lens_model": self.lens_model,
                "captured_at": self.captured_at.isoformat() if self.captured_at else None,
                "iso": self.iso,
                "exposure_time_s": self.exposure_time_s,
                "f_number": self.f_number,
                "focal_length_mm": self.focal_length_mm,
                "orientation": self.orientation,
                "has_gps": self.gps is not None,
            }
        )


@dataclass(frozen=True)
class ImageMetadata:
    """EXIF plus every parameter recorded by the stages, in execution order.

    Instances are immutable; use :meth:`with_record` to obtain a new instance.

    Attributes:
        exif: Camera metadata of the source file.
        records: Mapping ``stage name -> parameters applied``.
    """

    exif: ExifInfo = field(default_factory=ExifInfo)
    records: Mapping[str, ParamRecord] = field(default_factory=lambda: MappingProxyType({}))

    def with_record(self, stage: str, params: Mapping[str, Scalar]) -> ImageMetadata:
        """Return a copy with ``params`` recorded under ``stage`` (replacing any earlier record)."""
        records = dict(self.records)
        records[stage] = freeze_record(params)
        return ImageMetadata(exif=self.exif, records=MappingProxyType(records))

    def with_exif(self, exif: ExifInfo) -> ImageMetadata:
        """Return a copy with ``exif`` replaced."""
        return ImageMetadata(exif=exif, records=self.records)

    def flatten(self) -> dict[str, Scalar]:
        """Flatten records to ``{"stage.param": value}`` plus ``exif.*`` columns."""
        flat: dict[str, Scalar] = {f"exif.{k}": v for k, v in self.exif.as_record().items()}
        for stage, params in self.records.items():
            for key, value in params.items():
                flat[f"{stage}.{key}"] = value
        return flat


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates, ``x0 <= x < x1`` and ``y0 <= y < y1``."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (self.x1 >= self.x0 and self.y1 >= self.y0):
            msg = f"invalid box {self}: x1 < x0 or y1 < y0"
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def scaled(self, sx: float, sy: float) -> BoundingBox:
        """Return the box with x scaled by ``sx`` and y by ``sy``."""
        return BoundingBox(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy)

    def clipped(self, width: int, height: int) -> BoundingBox:
        """Return the box clipped to an image of ``width`` x ``height`` pixels."""
        return BoundingBox(
            min(max(self.x0, 0.0), float(width)),
            min(max(self.y0, 0.0), float(height)),
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
        )


@dataclass(frozen=True)
class FaceRegion:
    """A detected face.

    Coordinates are in the pixel grid of the image they refer to. Detectors
    return preview coordinates; the meter stage maps them to full resolution
    with :func:`raw_pipeline.core.geometry.preview_to_full` before storing them.

    Attributes:
        box: Face bounding box.
        confidence: Detector score in ``[0, 1]``.
        detector: Registered name of the detector that produced it.
        landmarks: Optional ``(N, 2)`` float32 array of ``(x, y)`` points.
        landmark_scheme: Identifier of the landmark layout, e.g.
            ``"mediapipe_face_mesh_478"``. ``None`` when ``landmarks`` is ``None``.
    """

    box: BoundingBox
    confidence: float
    detector: str
    landmarks: NDArray[np.float32] | None = None
    landmark_scheme: str | None = None


@dataclass(frozen=True)
class ExposureMeasurement:
    """Result of a metering strategy.

    Attributes:
        measured_luminance: Weighted scene-linear luminance (Rec.709/sRGB weights)
            of the metered area, before correction.
        target_luminance: Scene-linear luminance the strategy aims for.
        correction_ev: Exposure correction in EV stops (``image * 2**correction_ev``),
            already clamped by the strategy's limits.
        method: Registered name of the strategy that produced the value
            (after any fallback, e.g. ``"center_weighted"``).
        faces_used: Number of faces that contributed to the measurement.
    """

    measured_luminance: float
    target_luminance: float
    correction_ev: float
    method: str
    faces_used: int = 0


@dataclass(frozen=True)
class QAResult:
    """Flags returned by a QA provider. QA never edits pixels.

    Attributes:
        provider: Registered provider name (``"none"`` for the null provider).
        flags: Issue identifiers, e.g. ``{"closed_eyes", "blur", "skin_tone"}``.
        notes: Short free-text explanation from the provider.
        skipped: ``True`` when no review happened (disabled, no API key, API error).
        model: Provider model identifier, if any.
    """

    provider: str
    flags: frozenset[str] = frozenset()
    notes: str = ""
    skipped: bool = False
    model: str | None = None


@dataclass(frozen=True)
class RetouchResult:
    """Output of a :class:`~raw_pipeline.core.protocols.Retoucher`.

    Attributes:
        image: Retouched scene-linear float32 ``(H, W, 3)`` image.
        params: Parameters applied, recorded in the report (e.g. strength, mask coverage).
    """

    image: NDArray[np.float32]
    params: ParamRecord
