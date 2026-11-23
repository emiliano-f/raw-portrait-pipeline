"""Meter stage: find faces on a preview and measure the exposure correction.

The stage never changes pixels. It writes the ``FACES`` artifact (full-resolution
coordinates) and the ``EXPOSURE`` measurement that the tone stage applies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import ClassVar, Final, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import EXPOSURE, FACES
from raw_pipeline.core.color import luminance, srgb_encode, to_uint8
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import DetectionError, MeteringError, StageError
from raw_pipeline.core.geometry import PreviewGeometry, preview_to_full
from raw_pipeline.core.protocols import FaceDetector, MeteringStrategy
from raw_pipeline.core.registries import face_detector_registry, metering_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import ExposureMeasurement, FaceRegion

logger = logging.getLogger(__name__)

PREVIEW_TARGET_MEDIAN_LUMINANCE: Final = 0.18
"""The detection preview is brightened/darkened so its median luminance lands here."""
PREVIEW_MAX_GAIN_EV: Final = 4.0
"""Limit of that preview-only gain, in EV stops."""
_MIN_LUMINANCE: Final = 1e-6


@dataclass(frozen=True)
class _Detection:
    faces: list[FaceRegion]
    detector_used: str | None
    errors: tuple[str, ...]


@stage_registry.register("meter")
class MeterStage:
    """Detect faces and meter exposure on them, falling back to center-weighted metering.

    The primary detector runs first; the fallback detector runs when the
    primary one raises :class:`DetectionError` (e.g. MediaPipe not installed)
    or finds no face. When no face is found at all, or the primary strategy
    cannot measure, ``fallback_strategy`` is used.

    Args:
        detector: Primary face detector.
        fallback_detector: Second detector, or ``None``.
        strategy: Metering strategy used when faces were found.
        fallback_strategy: Metering strategy used without faces.
        detection_long_edge_px: Long edge of the detection preview in pixels.
        min_face_confidence: Faces below this detector score are discarded.
    """

    name: ClassVar[str] = "meter"

    def __init__(
        self,
        detector: FaceDetector,
        fallback_detector: FaceDetector | None,
        strategy: MeteringStrategy,
        fallback_strategy: MeteringStrategy,
        *,
        detection_long_edge_px: int,
        min_face_confidence: float,
    ) -> None:
        self.detector = detector
        self.fallback_detector = fallback_detector
        self.strategy = strategy
        self.fallback_strategy = fallback_strategy
        self.detection_long_edge_px = detection_long_edge_px
        self.min_face_confidence = min_face_confidence

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``meter`` section, resolving components through the registries."""
        cfg = app_config(env).meter
        fallback = face_detector_registry.create(cfg.fallback_detector, env) if cfg.fallback_detector else None
        return cls(
            detector=face_detector_registry.create(cfg.detector, env),
            fallback_detector=fallback,
            strategy=metering_registry.create(cfg.strategy, env),
            fallback_strategy=metering_registry.create(cfg.fallback_strategy, env),
            detection_long_edge_px=cfg.detection_long_edge_px,
            min_face_confidence=cfg.min_face_confidence,
        )

    def process(self, ctx: ImageContext) -> ImageContext:
        """Set ``FACES`` and ``EXPOSURE`` and record ``meter.*``; pixels are untouched.

        Raises:
            StageError: The context has not been decoded.
            MeteringError: Neither strategy could measure the image.
        """
        if not ctx.is_decoded:
            msg = f"{ctx.source_path.name}: meter needs a decoded image"
            raise StageError(msg)
        height, width = ctx.image.shape[:2]
        geometry = PreviewGeometry.for_long_edge(width, height, self.detection_long_edge_px)
        gain_ev = preview_gain_ev(ctx.image)
        detection = self._detect(build_preview(ctx.image, geometry, gain_ev), ctx)
        faces = tuple(preview_to_full(f, geometry) for f in detection.faces)
        with_faces = ctx.with_artifact(FACES, faces)
        measurement = self._measure(with_faces, ctx)
        return with_faces.with_artifact(EXPOSURE, measurement).with_record(
            self.name,
            {
                "detector": self.detector.name,
                "detector_used": detection.detector_used,
                "detection_errors": "; ".join(detection.errors) or None,
                "faces_found": len(faces),
                "faces_used": measurement.faces_used,
                "strategy": self.strategy.name,
                "method": measurement.method,
                "fallback_used": measurement.method != self.strategy.name,
                "measured_luminance": round(measurement.measured_luminance, 6),
                "target_luminance": measurement.target_luminance,
                "correction_ev": round(measurement.correction_ev, 4),
                "preview_width": geometry.preview_width,
                "preview_height": geometry.preview_height,
                "preview_gain_ev": round(gain_ev, 4),
                "min_face_confidence": self.min_face_confidence,
            },
        )

    def _detect(self, preview: NDArray[np.uint8], ctx: ImageContext) -> _Detection:
        errors: list[str] = []
        detectors = [self.detector] + ([self.fallback_detector] if self.fallback_detector is not None else [])
        for detector in detectors:
            try:
                found = detector.detect(preview)
            except DetectionError as exc:
                logger.warning("%s: detector '%s' failed: %s", ctx.source_path.name, detector.name, exc)
                errors.append(f"{detector.name}: {exc}")
                continue
            faces = sorted(
                (f for f in found if f.confidence >= self.min_face_confidence),
                key=lambda f: (-f.box.area, f.box.x0, f.box.y0),
            )
            if faces:
                return _Detection(faces, detector.name, tuple(errors))
        return _Detection([], None, tuple(errors))

    def _measure(self, ctx: ImageContext, original: ImageContext) -> ExposureMeasurement:
        if ctx.artifact(FACES):
            try:
                return self.strategy.measure(ctx)
            except MeteringError as exc:
                logger.warning(
                    "%s: '%s' metering failed (%s); using '%s'",
                    original.source_path.name,
                    self.strategy.name,
                    exc,
                    self.fallback_strategy.name,
                )
        else:
            logger.info("%s: no face found; using '%s'", original.source_path.name, self.fallback_strategy.name)
        return self.fallback_strategy.measure(ctx)


def preview_gain_ev(image: NDArray[np.float32]) -> float:
    """EV gain that brings the median luminance of ``image`` to ``PREVIEW_TARGET_MEDIAN_LUMINANCE``.

    Only used to make the detection preview legible; clamped to ``±PREVIEW_MAX_GAIN_EV``.
    """
    lum = luminance(image[:: max(1, image.shape[0] // 256), :: max(1, image.shape[1] // 256)])
    finite = lum[np.isfinite(lum)]
    median = float(np.median(finite)) if finite.size else 0.0
    ev = math.log2(PREVIEW_TARGET_MEDIAN_LUMINANCE / max(median, _MIN_LUMINANCE))
    return float(min(max(ev, -PREVIEW_MAX_GAIN_EV), PREVIEW_MAX_GAIN_EV))


def build_preview(image: NDArray[np.float32], geometry: PreviewGeometry, gain_ev: float = 0.0) -> NDArray[np.uint8]:
    """Downscale (area filter), apply ``gain_ev`` and sRGB-encode a scene-linear image to uint8.

    Args:
        image: Scene-linear float32 ``(H, W, 3)``.
        geometry: Target preview size.
        gain_ev: Exposure gain in EV applied to the preview only.

    Returns:
        ``(h, w, 3)`` uint8 sRGB-encoded preview.
    """
    size = (geometry.preview_width, geometry.preview_height)
    small = image if size == (image.shape[1], image.shape[0]) else cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    linear = np.nan_to_num(np.asarray(small, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return to_uint8(srgb_encode((linear * np.float32(2.0**gain_ev)).astype(np.float32)))
