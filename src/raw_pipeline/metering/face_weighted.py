"""Face-weighted metering: expose so the faces land on the target luminance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import MeteringError
from raw_pipeline.core.registries import metering_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, ExposureMeasurement, FaceRegion
from raw_pipeline.metering.stats import (
    MIN_MEASURABLE_LUMINANCE,
    correction_ev,
    subsampled_luminance,
    trimmed_mean,
)

SKIN_REGION_X: Final = (0.25, 0.75)
"""Horizontal span of the metered skin area, as fractions of the face box width."""
SKIN_REGION_Y: Final = (0.2, 0.8)
"""Vertical span of the metered skin area, as fractions of the face box height
(below the hairline, above the chin and neck)."""
TRIM_FRACTION: Final = 0.2
"""Fraction of the darkest and of the brightest skin samples ignored (eyes, brows,
nostrils, specular highlights)."""
FACE_HIGHLIGHT_PERCENTILE: Final = 90.0
"""Percentile of the skin luminance kept below ``highlight_ceiling`` (the sunlit side
of a split-lit face, ignoring small specular highlights)."""
MIN_RELATIVE_FACE_AREA: Final = 0.25
"""Faces smaller than this fraction of the largest face are not metered (background people)."""


@metering_registry.register("face_weighted")
class FaceWeightedMetering:
    """Trimmed-mean luminance of the central skin area of the detected faces.

    Faces come from the ``FACES`` artifact (full-resolution coordinates, set by
    the meter stage). Several faces are combined with weights proportional to
    their box area. The correction is lowered when it would push the
    ``FACE_HIGHLIGHT_PERCENTILE`` skin luminance of any face above
    ``highlight_ceiling``, so the lit side of a split-lit face keeps its detail.

    Args:
        target_luminance: Scene-linear face luminance to reach (Rec.709 weights).
        max_correction_ev: Absolute correction limit in EV stops.
        highlight_ceiling: Scene-linear luminance the bright skin may reach after
            the correction; ``math.inf`` disables the limit.
    """

    name: ClassVar[str] = "face_weighted"

    def __init__(self, target_luminance: float, max_correction_ev: float, highlight_ceiling: float = math.inf) -> None:
        self.target_luminance = target_luminance
        self.max_correction_ev = max_correction_ev
        self.highlight_ceiling = highlight_ceiling

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``meter`` section (target, correction limit, highlight ceiling)."""
        cfg = app_config(env).meter
        return cls(cfg.target_face_luminance, cfg.max_correction_ev, cfg.face_highlight_ceiling)

    def measure(self, ctx: ImageContext) -> ExposureMeasurement:
        """Meter the faces of ``ctx`` (scene-linear image).

        Raises:
            MeteringError: No face in the context, or no measurable skin pixel.
        """
        found = ctx.artifact(FACES)
        faces = metered_faces(found if found is not None else ())
        height, width = ctx.image.shape[:2]
        values: list[float] = []
        weights: list[float] = []
        highlights: list[float] = []
        for face in faces:
            x0, y0, x1, y1 = _pixel_bounds(skin_region(face.box), width, height)
            if x1 <= x0 or y1 <= y0:
                continue
            lum = subsampled_luminance(ctx.image[y0:y1, x0:x1])
            values.append(trimmed_mean(lum, TRIM_FRACTION))
            weights.append(face.box.area)
            highlights.append(_highlight(lum))
        if not values:
            msg = "face-weighted metering needs at least one face with a non-empty skin area"
            raise MeteringError(msg)
        measured = float(np.average(np.asarray(values), weights=np.asarray(weights)))
        ev = min(
            correction_ev(measured, self.target_luminance, self.max_correction_ev),
            self._highlight_limit_ev(max(highlights)),
        )
        return ExposureMeasurement(
            measured_luminance=measured,
            target_luminance=self.target_luminance,
            correction_ev=max(ev, -self.max_correction_ev),
            method=self.name,
            faces_used=len(values),
        )

    def _highlight_limit_ev(self, highlight: float) -> float:
        """Largest correction (EV) keeping ``highlight`` at or below the ceiling."""
        if math.isinf(self.highlight_ceiling):
            return math.inf
        return math.log2(self.highlight_ceiling / max(highlight, MIN_MEASURABLE_LUMINANCE))


def _highlight(lum: NDArray[np.float32]) -> float:
    finite = lum[np.isfinite(lum)]
    return float(np.percentile(finite, FACE_HIGHLIGHT_PERCENTILE)) if finite.size else 0.0


def metered_faces(faces: Sequence[FaceRegion]) -> list[FaceRegion]:
    """Faces at least ``MIN_RELATIVE_FACE_AREA`` of the largest one."""
    if not faces:
        return []
    largest = max(f.box.area for f in faces)
    return [f for f in faces if f.box.area > 0 and f.box.area >= MIN_RELATIVE_FACE_AREA * largest]


def skin_region(box: BoundingBox) -> BoundingBox:
    """Central skin part of a face box (excludes hair, ears and background)."""
    return BoundingBox(
        box.x0 + SKIN_REGION_X[0] * box.width,
        box.y0 + SKIN_REGION_Y[0] * box.height,
        box.x0 + SKIN_REGION_X[1] * box.width,
        box.y0 + SKIN_REGION_Y[1] * box.height,
    )


def _pixel_bounds(box: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    clipped = box.clipped(width, height)
    return round(clipped.x0), round(clipped.y0), round(clipped.x1), round(clipped.y1)
