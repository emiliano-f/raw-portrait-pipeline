"""OpenCV Haar-cascade face detector, the offline fallback when MediaPipe is unavailable.

The cascade ships with ``opencv-python-headless`` (``cv2.data.haarcascades``),
so no model is downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.errors import DetectionError
from raw_pipeline.core.registries import face_detector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, FaceRegion

HAAR_CASCADE_FILE: Final = "haarcascade_frontalface_default.xml"
HAAR_SCALE_FACTOR: Final = 1.1
"""Image pyramid step between scales."""
HAAR_MIN_NEIGHBORS: Final = 6
"""Overlapping raw hits required to keep a detection (higher = fewer false positives)."""
MIN_FACE_FRACTION: Final = 0.06
"""Smallest face edge accepted, as a fraction of the preview's short edge."""
HAAR_CONFIDENCE: Final = 1.0
"""Haar cascades have no calibrated score; accepted detections report this value."""


def default_cascade_path() -> Path:
    """Frontal-face cascade shipped with OpenCV."""
    return Path(str(cv2.data.haarcascades)) / HAAR_CASCADE_FILE  # type: ignore[attr-defined]  # missing from cv2 stubs


@face_detector_registry.register("opencv")
class OpenCVDetector:
    """Frontal-face Haar cascade on the histogram-equalised grey preview.

    The classifier is loaded on each call, so instances stay picklable.

    Args:
        cascade_path: Haar cascade XML file.
        min_face_fraction: Smallest accepted face edge relative to the preview short edge.
    """

    name: ClassVar[str] = "opencv"

    def __init__(self, cascade_path: Path | None = None, min_face_fraction: float = MIN_FACE_FRACTION) -> None:
        self.cascade_path = cascade_path or default_cascade_path()
        self.min_face_fraction = min_face_fraction

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:  # noqa: ARG003 - no configurable settings
        """Build with the default cascade."""
        return cls()

    def detect(self, preview: NDArray[np.uint8]) -> list[FaceRegion]:
        """Detect frontal faces on an sRGB uint8 ``(h, w, 3)`` preview, in preview pixels.

        Raises:
            DetectionError: The cascade file cannot be loaded.
        """
        classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if classifier.empty():
            msg = f"cannot load Haar cascade {self.cascade_path}"
            raise DetectionError(msg)
        height, width = preview.shape[:2]
        grey = cv2.equalizeHist(cv2.cvtColor(np.ascontiguousarray(preview), cv2.COLOR_RGB2GRAY))
        min_edge = max(1, round(min(height, width) * self.min_face_fraction))
        hits = classifier.detectMultiScale(
            grey, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS, minSize=(min_edge, min_edge)
        )
        boxes = [
            BoundingBox(float(x), float(y), float(x + w), float(y + h)).clipped(width, height)
            for x, y, w, h in np.asarray(hits).reshape(-1, 4)
        ]
        boxes.sort(key=lambda b: (-b.area, b.x0, b.y0))
        return [FaceRegion(box=b, confidence=HAAR_CONFIDENCE, detector=self.name) for b in boxes if b.area > 0]
