"""Skin masks for retouching, from face-mesh landmarks or a box-ellipse fallback.

Masks are built for one face inside a rectangular region of the full image.
Protected features (eyes, irises, eyebrows, lips and mouth, nostrils) are
removed from the hard mask *before* feathering, so the soft mask is exactly 0
on them and outside the face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.values import FaceRegion

MEDIAPIPE_FACE_MESH_478: Final = "mediapipe_face_mesh_478"
"""Landmark scheme produced by the MediaPipe FaceLandmarker (478 points incl. irises)."""

MEDIAPIPE_LANDMARK_COUNT: Final = 478

# Landmark indices of the MediaPipe face mesh (from mediapipe's
# FaceLandmarksConnections, Apache-2.0). Regions are filled as convex hulls.
FACE_OVAL: Final = (
    10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176,
    234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454,
)  # fmt: skip
LEFT_EYE: Final = (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 474, 475, 476, 477)
RIGHT_EYE: Final = (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 469, 470, 471, 472)
LEFT_EYEBROW: Final = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
RIGHT_EYEBROW: Final = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
LIPS: Final = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)  # fmt: skip
NOSTRILS: Final = (1, 2, 4, 48, 64, 94, 97, 98, 278, 294, 326, 327)
"""Nose tip, columella and alae: the hull covers both nostrils."""

PROTECTED_REGIONS: Final = (LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS, NOSTRILS)

PROTECT_MARGIN_FACE_FRACTION: Final = 0.03
"""Protected regions are dilated by this fraction of the face box width (safety margin)."""

# Box-ellipse fallback (no landmarks): fractions of the face box. Detector boxes
# vary, so the protected bands are deliberately generous.
ELLIPSE_AXIS_FRACTION: Final = 0.45
"""Ellipse semi-axes as a fraction of box width / height."""
EYE_BAND_Y_FRACTION: Final = (0.15, 0.55)
"""Full-width band (top, bottom) protecting eyes and eyebrows."""
NOSE_BOX_FRACTION: Final = (0.33, 0.45, 0.67, 0.72)
"""(x0, y0, x1, y1) protecting the nose tip and nostrils."""
MOUTH_BOX_FRACTION: Final = (0.22, 0.62, 0.78, 0.92)
"""(x0, y0, x1, y1) protecting lips, mouth and teeth."""

_MIN_LANDMARKS_PER_REGION: Final = 3
_SUBPIXEL_BITS: Final = 4
_SUBPIXEL_SCALE: Final = 1 << _SUBPIXEL_BITS


@dataclass(frozen=True)
class SkinMask:
    """Skin mask of one face inside a region of the image.

    Attributes:
        hard: ``(h, w)`` bool, pixels that belong to skin (protected features excluded).
        soft: ``(h, w)`` float32 in ``[0, 1]``, ``hard`` feathered inwards; 0 wherever ``hard`` is false.
        source: ``"landmarks"`` or ``"ellipse"``.
    """

    hard: NDArray[np.bool_]
    soft: NDArray[np.float32]
    source: str


@dataclass(frozen=True)
class SkinMaskBuilder:
    """Builds :class:`SkinMask` objects.

    Attributes:
        feather_px: Width of the inward feather at the mask edge, in pixels (0 = hard edge).
    """

    feather_px: int

    def build(self, face: FaceRegion, origin: tuple[int, int], shape: tuple[int, int]) -> SkinMask:
        """Return the mask of ``face`` (full-resolution coordinates) in a region.

        Args:
            face: Face in full-resolution image coordinates.
            origin: ``(x, y)`` of the region's top-left corner in the image.
            shape: ``(height, width)`` of the region in pixels.
        """
        if has_face_mesh(face):
            hard, source = self._landmark_mask(face, origin, shape), "landmarks"
        else:
            hard, source = self._ellipse_mask(face, origin, shape), "ellipse"
        return SkinMask(hard=hard, soft=self._feather(hard), source=source)

    def _landmark_mask(self, face: FaceRegion, origin: tuple[int, int], shape: tuple[int, int]) -> NDArray[np.bool_]:
        assert face.landmarks is not None
        points = face.landmarks - np.asarray(origin, dtype=np.float32)
        skin = _hull_mask(points[list(FACE_OVAL)], shape)
        protected = np.zeros(shape, dtype=np.uint8)
        for region in PROTECTED_REGIONS:
            protected |= _hull_mask(points[list(region)], shape)
        margin = max(1, round(PROTECT_MARGIN_FACE_FRACTION * face.box.width))
        protected = _dilate(protected, margin)
        return np.asarray((skin > 0) & (protected == 0), dtype=np.bool_)

    def _ellipse_mask(self, face: FaceRegion, origin: tuple[int, int], shape: tuple[int, int]) -> NDArray[np.bool_]:
        box = face.box
        ox, oy = origin
        skin = np.zeros(shape, dtype=np.uint8)
        center = (round((box.x0 + box.x1) / 2 - ox), round((box.y0 + box.y1) / 2 - oy))
        axes = (max(1, round(box.width * ELLIPSE_AXIS_FRACTION)), max(1, round(box.height * ELLIPSE_AXIS_FRACTION)))
        cv2.ellipse(skin, center, axes, 0.0, 0.0, 360.0, 1, thickness=-1)

        def rows(f0: float, f1: float) -> slice:
            return _span(box.y0 + f0 * box.height, box.y0 + f1 * box.height, oy)

        def cols(f0: float, f1: float) -> slice:
            return _span(box.x0 + f0 * box.width, box.x0 + f1 * box.width, ox)

        skin[rows(*EYE_BAND_Y_FRACTION), :] = 0
        for x0, y0, x1, y1 in (NOSE_BOX_FRACTION, MOUTH_BOX_FRACTION):
            skin[rows(y0, y1), cols(x0, x1)] = 0
        return np.asarray(skin > 0, dtype=np.bool_)

    def _feather(self, hard: NDArray[np.bool_]) -> NDArray[np.float32]:
        if self.feather_px == 0 or not hard.any():
            return hard.astype(np.float32)
        padded = np.pad(hard, 1).astype(np.uint8)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        t = np.clip(dist / np.float32(self.feather_px), 0.0, 1.0)
        soft = t * t * (3.0 - 2.0 * t)
        return np.where(hard, soft, 0.0).astype(np.float32)


def has_face_mesh(face: FaceRegion) -> bool:
    """``True`` when ``face`` carries a full MediaPipe 478-point face mesh."""
    return (
        face.landmarks is not None
        and face.landmark_scheme == MEDIAPIPE_FACE_MESH_478
        and face.landmarks.shape == (MEDIAPIPE_LANDMARK_COUNT, 2)
        and bool(np.isfinite(face.landmarks).all())
    )


def _span(start: float, stop: float, offset: int) -> slice:
    """Pixel slice covering ``[start, stop)`` (image coordinates) in a region starting at ``offset``."""
    return slice(max(0, math.floor(start) - offset), max(0, math.ceil(stop) - offset))


def _hull_mask(points: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.uint8]:
    """Fill the convex hull of ``points`` (region coordinates) with sub-pixel precision.

    The hull is rasterised on its own canvas and then pasted, so the result
    does not depend on where the region boundary cuts the polygon.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < _MIN_LANDMARKS_PER_REGION:
        return mask
    x0, y0 = math.floor(float(points[:, 0].min())), math.floor(float(points[:, 1].min()))
    x1, y1 = math.ceil(float(points[:, 0].max())) + 1, math.ceil(float(points[:, 1].max())) + 1
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local = np.rint((points - np.asarray([x0, y0], dtype=np.float32)) * _SUBPIXEL_SCALE).astype(np.int32)
    cv2.fillConvexPoly(canvas, cv2.convexHull(local), 1, lineType=cv2.LINE_8, shift=_SUBPIXEL_BITS)
    height, width = shape
    cy0, cx0 = max(0, -y0), max(0, -x0)
    ry0, rx0 = max(0, y0), max(0, x0)
    ry1, rx1 = min(height, y1), min(width, x1)
    if ry1 > ry0 and rx1 > rx0:
        mask[ry0:ry1, rx0:rx1] = canvas[cy0 : cy0 + ry1 - ry0, cx0 : cx0 + rx1 - rx0]
    return mask


def _dilate(mask: NDArray[np.uint8], radius_px: int) -> NDArray[np.uint8]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return np.asarray(cv2.dilate(mask, kernel), dtype=np.uint8)
