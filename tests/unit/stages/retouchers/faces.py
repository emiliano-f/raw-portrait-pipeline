"""Synthetic MediaPipe-style 478-point faces with known feature geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.values import BoundingBox, FaceRegion
from raw_pipeline.stages.retouchers import skin_mask as sm


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float

    def points(self, n: int) -> NDArray[np.float32]:
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([self.cx + self.rx * np.cos(t), self.cy + self.ry * np.sin(t)], axis=1).astype(np.float32)

    def mask(self, height: int, width: int) -> NDArray[np.bool_]:
        yy, xx = np.mgrid[0:height, 0:width]
        return ((xx - self.cx) / self.rx) ** 2 + ((yy - self.cy) / self.ry) ** 2 <= 1.0


@dataclass(frozen=True)
class SyntheticFace:
    face: FaceRegion
    oval: Ellipse
    features: tuple[Ellipse, ...]

    def feature_mask(self, height: int, width: int) -> NDArray[np.bool_]:
        out = np.zeros((height, width), dtype=np.bool_)
        for e in self.features:
            out |= e.mask(height, width)
        return out


def mesh_face(cx: float = 100.0, cy: float = 90.0, w: float = 120.0, h: float = 150.0) -> SyntheticFace:
    oval = Ellipse(cx, cy, w / 2, h / 2)
    left_eye = Ellipse(cx + 0.2 * w, cy - 0.12 * h, 0.1 * w, 0.04 * h)
    right_eye = Ellipse(cx - 0.2 * w, cy - 0.12 * h, 0.1 * w, 0.04 * h)
    left_brow = Ellipse(cx + 0.2 * w, cy - 0.24 * h, 0.12 * w, 0.025 * h)
    right_brow = Ellipse(cx - 0.2 * w, cy - 0.24 * h, 0.12 * w, 0.025 * h)
    lips = Ellipse(cx, cy + 0.28 * h, 0.18 * w, 0.06 * h)
    nose = Ellipse(cx, cy + 0.1 * h, 0.09 * w, 0.05 * h)
    lm = np.tile(np.float32([cx, cy]), (sm.MEDIAPIPE_LANDMARK_COUNT, 1))
    for idx, ellipse in (
        (sm.FACE_OVAL, oval),
        (sm.LEFT_EYE, left_eye),
        (sm.RIGHT_EYE, right_eye),
        (sm.LEFT_EYEBROW, left_brow),
        (sm.RIGHT_EYEBROW, right_brow),
        (sm.LIPS, lips),
        (sm.NOSTRILS, nose),
    ):
        lm[list(idx)] = ellipse.points(len(idx))
    face = FaceRegion(
        box=BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=0.99,
        detector="synthetic",
        landmarks=lm,
        landmark_scheme=sm.MEDIAPIPE_FACE_MESH_478,
    )
    return SyntheticFace(face, oval, (left_eye, right_eye, left_brow, right_brow, lips, nose))


def skin_image(height: int = 200, width: int = 200, seed: int = 0) -> NDArray[np.float32]:
    """Skin-coloured image with blotches (mid frequency) and fine texture (seeded noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    blotch = 0.04 * np.sin(xx / 6.0) * np.cos(yy / 7.0)
    texture = rng.normal(0.0, 0.01, (height, width)).astype(np.float32)
    base = np.float32([0.35, 0.25, 0.2])
    return (base[None, None, :] * (1.0 + blotch + texture)[..., None]).astype(np.float32)
