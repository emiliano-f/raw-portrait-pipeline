"""Coordinate mapping between the detection preview and the full-resolution image."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from raw_pipeline.core.values import FaceRegion


@dataclass(frozen=True)
class PreviewGeometry:
    """Size of a downscaled preview relative to the full-resolution image.

    Attributes:
        full_width: Full-resolution width in pixels.
        full_height: Full-resolution height in pixels.
        preview_width: Preview width in pixels.
        preview_height: Preview height in pixels.
    """

    full_width: int
    full_height: int
    preview_width: int
    preview_height: int

    @classmethod
    def for_long_edge(cls, full_width: int, full_height: int, long_edge_px: int) -> PreviewGeometry:
        """Geometry of a preview whose long edge is at most ``long_edge_px`` (never upscaled)."""
        scale = min(1.0, long_edge_px / max(full_width, full_height))
        return cls(
            full_width=full_width,
            full_height=full_height,
            preview_width=max(1, round(full_width * scale)),
            preview_height=max(1, round(full_height * scale)),
        )

    @property
    def scale_x(self) -> float:
        """Full-resolution pixels per preview pixel along x."""
        return self.full_width / self.preview_width

    @property
    def scale_y(self) -> float:
        """Full-resolution pixels per preview pixel along y."""
        return self.full_height / self.preview_height


def preview_to_full(face: FaceRegion, geometry: PreviewGeometry) -> FaceRegion:
    """Map a face detected on the preview to full-resolution coordinates.

    This is the single place where preview coordinates are converted; boxes
    and landmarks are scaled independently on each axis and clipped to the
    full-resolution frame.
    """
    box = face.box.scaled(geometry.scale_x, geometry.scale_y).clipped(geometry.full_width, geometry.full_height)
    landmarks = None
    if face.landmarks is not None:
        scale = np.asarray([geometry.scale_x, geometry.scale_y], dtype=np.float32)
        landmarks = (face.landmarks.astype(np.float32) * scale).astype(np.float32)
    return replace(face, box=box, landmarks=landmarks)
