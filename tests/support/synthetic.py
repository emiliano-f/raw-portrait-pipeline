"""Synthetic test data: images, faces and CFA frames with injected defects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.raw import CFAFrame
from raw_pipeline.core.values import BoundingBox, FaceRegion

RGGB = ((0, 1), (3, 2))
"""LibRaw pattern for an RGGB sensor with color_desc 'RGBG' (Canon 800D layout)."""
WHITE_LEVEL = 15000
BLACK_LEVEL = 2048


def gradient_image(height: int = 64, width: int = 96, peak: float = 1.5) -> NDArray[np.float32]:
    """Horizontal scene-linear gradient from 0 to ``peak`` (values > 1 included)."""
    ramp = np.linspace(0.0, peak, width, dtype=np.float32)
    return np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy()


def flat_image(value: float = 0.18, height: int = 64, width: int = 96) -> NDArray[np.float32]:
    """Uniform grey scene-linear image."""
    return np.full((height, width, 3), value, dtype=np.float32)


def face_image(
    face_luminance: float = 0.05, background: float = 0.4, height: int = 120, width: int = 160
) -> tuple[NDArray[np.float32], FaceRegion]:
    """Image with a flat "face" patch of known luminance and the matching region."""
    img = flat_image(background, height, width)
    box = BoundingBox(width * 0.35, height * 0.25, width * 0.65, height * 0.7)
    img[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)] = face_luminance
    return img, FaceRegion(box=box, confidence=0.99, detector="synthetic")


def decoded_context(image: NDArray[np.float32] | None = None, name: str = "IMG_0001.CR2") -> ImageContext:
    """A context as it looks right after decoding."""
    img = gradient_image() if image is None else image
    return ImageContext.initial(Path(name)).with_image(img)


def cfa_frame(
    height: int = 64,
    width: int = 64,
    *,
    level: int = 4000,
    noise_sigma: float = 20.0,
    seed: int = 0,
    edge_at_x: int | None = None,
    edge_level: int = 12000,
) -> CFAFrame:
    """Bayer frame with per-channel levels, mild noise and an optional sharp vertical edge.

    When ``edge_at_x`` is set, columns ``>= edge_at_x`` are at ``edge_level``.
    """
    rng = np.random.default_rng(seed)
    channel_gain = np.asarray([1.0, 0.6, 1.0, 0.6])[np.tile(np.asarray(RGGB), (height // 2, width // 2))]
    base = np.full((height, width), float(level))
    if edge_at_x is not None:
        base[:, edge_at_x:] = edge_level
    data = BLACK_LEVEL + base * channel_gain + rng.normal(0.0, noise_sigma, (height, width))
    return CFAFrame(
        data=np.clip(data, 0, WHITE_LEVEL).astype(np.uint16),
        pattern=RGGB,
        color_desc="RGBG",
        black_level=(BLACK_LEVEL,) * 4,
        white_level=WHITE_LEVEL,
    )


def inject_hot(frame: CFAFrame, mask: NDArray[np.bool_], value: int = WHITE_LEVEL) -> CFAFrame:
    """Return ``frame`` with photosites in ``mask`` set to ``value`` (a stuck/hot line)."""
    data = frame.data.copy()
    data[mask] = value
    return frame.with_data(data)
