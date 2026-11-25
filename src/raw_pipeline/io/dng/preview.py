"""Embedded preview images of a DNG: render, encode and replace (ADR 0003).

DNG previews are stored in sensor orientation (IFD0 Orientation applies to
them), so they are rendered without LibRaw's flip.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final

import numpy as np
import rawpy
from numpy.typing import NDArray
from PIL import Image

PREVIEW_JPEG_QUALITY: Final = 90
NEUTRAL_GREY: Final = 128
"""8-bit value of a removed (blanked) thumbnail."""
NO_FLIP: Final = 0


def render_preview(path: Path) -> NDArray[np.uint8]:
    """Quick sRGB 8-bit development of the DNG at ``path`` (camera WB, half size, unrotated)."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True, user_flip=NO_FLIP, output_bps=8)
    return np.asarray(rgb, dtype=np.uint8)


def resized(rgb: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    """``rgb`` resampled to ``width`` x ``height`` (Lanczos)."""
    image = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def grey(width: int, height: int) -> NDArray[np.uint8]:
    """A neutral grey ``width`` x ``height`` RGB image."""
    return np.full((height, width, 3), NEUTRAL_GREY, dtype=np.uint8)


def encode_jpeg(rgb: NDArray[np.uint8]) -> bytes:
    """Baseline JPEG of ``rgb`` without metadata (deterministic)."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY, optimize=False)
    return buffer.getvalue()
