"""The JPEG preview that is the only image data the QA step may send externally."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from raw_pipeline.core.color import srgb_encode, to_uint8
from raw_pipeline.core.geometry import PreviewGeometry

MAX_PREVIEW_LONG_EDGE_PX: Final = 1024
"""Hard upper bound on the long edge of anything sent to a QA provider (AGENTS.md invariant)."""

_JPEG_SUBSAMPLING_420: Final = 2
_JPEG_MAGIC: Final = b"\xff\xd8\xff"


@dataclass(frozen=True)
class PreviewInfo:
    """What a provider may check about a preview before sending it.

    Attributes:
        width: Width in pixels.
        height: Height in pixels.
        has_exif: Whether the JPEG carries an EXIF block.
    """

    width: int
    height: int
    has_exif: bool

    @property
    def long_edge_px(self) -> int:
        """Longest side in pixels."""
        return max(self.width, self.height)


def encode_preview(image: NDArray[np.float32], long_edge_px: int, jpeg_quality: int) -> bytes:
    """Downscale a scene-linear image and encode it as an EXIF-free sRGB JPEG.

    Downscaling happens in linear light (box filter) before the sRGB transfer
    function is applied. The long edge never exceeds
    :data:`MAX_PREVIEW_LONG_EDGE_PX`, whatever ``long_edge_px`` says, and the
    image is never upscaled.

    Args:
        image: ``(H, W, 3)`` float32 scene-linear working-space image.
        long_edge_px: Requested long edge in pixels.
        jpeg_quality: JPEG quality (1-100).

    Returns:
        Deterministic baseline JPEG bytes with no EXIF, ICC or comment blocks.

    Raises:
        ValueError: If ``image`` is empty or not ``(H, W, 3)``.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:  # noqa: PLR2004
        msg = f"preview needs a non-empty (H, W, 3) image, got {image.shape}"
        raise ValueError(msg)
    height, width = image.shape[:2]
    geometry = PreviewGeometry.for_long_edge(width, height, min(long_edge_px, MAX_PREVIEW_LONG_EDGE_PX))
    size = (geometry.preview_width, geometry.preview_height)
    channels = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(image[:, :, c], dtype=np.float32), mode="F").resize(
                size, Image.Resampling.BOX
            ),
            dtype=np.float32,
        )
        for c in range(3)
    ]
    rgb = to_uint8(srgb_encode(np.stack(channels, axis=-1)))
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buf, format="JPEG", quality=jpeg_quality, optimize=False, progressive=False, subsampling=_JPEG_SUBSAMPLING_420
    )
    return buf.getvalue()


def inspect_preview(preview_jpeg: bytes) -> PreviewInfo:
    """Return the size and EXIF presence of a JPEG preview.

    Raises:
        ValueError: If the bytes are not a readable JPEG.
    """
    if not preview_jpeg.startswith(_JPEG_MAGIC):
        msg = "preview must be a JPEG (missing JPEG start-of-image marker)"
        raise ValueError(msg)
    try:
        with Image.open(io.BytesIO(preview_jpeg), formats=["JPEG"]) as img:
            return PreviewInfo(width=img.width, height=img.height, has_exif="exif" in img.info)
    except (OSError, Image.DecompressionBombError) as exc:
        msg = f"preview is not a readable JPEG: {exc}"
        raise ValueError(msg) from exc
