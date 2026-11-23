"""Colour math shared by every package.

The working space is scene-linear RGB with sRGB / Rec.709 primaries and a D65
white point. These helpers are the single definition of luminance and of the
sRGB transfer function, so metering, tone, exporters, reports and QA previews
agree. Only encoders (exporters, preview builders) may call `srgb_encode`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

REC709_LUMA_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)
"""Relative luminance weights for linear Rec.709 / sRGB primaries."""

_SRGB_LINEAR_CUTOFF: Final = 0.0031308
_SRGB_ENCODED_CUTOFF: Final = 0.04045
_SRGB_LINEAR_SLOPE: Final = 12.92
_SRGB_GAMMA: Final = 2.4
_SRGB_OFFSET: Final = 0.055


def luminance(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return per-pixel relative luminance of a scene-linear `(H, W, 3)` image.

    Args:
        image: Scene-linear RGB in the working space. Values above 1.0 allowed.

    Returns:
        `(H, W)` float32 luminance (same scale as the input).
    """
    weights = np.asarray(REC709_LUMA_WEIGHTS, dtype=np.float32)
    return np.asarray(image @ weights, dtype=np.float32)


def srgb_encode(linear: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the sRGB transfer function (IEC 61966-2-1) after clipping to [0, 1].

    Args:
        linear: Scene-linear values in the working space.

    Returns:
        Encoded values in [0, 1], float32, same shape.
    """
    x = np.clip(linear, 0.0, 1.0).astype(np.float32)
    low = x * _SRGB_LINEAR_SLOPE
    high = (1.0 + _SRGB_OFFSET) * np.power(x, 1.0 / _SRGB_GAMMA) - _SRGB_OFFSET
    encoded = np.where(x <= _SRGB_LINEAR_CUTOFF, low, high)
    return np.clip(encoded, 0.0, 1.0).astype(np.float32)


def srgb_decode(encoded: NDArray[np.float32]) -> NDArray[np.float32]:
    """Invert `srgb_encode` for values in [0, 1].

    Args:
        encoded: sRGB-encoded values in [0, 1].

    Returns:
        Scene-linear values, float32, same shape.
    """
    x = np.clip(encoded, 0.0, 1.0).astype(np.float32)
    low = x / _SRGB_LINEAR_SLOPE
    high = np.power((x + _SRGB_OFFSET) / (1.0 + _SRGB_OFFSET), _SRGB_GAMMA)
    return np.where(x <= _SRGB_ENCODED_CUTOFF, low, high).astype(np.float32)


def to_uint8(encoded: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Quantize encoded [0, 1] values to uint8 with round-half-to-even."""
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_uint16(encoded: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Quantize encoded [0, 1] values to uint16 with round-half-to-even."""
    return np.rint(np.clip(encoded, 0.0, 1.0) * 65535.0).astype(np.uint16)
