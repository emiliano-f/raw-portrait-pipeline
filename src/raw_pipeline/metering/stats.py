"""Robust luminance statistics and EV maths shared by the metering strategies."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.color import luminance
from raw_pipeline.core.errors import MeteringError

METERING_MAX_LONG_EDGE_PX: Final = 1024
"""Images are subsampled (strided, deterministic) to about this long edge before metering."""
MIN_MEASURABLE_LUMINANCE: Final = 1e-6
"""Floor applied to the measurement so black frames give a finite (clamped) correction."""


def subsampled_luminance(
    image: NDArray[np.float32], max_long_edge_px: int = METERING_MAX_LONG_EDGE_PX
) -> NDArray[np.float32]:
    """Luminance of ``image`` (scene-linear ``(H, W, 3)``) on a strided grid.

    Returns:
        ``(h, w)`` float32 luminance with a long edge of at most about ``max_long_edge_px``.

    Raises:
        MeteringError: The image is empty.
    """
    if image.ndim != 3 or image.size == 0:  # noqa: PLR2004
        msg = f"cannot meter an empty image of shape {image.shape}"
        raise MeteringError(msg)
    step = max(1, math.ceil(max(image.shape[:2]) / max_long_edge_px))
    return luminance(image[::step, ::step])


def trimmed_mean(values: NDArray[np.float32], trim_fraction: float) -> float:
    """Mean of ``values`` after dropping ``trim_fraction`` of the lowest and of the highest.

    Non-finite values are ignored.

    Raises:
        MeteringError: No finite value.
    """
    flat = np.sort(values[np.isfinite(values)], axis=None)
    if flat.size == 0:
        msg = "no finite luminance to meter"
        raise MeteringError(msg)
    cut = int(flat.size * trim_fraction)
    kept = flat[cut : flat.size - cut] if flat.size - 2 * cut > 0 else flat
    return float(np.mean(kept, dtype=np.float64))


def weighted_median(values: NDArray[np.float32], weights: NDArray[np.float32]) -> float:
    """Weighted median of ``values`` (same shape as ``weights``); non-finite values ignored.

    Raises:
        MeteringError: No finite value with positive weight.
    """
    v = values.ravel()
    w = weights.ravel().astype(np.float64)
    keep = np.isfinite(v) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        msg = "no finite luminance to meter"
        raise MeteringError(msg)
    order = np.argsort(v, kind="stable")
    cumulative = np.cumsum(w[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(v[order][index])


def correction_ev(measured: float, target: float, max_correction_ev: float) -> float:
    """``log2(target / measured)`` clamped to ``±max_correction_ev`` (EV stops)."""
    ev = math.log2(target / max(measured, MIN_MEASURABLE_LUMINANCE))
    return float(min(max(ev, -max_correction_ev), max_correction_ev))
