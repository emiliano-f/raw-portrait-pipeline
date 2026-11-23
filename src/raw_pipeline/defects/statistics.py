"""Robust per-frame statistics used by :mod:`raw_pipeline.defects.detection`.

All residuals compare a photosite with its same-colour neighbours (offsets
that are multiples of :data:`~raw_pipeline.core.raw.CFA_TILE_SIZE`), which is
a high-pass filter: smooth scene content cancels out, so the same statistics
work on dark frames and on normal portraits.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.raw import CFA_TILE_SIZE

MAD_TO_SIGMA: Final = 1.4826
"""Scale from median absolute deviation to the standard deviation of a normal distribution."""

MIN_ROBUST_SIGMA_RAW: Final = 0.5
"""Floor of the robust standard deviation, in raw units, so noise-free data cannot divide by zero."""

CHUNK_ROWS: Final = 256
"""Rows processed at once when stacking neighbours, to bound memory on full-size frames."""

_T: Final = CFA_TILE_SIZE
LINE_REACH_STEPS: Final = 3
"""Same-colour neighbours taken on each side for line residuals; the median stays clean for bands up to 4 px wide."""

LINE_NEIGHBOURS: Final = tuple((0, sign * k * _T) for sign in (-1, 1) for k in range(1, LINE_REACH_STEPS + 1))
"""Same-colour neighbours of a photosite along its row, used to detect column defects."""

RING_NEIGHBOURS: Final = tuple((dy, dx) for dy in (-_T, 0, _T) for dx in (-_T, 0, _T) if (dy, dx) != (0, 0))
"""The 8 nearest same-colour neighbours, used to detect pixel defects."""


def same_colour_residual(
    data: NDArray[np.float32],
    offsets: tuple[tuple[int, int], ...],
    ignore: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Return ``data`` minus the median of its same-colour neighbours at ``offsets``.

    Borders are mirrored (``reflect``); even offsets keep the CFA colour of
    the mirrored photosite. The median keeps the residual of a photosite next
    to a hot line close to zero.

    Args:
        data: ``(H, W)`` raw values as float32.
        offsets: ``(dy, dx)`` neighbour offsets, multiples of ``CFA_TILE_SIZE``.
        ignore: Optional ``(H, W)`` mask of photosites never used as neighbours
            (e.g. lines already found). Photosites without any usable
            neighbour get a residual of 0.
    """
    height, width = data.shape
    pad = max(abs(v) for offset in offsets for v in offset)
    source = data if ignore is None else np.where(ignore, np.float32(np.nan), data)
    padded = np.pad(source, pad, mode="reflect")
    out = np.empty_like(data)
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(y0 + CHUNK_ROWS, height)
        neighbours = np.stack(
            [padded[pad + y0 + dy : pad + y1 + dy, pad + dx : pad + width + dx] for dy, dx in offsets]
        )
        reference = _nan_median_first_axis(neighbours)
        out[y0:y1] = np.where(np.isnan(reference), np.float32(0.0), data[y0:y1] - reference)
    return out


def _nan_median_first_axis(stack: NDArray[np.float32]) -> NDArray[np.float32]:
    """Median over axis 0 ignoring NaN (NaN where all are NaN); much faster than ``np.nanmedian`` on short axes."""
    ordered = np.sort(stack, axis=0)  # NaN sorts last
    valid = np.count_nonzero(~np.isnan(stack), axis=0)
    last = stack.shape[0] - 1
    low = np.take_along_axis(ordered, np.clip((valid - 1) // 2, 0, last)[None], axis=0)[0]
    high = np.take_along_axis(ordered, np.clip(valid // 2, 0, last)[None], axis=0)[0]
    return np.where(valid > 0, (low + high) / np.float32(2.0), np.float32(np.nan)).astype(np.float32)


def robust_z(values: NDArray[np.float32], axis: int | None = None) -> NDArray[np.float32]:
    """Standardise ``values`` with the median and MAD along ``axis`` (all values when ``None``)."""
    center = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - center), axis=axis, keepdims=True)
    sigma = np.maximum(MAD_TO_SIGMA * mad, MIN_ROBUST_SIGMA_RAW)
    return ((values - center) / sigma).astype(np.float32)


def line_segment_z(data: NDArray[np.float32], segment_px: int) -> NDArray[np.float32]:
    """Outlier score of every column, per segment of ``segment_px`` rows.

    Each column's residual against its row neighbours (:data:`LINE_NEIGHBOURS`)
    is reduced to its median per segment; segments are then standardised
    across columns of the same parity (same CFA colours).

    Returns:
        ``(ceil(H / segment_px), W)`` robust z-scores.
    """
    residual = same_colour_residual(data, LINE_NEIGHBOURS)
    height = data.shape[0]
    medians = np.stack(
        [np.median(residual[y0 : y0 + segment_px], axis=0) for y0 in range(0, height, segment_px)]
    ).astype(np.float32)
    z = np.empty_like(medians)
    for parity in range(_T):
        z[:, parity::_T] = robust_z(medians[:, parity::_T], axis=1)
    return z


def pixel_z(data: NDArray[np.float32], ignore: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """Outlier score of every photosite against its 8 same-colour neighbours, per CFA colour plane.

    Args:
        data: ``(H, W)`` raw values as float32.
        ignore: Photosites not used as neighbours (lines flagged in this frame),
            so pixels where two hot lines cross are not mistaken for pixel defects.
    """
    residual = same_colour_residual(data, RING_NEIGHBOURS, ignore)
    z = np.empty_like(residual)
    for py in range(_T):
        for px in range(_T):
            z[py::_T, px::_T] = robust_z(residual[py::_T, px::_T])
    return z
