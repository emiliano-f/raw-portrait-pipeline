"""Robust per-frame statistics used by :mod:`raw_pipeline.defects.detection`.

All residuals compare a photosite with its same-colour neighbours (offsets
that are multiples of :data:`~raw_pipeline.core.raw.CFA_TILE_SIZE`), which is
a high-pass filter: smooth scene content cancels out, so the same statistics
work on dark frames and on normal portraits.
"""

from __future__ import annotations

import warnings
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

MAX_LINE_PASSES: Final = 8
"""Upper bound on the line-scoring passes of :func:`line_segment_z`."""

_T: Final = CFA_TILE_SIZE
LINE_REACH_STEPS: Final = 3
"""Same-colour neighbours taken on each side for line residuals; the median stays clean for bands up to 4 px wide."""

LINE_SEARCH_STEPS: Final = 8
"""How far (in same-colour steps) each side is searched for usable neighbours once lines are excluded."""

LINE_REACH_PX: Final = LINE_REACH_STEPS * _T
"""Columns closer than this to the sensor border may be scored on one side only."""

LINE_SIDES: Final = tuple(tuple((0, sign * k * _T) for k in range(1, LINE_SEARCH_STEPS + 1)) for sign in (-1, 1))
"""Same-colour photosites along the row, nearest first, left side then right side, used to detect column defects."""

RING_NEIGHBOURS: Final = tuple((dy, dx) for dy in (-_T, 0, _T) for dx in (-_T, 0, _T) if (dy, dx) != (0, 0))
"""The 8 nearest same-colour neighbours, used to detect pixel defects."""

MAX_IGNORED_RING_FRACTION: Final = 1 / 3
"""Fraction of a pixel's in-frame ring neighbours that may lie on detected lines before it is no longer scored."""


def same_colour_residual(
    data: NDArray[np.float32],
    offsets: tuple[tuple[int, int], ...],
    ignore: NDArray[np.bool_] | None = None,
    missing: float = 0.0,
    nearest: int | None = None,
) -> NDArray[np.float32]:
    """Return ``data`` minus the median of its same-colour neighbours at ``offsets``.

    Neighbours outside the frame are skipped (mirroring would count a defect
    at the border twice). The median keeps the residual of a photosite next
    to a hot line close to zero.

    Args:
        data: ``(H, W)`` raw values as float32.
        offsets: ``(dy, dx)`` neighbour offsets, multiples of ``CFA_TILE_SIZE``.
        ignore: Optional ``(H, W)`` mask of photosites never used as neighbours
            (e.g. lines already found).
        missing: Residual of photosites without any usable neighbour.
        nearest: Use only the first ``nearest`` usable neighbours, in
            ``offsets`` order (nearest first), instead of all of them.
    """
    if nearest is not None and ignore is None:
        offsets = offsets[:nearest]  # nothing is skipped inside the frame; further offsets only leave it
    height, width = data.shape
    pad = max(abs(v) for offset in offsets for v in offset)
    source = data if ignore is None else np.where(ignore, np.float32(np.nan), data)
    padded = np.pad(source, pad, constant_values=np.nan)
    out = np.empty_like(data)
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(y0 + CHUNK_ROWS, height)
        neighbours = np.stack(
            [padded[pad + y0 + dy : pad + y1 + dy, pad + dx : pad + width + dx] for dy, dx in offsets]
        )
        if nearest is not None and len(offsets) > nearest:
            closest = neighbours[:nearest]
            short = np.isnan(closest).any(axis=0)  # only near skipped photosites and the border
            closest[:, short] = _first_valid(neighbours[:, short], nearest)
            neighbours = closest
        reference = _nan_median_first_axis(neighbours)
        out[y0:y1] = np.where(np.isnan(reference), np.float32(missing), data[y0:y1] - reference)
    return out


def _first_valid(stack: NDArray[np.float32], count: int) -> NDArray[np.float32]:
    """The first ``count`` non-NaN values along axis 0 (NaN where fewer exist)."""
    valid = ~np.isnan(stack)
    rank = np.cumsum(valid, axis=0)
    picked = np.empty((count, *stack.shape[1:]), dtype=np.float32)
    for j in range(count):
        chosen = valid & (rank == j + 1)
        index = np.argmax(chosen, axis=0)
        picked[j] = np.take_along_axis(stack, index[None], axis=0)[0]
        picked[j][~chosen.any(axis=0)] = np.nan
    return picked


def _nan_median_first_axis(stack: NDArray[np.float32]) -> NDArray[np.float32]:
    """Median over axis 0 ignoring NaN (NaN where all are NaN); much faster than ``np.nanmedian`` on short axes."""
    if stack.shape[0] == 3:  # noqa: PLR2004 - closed form for the line sides, the hot path
        return _nan_median_of_three(stack)
    ordered = np.sort(stack, axis=0)  # NaN sorts last
    valid = np.count_nonzero(~np.isnan(stack), axis=0)
    last = stack.shape[0] - 1
    low = np.take_along_axis(ordered, np.clip((valid - 1) // 2, 0, last)[None], axis=0)[0]
    high = np.take_along_axis(ordered, np.clip(valid // 2, 0, last)[None], axis=0)[0]
    return np.where(valid > 0, (low + high) / np.float32(2.0), np.float32(np.nan)).astype(np.float32)


def _nan_median_of_three(stack: NDArray[np.float32]) -> NDArray[np.float32]:
    valid = np.count_nonzero(~np.isnan(stack), axis=0)
    total = np.nansum(stack, axis=0)
    full = total - np.fmax.reduce(stack, axis=0) - np.fmin.reduce(stack, axis=0)
    choices = [full, total / np.float32(2.0), total]
    median: NDArray[np.float32] = np.select([valid == 3, valid == 2, valid == 1], choices, np.nan).astype(np.float32)  # noqa: PLR2004
    return median


def robust_z(values: NDArray[np.float32], axis: int | None = None) -> NDArray[np.float32]:
    """Standardise ``values`` with the median and MAD along ``axis`` (all values when ``None``)."""
    center = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - center), axis=axis, keepdims=True)
    sigma = np.maximum(MAD_TO_SIGMA * mad, MIN_ROBUST_SIGMA_RAW)
    return ((values - center) / sigma).astype(np.float32)


def _nan_robust_z(values: NDArray[np.float32], axis: int) -> NDArray[np.float32]:
    """:func:`robust_z` ignoring NaN; NaN stays NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices stay NaN
        center = np.nanmedian(values, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(values - center), axis=axis, keepdims=True)
    sigma = np.maximum(MAD_TO_SIGMA * np.nan_to_num(mad), MIN_ROBUST_SIGMA_RAW)
    return ((values - center) / sigma).astype(np.float32)


def _side_segment_z(
    data: NDArray[np.float32],
    side: tuple[tuple[int, int], ...],
    segment_px: int,
    ignore: NDArray[np.bool_] | None,
) -> NDArray[np.float32]:
    residual = same_colour_residual(data, side, ignore, missing=np.nan, nearest=LINE_REACH_STEPS)
    height = data.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns at the border have no neighbour on one side
        medians = np.stack(
            [np.nanmedian(residual[y0 : y0 + segment_px], axis=0) for y0 in range(0, height, segment_px)]
        ).astype(np.float32)
    z = np.empty_like(medians)
    for parity in range(_T):
        z[:, parity::_T] = _nan_robust_z(medians[:, parity::_T], axis=1)
    return z


def _line_z(data: NDArray[np.float32], segment_px: int, ignore: NDArray[np.bool_] | None) -> NDArray[np.float32]:
    left, right = (_side_segment_z(data, side, segment_px, ignore) for side in LINE_SIDES)
    agree = np.sign(left) == np.sign(right)
    both = np.where(agree, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), np.float32(0.0))
    width = data.shape[1]
    columns = np.arange(width)
    near_border = (columns < LINE_REACH_PX) | (columns >= width - LINE_REACH_PX)
    one = np.where(near_border, np.where(np.isnan(left), right, left), np.float32(0.0))
    return np.nan_to_num(np.where(np.isnan(left) | np.isnan(right), one, both)).astype(np.float32)


def line_segment_z(data: NDArray[np.float32], segment_px: int, threshold_sigma: float) -> NDArray[np.float32]:
    """Outlier score of every column, per segment of ``segment_px`` rows.

    Each column's residual against the median of its :data:`LINE_REACH_STEPS`
    nearest usable same-colour neighbours on the left and on the right
    (:data:`LINE_SIDES`; segments already flagged are skipped) is reduced to its median per segment and
    standardised across columns of the same parity (same CFA colours), per
    side. A defect stands out from *both* sides, so the score is the smaller
    of the two z-scores when they agree in sign and 0 otherwise: a sharp
    scene edge, which differs from one side only, scores 0. A side without
    usable neighbours counts as missing; within :data:`LINE_REACH_PX` of the
    sensor border the other side is used alone, elsewhere the score is 0.

    Segments scoring above ``threshold_sigma`` are then excluded as
    neighbours and the scores recomputed, until no new segment is flagged
    (at most :data:`MAX_LINE_PASSES` passes). Each pass uncovers the next
    column of a band that the previous one hid, e.g. a band at the sensor
    border whose inner columns only had defective neighbours on one side.
    Each segment keeps its strongest score over all passes, so a segment
    flagged once stays flagged.

    Returns:
        ``(ceil(H / segment_px), W)`` robust z-scores.
    """
    height = data.shape[0]
    best = _line_z(data, segment_px, None)
    flagged = np.abs(best) > threshold_sigma
    for _ in range(MAX_LINE_PASSES - 1):
        if not flagged.any():
            break
        z = _line_z(data, segment_px, np.repeat(flagged, segment_px, axis=0)[:height])
        best = np.where(np.abs(z) > np.abs(best), z, best)
        grown = np.abs(best) > threshold_sigma
        if np.array_equal(grown, flagged):
            break
        flagged = grown
    return best


def _ring_in_frame(height: int, width: int) -> NDArray[np.uint8]:
    """Number of :data:`RING_NEIGHBOURS` of every photosite that are inside the frame."""

    def lines_in(length: int) -> NDArray[np.uint8]:
        index = np.arange(length)
        return (1 + (index >= _T) + (index < length - _T)).astype(np.uint8)

    return (lines_in(height)[:, None] * lines_in(width)[None, :] - 1).astype(np.uint8)


def _ring_count(mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Number of :data:`RING_NEIGHBOURS` of every photosite that are inside the frame and set in ``mask``."""
    height, width = mask.shape
    padded = np.pad(mask, _T, constant_values=False)
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in RING_NEIGHBOURS:
        count += padded[_T + dy : _T + height + dy, _T + dx : _T + width + dx]
    return count


def pixel_z(data: NDArray[np.float32], ignore: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """Outlier score of every photosite against its 8 same-colour neighbours, per CFA colour plane.

    Args:
        data: ``(H, W)`` raw values as float32.
        ignore: Photosites not used as neighbours (lines flagged in this frame),
            so pixels where two hot lines cross are not mistaken for pixel defects.
            A photosite with more than :data:`MAX_IGNORED_RING_FRACTION` of
            its in-frame neighbours ignored scores 0: its remaining neighbours all lie on
            one side, where a scene edge could pass for a defect.
    """
    if ignore is not None and not ignore.any():
        ignore = None
    residual = same_colour_residual(data, RING_NEIGHBOURS, ignore)
    if ignore is not None:
        residual[_ring_count(ignore) > MAX_IGNORED_RING_FRACTION * _ring_in_frame(*ignore.shape)] = 0.0
    z = np.empty_like(residual)
    for py in range(_T):
        for px in range(_T):
            z[py::_T, px::_T] = robust_z(residual[py::_T, px::_T])
    return z
