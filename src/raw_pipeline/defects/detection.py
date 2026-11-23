"""``StatisticalDefectDetector``: finds hot/stuck lines and pixels for ``detect-defects``."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, DefectMap, DetectedDefect, PixelDefect, RowDefect
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects.statistics import line_segment_z, pixel_z

logger = logging.getLogger(__name__)

SEGMENT_LENGTH_PX: Final = 64
"""Length of the line segments scored independently; partial lines are located to this granularity."""

FRACTION_TOLERANCE: Final = 1e-9
"""Slack on ``min_frame_fraction * frames`` so float rounding cannot raise the required count (0.56 * 25)."""

MIN_LINE_SEGMENTS: Final = 2
"""Consecutive outlier segments needed to report a line rather than isolated pixels."""


@dataclass(frozen=True)
class _Run:
    line: int
    start: int
    end: int
    score: float
    frames_flagged: int


@dataclass
class _Band:
    index: int
    width: int
    start: int
    end: int
    score: float
    frames_flagged: int

    def accepts(self, run: _Run) -> bool:
        return run.line == self.index + self.width and (run.start, run.end) == (self.start, self.end)

    def extend(self, run: _Run) -> None:
        self.width += 1
        self.score = max(self.score, run.score)
        self.frames_flagged = max(self.frames_flagged, run.frames_flagged)


class _LineAccumulator:
    """Per-segment outlier counts of columns (pass transposed data for rows)."""

    def __init__(self, threshold_sigma: float, segment_px: int, *, one_sided_border: bool) -> None:
        self._threshold = threshold_sigma
        self._segment_px = segment_px
        self._one_sided_border = one_sided_border
        self._hits: NDArray[np.int64] | None = None
        self._score_sum: NDArray[np.float64] | None = None

    def add(self, data: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Score one frame; return its ``(H, W)`` mask of photosites on outlier segments."""
        z = np.abs(line_segment_z(data, self._segment_px, self._threshold, one_sided_border=self._one_sided_border))
        flagged = z > self._threshold
        if self._hits is None or self._score_sum is None:
            self._hits = np.zeros(z.shape, dtype=np.int64)
            self._score_sum = np.zeros(z.shape, dtype=np.float64)
        self._hits += flagged
        self._score_sum += np.where(flagged, z, 0.0)
        return np.repeat(flagged, self._segment_px, axis=0)[: data.shape[0]]

    def bands(self, required_frames: int, length: int) -> list[_Band]:
        """Group consistent outlier runs of adjacent lines into bands, sorted by index."""
        if self._hits is None or self._score_sum is None:
            return []
        consistent = self._hits >= required_frames
        min_segments = min(MIN_LINE_SEGMENTS, consistent.shape[0])
        bands: list[_Band] = []
        for run in self._runs(consistent, min_segments, length):
            band = next((b for b in bands if b.accepts(run)), None)
            if band is None:
                bands.append(_Band(run.line, 1, run.start, run.end, run.score, run.frames_flagged))
            else:
                band.extend(run)
        return bands

    def _runs(self, consistent: NDArray[np.bool_], min_segments: int, length: int) -> list[_Run]:
        assert self._hits is not None
        assert self._score_sum is not None
        runs: list[_Run] = []
        for line in np.flatnonzero(consistent.any(axis=0)):
            padded = np.concatenate(([False], consistent[:, line], [False])).astype(np.int8)
            edges = np.flatnonzero(np.diff(padded))
            for first, stop in zip(edges[::2], edges[1::2], strict=True):
                if stop - first < min_segments:
                    continue
                hits = self._hits[first:stop, line]
                runs.append(
                    _Run(
                        line=int(line),
                        start=int(first) * self._segment_px,
                        end=min(int(stop) * self._segment_px, length),
                        score=float(self._score_sum[first:stop, line].sum() / hits.sum()),
                        frames_flagged=int(hits.min()),
                    )
                )
        return runs


class _PixelAccumulator:
    """Per-photosite outlier counts."""

    def __init__(self, threshold_sigma: float, *, score_border: bool) -> None:
        self._threshold = threshold_sigma
        self._score_border = score_border
        self._hits: NDArray[np.uint32] | None = None
        self._score_sum: NDArray[np.float32] | None = None

    def add(self, data: NDArray[np.float32], lines: NDArray[np.bool_]) -> None:
        z = np.abs(pixel_z(data, lines, score_border=self._score_border))
        flagged = z > self._threshold
        if self._hits is None or self._score_sum is None:
            self._hits = np.zeros(z.shape, dtype=np.uint32)
            self._score_sum = np.zeros(z.shape, dtype=np.float32)
        self._hits += flagged
        self._score_sum += np.where(flagged, z, np.float32(0.0))

    def pixels(self, required_frames: int, exclude: NDArray[np.bool_], limit: int) -> list[tuple[int, int, float, int]]:
        """Return ``(x, y, score, frames_flagged)`` of the ``limit`` strongest consistent outliers."""
        if self._hits is None or self._score_sum is None:
            return []
        ys, xs = np.nonzero((self._hits >= required_frames) & ~exclude)
        hits = self._hits[ys, xs]
        scores = self._score_sum[ys, xs] / hits
        order = np.lexsort((xs, ys, -scores))
        if order.size > limit:
            logger.warning("found %d defective pixels; keeping the %d strongest", order.size, limit)
        keep = np.sort(order[:limit])
        return [(int(xs[i]), int(ys[i]), float(scores[i]), int(hits[i])) for i in keep]


class StatisticalDefectDetector:
    """Finds columns, rows and pixels that are consistent statistical outliers across frames.

    Every frame is reduced to high-pass residuals against same-colour
    neighbours (median, so neighbours of a hot line are not flagged):

    * **Lines**: each column's residual is summarised by its median over
      segments of :data:`SEGMENT_LENGTH_PX` rows and scored with robust
      z-scores (median/MAD) across columns of the same CFA parity. Rows use
      the same procedure on the transposed frame. At least
      :data:`MIN_LINE_SEGMENTS` consecutive outlier segments make a line;
      runs that do not span the whole sensor become partial lines
      (``start``/``end``, segment granularity), and adjacent lines with the
      same extent are merged into one entry with ``width > 1``.
    * **Pixels**: each photosite's residual against its 8 same-colour
      neighbours is scored per colour plane. Pixels inside detected lines are
      dropped; at most ``max_pixel_defects`` (strongest first) are reported.

    A candidate must be an outlier (``|z| > outlier_threshold_sigma``) in at
    least ``ceil(min_frame_fraction * frames)`` frames. A single dark frame
    (lens cap on, same ISO) is the most reliable input; on a batch of normal
    photos scene edges are outliers in single frames only and are rejected by
    the frame fraction. Frames are consumed one at a time; memory is a few
    full-frame arrays regardless of the number of frames.

    Columns and rows within 6 photosites of the sensor border are scored
    against their inner neighbours alone, and pixels in the outer 2
    photosites are scored at all, only with ``dark_frames=True``. On a batch
    of photos a stationary scene edge parallel to and near the border looks
    exactly like a border defect, so in batch mode border defects are not
    detected; map them by hand or detect from a dark frame.

    Args:
        outlier_threshold_sigma: Minimum ``|z|`` (robust standard deviations) per frame.
        min_frame_fraction: Fraction of frames in which a candidate must be an outlier, in ``(0, 1]``.
        max_pixel_defects: Maximum number of pixel defects reported.
        segment_px: Segment length in photosites for line scoring.
        dark_frames: The frames are dark frames (no scene), which enables
            one-sided scoring at the sensor border.
    """

    def __init__(
        self,
        outlier_threshold_sigma: float = 8.0,
        min_frame_fraction: float = 0.8,
        max_pixel_defects: int = 500,
        segment_px: int = SEGMENT_LENGTH_PX,
        *,
        dark_frames: bool = False,
    ) -> None:
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self.min_frame_fraction = min_frame_fraction
        self.max_pixel_defects = max_pixel_defects
        self.segment_px = segment_px
        self.dark_frames = dark_frames

    @classmethod
    def from_config(cls, config: DefectDetectionConfig, *, dark_frames: bool = False) -> Self:
        """Build from the ``defects.detection`` config section."""
        return cls(
            config.outlier_threshold_sigma,
            config.min_frame_fraction,
            config.max_pixel_defects,
            dark_frames=dark_frames,
        )

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from ``env.config.defects.detection``."""
        return cls.from_config(app_config(env).defects.detection)

    def detect(self, frames: Iterable[CFAFrame]) -> list[DetectedDefect]:
        """Analyse ``frames`` and return defects: columns, then rows, then pixels (row-major).

        Raises:
            ValueError: If the frames do not all have the same shape.
        """
        columns = _LineAccumulator(self.outlier_threshold_sigma, self.segment_px, one_sided_border=self.dark_frames)
        rows = _LineAccumulator(self.outlier_threshold_sigma, self.segment_px, one_sided_border=self.dark_frames)
        pixels = _PixelAccumulator(self.outlier_threshold_sigma, score_border=self.dark_frames)
        shape: tuple[int, ...] | None = None
        total = 0
        for frame in frames:
            if shape is None:
                shape = frame.data.shape
            elif frame.data.shape != shape:
                msg = f"all frames must have the same size; got {frame.data.shape} after {shape}"
                raise ValueError(msg)
            data = frame.data.astype(np.float32)
            frame_lines = columns.add(data) | rows.add(data.T).T
            pixels.add(data, frame_lines)
            total += 1
        if shape is None:
            return []
        height, width = shape
        required = max(1, math.ceil(self.min_frame_fraction * total - FRACTION_TOLERANCE))
        lines = [
            *(self._column(b, height, total) for b in columns.bands(required, height)),
            *(self._row(b, width, total) for b in rows.bands(required, width)),
        ]
        line_mask = DefectMap(tuple(d.defect for d in lines)).mask(height, width)
        found = lines + [
            DetectedDefect(PixelDefect(x=x, y=y), score, flagged, total)
            for x, y, score, flagged in pixels.pixels(required, line_mask, self.max_pixel_defects)
        ]
        logger.info("analysed %d frame(s): %d defect(s) found", total, len(found))
        return found

    @staticmethod
    def _extent(band: _Band, length: int) -> tuple[int | None, int | None]:
        return (None if band.start == 0 else band.start), (None if band.end == length else band.end)

    def _column(self, band: _Band, height: int, total: int) -> DetectedDefect:
        start, end = self._extent(band, height)
        return DetectedDefect(ColumnDefect(band.index, band.width, start, end), band.score, band.frames_flagged, total)

    def _row(self, band: _Band, width: int, total: int) -> DetectedDefect:
        start, end = self._extent(band, width)
        return DetectedDefect(RowDefect(band.index, band.width, start, end), band.score, band.frames_flagged, total)
