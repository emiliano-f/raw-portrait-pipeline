"""``directional``: edge-aware interpolation along the lowest-gradient direction."""

from __future__ import annotations

import math
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.raw import CFA_TILE_SIZE, CFAFrame, DefectCorrectionResult, DefectMap
from raw_pipeline.core.registries import defect_corrector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects.sampling import RING, SameColourSampler, Step, defect_mask, replace, require_found

_T: Final = CFA_TILE_SIZE
DIRECTIONS: Final[tuple[tuple[Step, Step], ...]] = (
    ((0, _T), (_T, 0)),  # horizontal
    ((_T, 0), (0, _T)),  # vertical
    ((_T, _T), (_T, -_T)),  # diagonal
    ((_T, -_T), (_T, _T)),  # anti-diagonal
)
"""Candidate interpolation directions, each with the step to its parallel lines (used for the gradient)."""

PARALLEL_LINE_OFFSETS: Final = (-1, 0, 1)
"""Parallel lines (in perpendicular same-colour steps) whose gradients are averaged per direction."""


def _endpoints(
    sampler: SameColourSampler, ys: NDArray[np.intp], xs: NDArray[np.intp], step: Step
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    back, back_k = sampler.nearest(ys, xs, (-step[0], -step[1]))
    fwd, fwd_k = sampler.nearest(ys, xs, step)
    return back, back_k, fwd, fwd_k


def _interpolate(
    sampler: SameColourSampler, ys: NDArray[np.intp], xs: NDArray[np.intp], step: Step
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Distance-weighted linear interpolation between the endpoints; ``False`` where one side is missing."""
    back, back_k, fwd, fwd_k = _endpoints(sampler, ys, xs, step)
    two_sided = (back_k > 0) & (fwd_k > 0)
    span = np.where(two_sided, back_k + fwd_k, 1)
    return (back * fwd_k + fwd * back_k) / span, two_sided


def _mean_gradient(
    sampler: SameColourSampler, ys: NDArray[np.intp], xs: NDArray[np.intp], direction: tuple[Step, Step]
) -> NDArray[np.float64]:
    """Mean ``|endpoint difference| / distance`` over the line through each photosite and its parallels."""
    step, perpendicular = direction
    height, width = sampler.data.shape
    step_length = math.hypot(*step)
    total = np.zeros(ys.shape, dtype=np.float64)
    count = np.zeros(ys.shape, dtype=np.int64)
    for offset in PARALLEL_LINE_OFFSETS:
        qy, qx = ys + offset * perpendicular[0], xs + offset * perpendicular[1]
        inside = np.flatnonzero((qy >= 0) & (qy < height) & (qx >= 0) & (qx < width))
        back, back_k, fwd, fwd_k = _endpoints(sampler, qy[inside], qx[inside], step)
        ok = (back_k > 0) & (fwd_k > 0)
        idx = inside[ok]
        total[idx] += np.abs(fwd[ok] - back[ok]) / ((back_k[ok] + fwd_k[ok]) * step_length)
        count[idx] += 1
    return np.divide(total, count, out=np.full_like(total, np.inf), where=count > 0)


@defect_corrector_registry.register("directional")
class DirectionalCorrector:
    """Edge-aware correction: interpolates each photosite along its smoothest direction.

    For every defective photosite and each candidate direction (horizontal,
    vertical, both diagonals), the nearest valid same-colour photosites on
    both sides are found. The direction's gradient is the mean absolute
    difference per pixel of distance between such endpoint pairs, measured on
    the line through the photosite and on the two parallel same-colour lines
    next to it. The photosite is set to the distance-weighted linear
    interpolation along the direction with the lowest gradient (first in
    :data:`DIRECTIONS` order on ties), so an edge crossing the line (hair,
    eyelashes) is continued instead of blurred. Photosites with no two-sided
    direction (image borders) use the mean of the nearest valid neighbours.
    """

    name: ClassVar[str] = "directional"

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:  # noqa: ARG003 - no settings to read
        """Build the corrector; it has no configuration."""
        return cls()

    def correct(self, frame: CFAFrame, defects: DefectMap) -> DefectCorrectionResult:
        """Return a corrected copy of ``frame``; every masked photosite is replaced once.

        Raises:
            DefectCorrectionError: An entry lies outside the sensor, or a
                photosite has no valid same-colour neighbour within reach.
        """
        if len(defects) == 0:
            return DefectCorrectionResult(frame=frame, corrected_pixel_count=0, method=self.name)
        mask = defect_mask(frame, defects)
        sampler = SameColourSampler(frame.data, ~mask)
        ys, xs = np.nonzero(mask)
        candidates = [_interpolate(sampler, ys, xs, step) for step, _ in DIRECTIONS]
        gradients = np.stack([_mean_gradient(sampler, ys, xs, d) for d in DIRECTIONS])
        gradients[~np.stack([two_sided for _, two_sided in candidates])] = np.inf
        best = np.argmin(gradients, axis=0)
        values = np.stack([value for value, _ in candidates])[best, np.arange(ys.size)]
        one_sided = ~np.isfinite(gradients.min(axis=0))
        if one_sided.any():
            fallback, found = sampler.mean_of_nearest(ys[one_sided], xs[one_sided], RING)
            require_found(found, ys[one_sided], xs[one_sided])
            values[one_sided] = fallback
        corrected = replace(frame, ys, xs, values)
        return DefectCorrectionResult(frame=corrected, corrected_pixel_count=int(ys.size), method=self.name)
