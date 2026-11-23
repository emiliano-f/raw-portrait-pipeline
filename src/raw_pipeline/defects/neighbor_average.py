"""``neighbor_average``: mean of the nearest same-colour photosites across the defect."""

from __future__ import annotations

from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.raw import CFAFrame, ColumnDefect, Defect, DefectCorrectionResult, DefectMap, RowDefect
from raw_pipeline.core.registries import defect_corrector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects.sampling import (
    DOWN,
    LEFT,
    RIGHT,
    RING,
    UP,
    SameColourSampler,
    Step,
    defect_coordinates,
    defect_mask,
    replace,
    require_found,
)

_ACROSS_COLUMN: Final = (LEFT, RIGHT)
_ACROSS_ROW: Final = (UP, DOWN)


def _directions(defect: Defect) -> tuple[Step, ...]:
    if isinstance(defect, ColumnDefect):
        return _ACROSS_COLUMN
    if isinstance(defect, RowDefect):
        return _ACROSS_ROW
    return RING


@defect_corrector_registry.register("neighbor_average")
class NeighborAverageCorrector:
    """Replaces each defective photosite with the mean of its nearest valid same-colour neighbours.

    Neighbours are searched perpendicular to the line (left/right for columns,
    up/down for rows) and in all 8 same-colour directions for single pixels,
    skipping every masked photosite, so bands wider than one photosite and
    image borders (one-sided mean) are handled. Where the perpendicular search
    finds nothing (e.g. where a row and a column defect cross) the 8-direction
    search is used instead. Simple and robust on smooth areas; blurs edges that
    cross the line (use ``directional`` there).
    """

    name: ClassVar[str] = "neighbor_average"

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
        sampler = SameColourSampler(frame.data, ~defect_mask(frame, defects))
        done = np.zeros(frame.data.shape, dtype=np.bool_)
        all_ys: list[NDArray[np.intp]] = []
        all_xs: list[NDArray[np.intp]] = []
        all_values: list[NDArray[np.float64]] = []
        for defect in defects:
            ys, xs = defect_coordinates(defect, frame.height, frame.width)
            todo = ~done[ys, xs]
            ys, xs = ys[todo], xs[todo]
            done[ys, xs] = True
            values, found = sampler.mean_of_nearest(ys, xs, _directions(defect))
            if not found.all():
                retry_values, retry_found = sampler.mean_of_nearest(ys[~found], xs[~found], RING)
                values[~found] = retry_values
                found[~found] = retry_found
            require_found(found, ys, xs)
            all_ys.append(ys)
            all_xs.append(xs)
            all_values.append(values)
        ys, xs = np.concatenate(all_ys), np.concatenate(all_xs)
        corrected = replace(frame, ys, xs, np.concatenate(all_values))
        return DefectCorrectionResult(frame=corrected, corrected_pixel_count=int(ys.size), method=self.name)
