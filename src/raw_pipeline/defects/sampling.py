"""Same-colour CFA sampling helpers shared by the defect correctors.

On a Bayer sensor the nearest photosite of the same colour is
:data:`~raw_pipeline.core.raw.CFA_TILE_SIZE` photosites away, so every
neighbour lookup here walks in steps of that size and only reads photosites
outside the defect mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.errors import DefectCorrectionError
from raw_pipeline.core.raw import CFA_TILE_SIZE, CFAFrame, ColumnDefect, Defect, DefectMap, PixelDefect

MAX_SEARCH_STEPS: Final = 16
"""How many same-colour steps (``CFA_TILE_SIZE`` photosites each) a neighbour search may walk."""

Step = tuple[int, int]
"""A ``(dy, dx)`` offset in photosites between two same-colour photosites."""

LEFT: Final[Step] = (0, -CFA_TILE_SIZE)
RIGHT: Final[Step] = (0, CFA_TILE_SIZE)
UP: Final[Step] = (-CFA_TILE_SIZE, 0)
DOWN: Final[Step] = (CFA_TILE_SIZE, 0)
RING: Final[tuple[Step, ...]] = tuple(
    (dy, dx)
    for dy in (-CFA_TILE_SIZE, 0, CFA_TILE_SIZE)
    for dx in (-CFA_TILE_SIZE, 0, CFA_TILE_SIZE)
    if (dy, dx) != (0, 0)
)
"""The 8 nearest same-colour directions around a photosite."""

_UINT16_MAX: Final = int(np.iinfo(np.uint16).max)


def defect_mask(frame: CFAFrame, defects: DefectMap) -> NDArray[np.bool_]:
    """Rasterise ``defects`` over ``frame``.

    Raises:
        DefectCorrectionError: If an entry lies completely outside the sensor.
    """
    try:
        return defects.mask(frame.height, frame.width)
    except ValueError as exc:
        raise DefectCorrectionError(str(exc)) from exc


def defect_coordinates(defect: Defect, height: int, width: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the ``(ys, xs)`` photosites covered by one defect, clipped to the sensor, row-major."""
    if isinstance(defect, PixelDefect):
        rows = range(defect.y, min(defect.y + 1, height))
        cols = range(defect.x, min(defect.x + 1, width))
    elif isinstance(defect, ColumnDefect):
        rows = range(max(defect.start or 0, 0), min(height if defect.end is None else defect.end, height))
        cols = range(max(defect.index, 0), min(defect.index + defect.width, width))
    else:
        rows = range(max(defect.index, 0), min(defect.index + defect.width, height))
        cols = range(max(defect.start or 0, 0), min(width if defect.end is None else defect.end, width))
    ys, xs = np.meshgrid(np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp), indexing="ij")
    return ys.ravel(), xs.ravel()


@dataclass(frozen=True)
class SameColourSampler:
    """Reads same-colour neighbours of photosites, skipping masked ones.

    Attributes:
        data: ``(H, W)`` raw values.
        valid: ``(H, W)`` mask of photosites that may be read (outside the defect mask).
    """

    data: NDArray[np.uint16]
    valid: NDArray[np.bool_]

    def nearest(
        self, ys: NDArray[np.intp], xs: NDArray[np.intp], step: Step, max_steps: int = MAX_SEARCH_STEPS
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Walk from each ``(y, x)`` along ``step`` to the first valid photosite.

        Args:
            ys: Start rows.
            xs: Start columns.
            step: Same-colour offset walked repeatedly (the start itself is never read).
            max_steps: Maximum number of steps.

        Returns:
            ``(values, steps)``: raw value found and how many steps away it is;
            ``steps == 0`` where nothing valid was found inside the sensor.
        """
        height, width = self.data.shape
        values = np.zeros(ys.shape, dtype=np.float64)
        steps = np.zeros(ys.shape, dtype=np.int64)
        pending = np.ones(ys.shape, dtype=np.bool_)
        for k in range(1, max_steps + 1):
            cy = ys + k * step[0]
            cx = xs + k * step[1]
            candidates = np.flatnonzero(pending & (cy >= 0) & (cy < height) & (cx >= 0) & (cx < width))
            if candidates.size == 0:
                break
            hits = candidates[self.valid[cy[candidates], cx[candidates]]]
            values[hits] = self.data[cy[hits], cx[hits]]
            steps[hits] = k
            pending[hits] = False
        return values, steps

    def mean_of_nearest(
        self, ys: NDArray[np.intp], xs: NDArray[np.intp], directions: tuple[Step, ...]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Mean of the nearest valid same-colour photosite found along each direction.

        Returns:
            ``(values, found)``; ``found`` is ``False`` where no direction found a sample.
        """
        total = np.zeros(ys.shape, dtype=np.float64)
        count = np.zeros(ys.shape, dtype=np.int64)
        for step in directions:
            values, steps = self.nearest(ys, xs, step)
            hit = steps > 0
            total[hit] += values[hit]
            count[hit] += 1
        found = count > 0
        return np.divide(total, count, out=np.zeros_like(total), where=found), found


def require_found(found: NDArray[np.bool_], ys: NDArray[np.intp], xs: NDArray[np.intp]) -> None:
    """Fail when some photosite had no valid same-colour neighbour within the search radius.

    Raises:
        DefectCorrectionError: Naming the first such photosite.
    """
    if found.all():
        return
    first = int(np.flatnonzero(~found)[0])
    reach = MAX_SEARCH_STEPS * CFA_TILE_SIZE
    msg = (
        f"no valid same-colour neighbour within {reach} px of photosite x={xs[first]}, y={ys[first]}; "
        "the defect map covers too large an area"
    )
    raise DefectCorrectionError(msg)


def replace(frame: CFAFrame, ys: NDArray[np.intp], xs: NDArray[np.intp], values: NDArray[np.float64]) -> CFAFrame:
    """Return a copy of ``frame`` with photosites ``(ys, xs)`` set to ``values`` (rounded, uint16-clipped)."""
    data = frame.data.copy()
    data[ys, xs] = np.clip(np.rint(values), 0, _UINT16_MAX).astype(np.uint16)
    return frame.with_data(data)
