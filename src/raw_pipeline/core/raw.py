"""Raw-level (pre-demosaic) value objects: CFA frames and sensor defect maps.

All coordinates here are **raw sensor coordinates** of the *visible* CFA area
as exposed by LibRaw (``raw_image_visible``): ``x`` is the column, ``y`` the
row, origin at the top-left of the visible area, before any rotation, flip or
crop applied when developing. See ADR 0002.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

CFA_TILE_SIZE = 2
"""Bayer tile edge length in pixels; same-colour neighbours are this far apart."""


@dataclass(frozen=True)
class CFAFrame:
    """Undemosaiced Bayer data of one exposure.

    Attributes:
        data: ``(H, W)`` uint16 raw values of the visible sensor area. Never
            mutated in place; correctors return a new frame.
        pattern: ``2 x 2`` colour indices of the tile at the visible origin,
            indices into ``color_desc`` (LibRaw ``raw_pattern``).
        color_desc: Channel letters, e.g. ``"RGBG"`` (LibRaw ``color_desc``).
        black_level: Per-channel black level in raw units, indexed like ``color_desc``.
        white_level: Saturation value in raw units.
    """

    data: NDArray[np.uint16]
    pattern: tuple[tuple[int, int], tuple[int, int]]
    color_desc: str
    black_level: tuple[int, ...]
    white_level: int

    def __post_init__(self) -> None:
        if self.data.ndim != 2:  # noqa: PLR2004 - a CFA is a 2-D array by definition
            msg = f"CFA data must be 2-D, got shape {self.data.shape}"
            raise ValueError(msg)
        if self.data.dtype != np.uint16:
            msg = f"CFA data must be uint16, got {self.data.dtype}"
            raise ValueError(msg)

    @property
    def height(self) -> int:
        """Number of sensor rows."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Number of sensor columns."""
        return int(self.data.shape[1])

    def color_index_map(self) -> NDArray[np.uint8]:
        """Return the ``(H, W)`` colour index of every photosite."""
        tile = np.asarray(self.pattern, dtype=np.uint8)
        reps = (-(-self.height // CFA_TILE_SIZE), -(-self.width // CFA_TILE_SIZE))
        return np.tile(tile, reps)[: self.height, : self.width]

    def with_data(self, data: NDArray[np.uint16]) -> CFAFrame:
        """Return a copy of this frame with ``data`` replaced (same layout and levels)."""
        return CFAFrame(
            data=data,
            pattern=self.pattern,
            color_desc=self.color_desc,
            black_level=self.black_level,
            white_level=self.white_level,
        )


@dataclass(frozen=True)
class ColumnDefect:
    """A defective column band.

    Attributes:
        index: First defective column (``x``).
        width: Number of adjacent defective columns starting at ``index``.
        start: First affected row (inclusive); ``None`` means row 0.
        end: Last affected row (exclusive); ``None`` means the frame height.
    """

    index: int
    width: int = 1
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class RowDefect:
    """A defective row band.

    Attributes:
        index: First defective row (``y``).
        width: Number of adjacent defective rows starting at ``index``.
        start: First affected column (inclusive); ``None`` means column 0.
        end: Last affected column (exclusive); ``None`` means the frame width.
    """

    index: int
    width: int = 1
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class PixelDefect:
    """A single defective photosite at column ``x``, row ``y``."""

    x: int
    y: int


Defect = ColumnDefect | RowDefect | PixelDefect


@dataclass(frozen=True)
class DefectMap:
    """Set of known sensor defects, in raw sensor coordinates.

    Attributes:
        defects: Defect entries. Overlaps are allowed; each photosite is
            corrected at most once.
    """

    defects: tuple[Defect, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[Defect]:
        return iter(self.defects)

    def __len__(self) -> int:
        return len(self.defects)

    def mask(self, height: int, width: int) -> NDArray[np.bool_]:
        """Rasterise the map to an ``(height, width)`` boolean mask.

        Raises:
            ValueError: If an entry lies completely outside the frame.
        """
        out = np.zeros((height, width), dtype=np.bool_)
        for defect in self.defects:
            ys, xs = _defect_slices(defect, height, width)
            if ys.start >= ys.stop or xs.start >= xs.stop:
                msg = f"defect {defect} lies outside the {width}x{height} sensor"
                raise ValueError(msg)
            out[ys, xs] = True
        return out


def _defect_slices(defect: Defect, height: int, width: int) -> tuple[slice, slice]:
    if isinstance(defect, PixelDefect):
        return slice(defect.y, min(defect.y + 1, height)), slice(defect.x, min(defect.x + 1, width))
    if isinstance(defect, ColumnDefect):
        rows = slice(max(defect.start or 0, 0), min(defect.end if defect.end is not None else height, height))
        return rows, slice(max(defect.index, 0), min(defect.index + defect.width, width))
    cols = slice(max(defect.start or 0, 0), min(defect.end if defect.end is not None else width, width))
    return slice(max(defect.index, 0), min(defect.index + defect.width, height)), cols


@dataclass(frozen=True)
class DefectCorrectionResult:
    """Output of a :class:`~raw_pipeline.core.protocols.DefectCorrector`.

    Attributes:
        frame: Corrected frame (a new object; the input frame is untouched).
        corrected_pixel_count: Number of photosites whose value was replaced.
        method: Registered name of the corrector.
    """

    frame: CFAFrame
    corrected_pixel_count: int
    method: str


@dataclass(frozen=True)
class DetectedDefect:
    """One defect found by a :class:`~raw_pipeline.core.protocols.DefectDetector`.

    Attributes:
        defect: The defect, ready to be written to ``defects.map``.
        score: Outlier strength in robust standard deviations (MAD-based).
        frames_flagged: Number of analysed frames where it was an outlier.
        frames_total: Number of analysed frames.
    """

    defect: Defect
    score: float
    frames_flagged: int
    frames_total: int


@dataclass(frozen=True)
class DefectResidual:
    """A configured defect the detector still finds after correction.

    Attributes:
        defect: The configured defect.
        score_before: Median absolute difference (DN) of its photosites from their nearest same-colour
            neighbours outside it, in the uncorrected frame.
        score_after: The same in the corrected frame.
    """

    defect: Defect
    score_before: float
    score_after: float

    @property
    def improved(self) -> bool:
        """Whether correction lowered the deviation (a weaker residual, not an unchanged, worse or new line)."""
        return self.score_after < self.score_before


@dataclass(frozen=True)
class DevelopedImage:
    """Demosaiced output of a :class:`~raw_pipeline.core.protocols.RawHandle`.

    Attributes:
        image: ``(H, W, 3)`` float32 scene-linear RGB in the working space
            (linear sRGB / Rec.709 primaries, D65), 1.0 = sensor white level
            after black subtraction and camera white balance. Not clipped.
        params: Development parameters applied (demosaic algorithm, WB multipliers, ...).
    """

    image: NDArray[np.float32]
    params: dict[str, str | int | float | bool | None]
