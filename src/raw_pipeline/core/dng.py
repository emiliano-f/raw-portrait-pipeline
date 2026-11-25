"""Value objects of ``fix-raw``: base DNGs, raw placement, patches and write results (ADR 0003).

Coordinates follow ADR 0002: defect maps are in the LibRaw *visible* CFA of
the reference CR2. :class:`CFAOffset` places that visible area inside the
*stored* raw image of a DNG (row ``y``, column ``x`` of the stored strip/tile
data, before ActiveArea, DefaultCrop or orientation are applied).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class PreviewMode(StrEnum):
    """What happens to the embedded previews, which still show the defect."""

    KEEP = "keep"
    REGENERATE = "regenerate"
    REMOVE = "remove"


class DigestHandling(StrEnum):
    """What happened to ``RawImageDigest`` / ``NewRawImageDigest``."""

    ABSENT = "absent"
    UNCHANGED = "unchanged"
    RECOMPUTED = "recomputed"
    REMOVED = "removed"


class GpsHandling(StrEnum):
    """What happened to GPS metadata (the EXIF GPS IFD and XMP ``exif:GPS*`` fields)."""

    ABSENT = "absent"
    KEPT = "kept"
    STRIPPED = "stripped"


@dataclass(frozen=True)
class BaseDng:
    """The unmodified DNG a source produced for one input.

    Attributes:
        path: The base DNG. For converters it lives in a temporary folder owned
            by the caller; for ``existing_dng`` it is the (read-only) input.
        reference: RAW file whose LibRaw visible area defines the defect-map
            coordinates (the CR2).
        source: Registered name of the :class:`~raw_pipeline.core.protocols.DngSource`.
    """

    path: Path
    reference: Path
    source: str


@dataclass(frozen=True)
class CFAOffset:
    """Position of the reference's visible CFA origin in the DNG stored raw image.

    Attributes:
        y: Row of the stored raw image holding visible row 0.
        x: Column of the stored raw image holding visible column 0.
    """

    y: int
    x: int


@dataclass(frozen=True)
class DngRawImage:
    """The stored CFA of a DNG's main raw IFD (NewSubfileType 0, CFA photometric).

    Attributes:
        data: ``(H, W)`` uint16 stored samples (whole stored image, masked borders included).
        cfa_pattern: Colour letters of the 2x2 tile at stored origin, row-major (e.g. ``"RGGB"``).
        active_area: ``(top, left, bottom, right)`` of the ActiveArea tag, or ``None``.
        default_crop_origin: ``(x, y)`` of DefaultCropOrigin relative to the active area, or ``None``.
        black_level: BlackLevel per position of the 2x2 tile at the active-area origin, row-major
            (one value when the tag holds a single level).
        white_level: First WhiteLevel value.
    """

    data: NDArray[np.uint16]
    cfa_pattern: str
    active_area: tuple[int, int, int, int] | None = None
    default_crop_origin: tuple[int, int] | None = None
    black_level: tuple[int, ...] = (0,)
    white_level: int = 65535


@dataclass(frozen=True)
class RawPatch:
    """Photosites to overwrite, in stored raw coordinates.

    Attributes:
        ys: Rows (``intp``).
        xs: Columns (``intp``), same length as ``ys``.
        values: New uint16 samples, same length as ``ys``.
    """

    ys: NDArray[np.intp]
    xs: NDArray[np.intp]
    values: NDArray[np.uint16]

    def __post_init__(self) -> None:
        if not (self.ys.shape == self.xs.shape == self.values.shape) or self.ys.ndim != 1:
            msg = "RawPatch ys, xs and values must be 1-D arrays of the same length"
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.ys.size)


@dataclass(frozen=True)
class DngWriteResult:
    """What :class:`~raw_pipeline.core.protocols.DngRawWriter` changed in the output file.

    Attributes:
        path: The DNG written.
        photosites_written: Samples overwritten in the raw image data.
        digest: Handling of the raw image digests.
        preview: Preview handling actually applied.
        gps: Handling of GPS metadata.
    """

    path: Path
    photosites_written: int
    digest: DigestHandling
    preview: PreviewMode
    gps: GpsHandling = GpsHandling.ABSENT
