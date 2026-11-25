"""Place the reference CR2's visible CFA inside a DNG's stored raw image (ADR 0003).

Defect maps use LibRaw visible-area coordinates of the CR2. A converter may
store more or fewer masked rows/columns, so the offset is never assumed:
candidates derived from ActiveArea / DefaultCrop are tried first, then a
small Bayer-phase-preserving search, and an offset is accepted only when
*every* overlapping photosite is identical and the CFA colours agree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.dng import CFAOffset, DngRawImage
from raw_pipeline.core.errors import CoordinateMappingError
from raw_pipeline.core.raw import CFA_TILE_SIZE, CFAFrame

SEARCH_RADIUS_PX: Final = 64
"""Largest shift (photosites, each axis) searched around every tag-derived candidate."""
SAMPLE_PHOTOSITES: Final = 4096
"""Photosites compared first to discard candidates cheaply before the full comparison."""
GREEN: Final = "G"


@dataclass(frozen=True)
class Placement:
    """Result of :func:`find_offset`.

    Attributes:
        offset: Visible origin in stored raw coordinates.
        method: ``active_area``, ``default_crop`` or ``origin`` when a tag-derived
            candidate matched; ``search`` when a nearby shift did.
        candidates_checked: Offsets compared photosite by photosite.
    """

    offset: CFAOffset
    method: str
    candidates_checked: int


def reference_letters(frame: CFAFrame) -> str:
    """CFA colour letters of ``frame``'s 2x2 tile, row-major (``G2`` reads as ``G``)."""
    letters = "".join(frame.color_desc[c] for row in frame.pattern for c in row)
    return letters.upper()


def letters_at(pattern: str, y: int, x: int) -> str:
    """Letters of the 2x2 tile whose top-left is stored position ``(y, x)`` of a CFA ``pattern``."""
    return "".join(
        pattern[((y + r) % CFA_TILE_SIZE) * CFA_TILE_SIZE + (x + c) % CFA_TILE_SIZE] for r in range(2) for c in range(2)
    )


def _tag_candidates(raw: DngRawImage) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    if raw.active_area is not None:
        top, left = raw.active_area[0], raw.active_area[1]
        out.append(("active_area", top, left))
        if raw.default_crop_origin is not None:
            out.append(("default_crop", top + raw.default_crop_origin[1], left + raw.default_crop_origin[0]))
    out.append(("origin", 0, 0))
    return out


def _search(bases: list[tuple[str, int, int]]) -> Iterator[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    shifts = sorted(
        (
            (dy, dx)
            for dy in range(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1)
            for dx in range(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1)
        ),
        key=lambda s: (abs(s[0]) + abs(s[1]), s),
    )
    for _, y, x in bases:
        for dy, dx in shifts:
            if (y + dy, x + dx) not in seen:
                seen.add((y + dy, x + dx))
                yield y + dy, x + dx


def find_offset(reference: CFAFrame, raw: DngRawImage) -> Placement:
    """Find where ``reference`` (CR2 visible CFA) lies in ``raw`` (DNG stored CFA).

    Raises:
        CoordinateMappingError: No offset, or more than one in the search,
            gives identical photosites with matching CFA colours.
    """
    ref = reference.data
    height, width = ref.shape
    want = reference_letters(reference)
    rng = np.random.default_rng(0)
    sample = (rng.integers(0, height, SAMPLE_PHOTOSITES), rng.integers(0, width, SAMPLE_PHOTOSITES))
    ref_sample = ref[sample]
    checked = 0

    def matches(y: int, x: int) -> bool:
        nonlocal checked
        if y < 0 or x < 0 or y + height > raw.data.shape[0] or x + width > raw.data.shape[1]:
            return False
        if letters_at(raw.cfa_pattern, y, x) != want:
            return False
        if not np.array_equal(raw.data[sample[0] + y, sample[1] + x], ref_sample):
            return False
        checked += 1
        return bool(np.array_equal(raw.data[y : y + height, x : x + width], ref))

    bases = _tag_candidates(raw)
    for method, y, x in bases:
        if matches(y, x):
            return Placement(CFAOffset(y, x), method, checked)
    found = [(y, x) for y, x in _search(bases) if matches(y, x)]
    if len(found) == 1:
        return Placement(CFAOffset(*found[0]), "search", checked)
    if found:
        msg = f"ambiguous placement: {len(found)} offsets match exactly (e.g. {found[:3]}); refusing to guess"
        raise CoordinateMappingError(msg)
    msg = (
        f"no offset within {SEARCH_RADIUS_PX} px of the ActiveArea/DefaultCrop/origin candidates gives "
        f"identical photosites with CFA {want}; the converter changed the raw values or the layout"
    )
    raise CoordinateMappingError(msg)


def stored_coordinates(mask: NDArray[np.bool_], offset: CFAOffset) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Stored-raw ``(ys, xs)`` of the photosites set in a visible-area ``mask``."""
    ys, xs = np.nonzero(mask)
    return (ys + offset.y).astype(np.intp), (xs + offset.x).astype(np.intp)


def visible_frame(reference: CFAFrame, raw: DngRawImage, offset: CFAOffset) -> CFAFrame:
    """The DNG photosites under the reference's visible area, with the DNG's black and white levels.

    The CFA pattern and colour description come from ``reference`` (they were
    checked to match the DNG at ``offset``); black levels are mapped from
    DNG tile positions to LibRaw channel indices.
    """
    height, width = reference.data.shape
    data = np.array(raw.data[offset.y : offset.y + height, offset.x : offset.x + width], dtype=np.uint16, copy=True)
    black = list(reference.black_level)
    if len(raw.black_level) == 1:
        black = [raw.black_level[0]] * len(black)
    else:
        top, left = raw.active_area[:2] if raw.active_area is not None else (0, 0)
        for r in range(CFA_TILE_SIZE):
            for c in range(CFA_TILE_SIZE):
                pos = ((offset.y + r - top) % CFA_TILE_SIZE) * CFA_TILE_SIZE + (offset.x + c - left) % CFA_TILE_SIZE
                black[reference.pattern[r][c]] = raw.black_level[pos]
    return CFAFrame(
        data=data,
        pattern=reference.pattern,
        color_desc=reference.color_desc,
        black_level=tuple(black),
        white_level=raw.white_level,
    )
