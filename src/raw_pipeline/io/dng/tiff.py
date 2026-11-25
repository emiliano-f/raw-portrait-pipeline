"""Read-only view of a DNG's TIFF structure: IFDs, tag entries and sample layout.

Parsing uses :mod:`tifffile`; the byte layout of strips/tiles is computed
here from the tag values, because the raw IFD is patched in place and
tifffile may merge contiguous strips (or read a multi-valued RowsPerStrip
as a single value).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import tifffile
from numpy.typing import NDArray

from raw_pipeline.core.dng import DngRawImage
from raw_pipeline.core.errors import DngFormatError

NEW_SUBFILE_TYPE: Final = 254
IMAGE_WIDTH: Final = 256
IMAGE_LENGTH: Final = 257
BITS_PER_SAMPLE: Final = 258
COMPRESSION: Final = 259
PHOTOMETRIC: Final = 262
STRIP_OFFSETS: Final = 273
SAMPLES_PER_PIXEL: Final = 277
ROWS_PER_STRIP: Final = 278
STRIP_BYTE_COUNTS: Final = 279
PLANAR_CONFIGURATION: Final = 284
TILE_WIDTH: Final = 322
TILE_LENGTH: Final = 323
TILE_OFFSETS: Final = 324
TILE_BYTE_COUNTS: Final = 325
SUB_IFDS: Final = 330
SAMPLE_FORMAT: Final = 339
CFA_REPEAT_PATTERN_DIM: Final = 33421
CFA_PATTERN: Final = 33422
DNG_VERSION: Final = 50706
CFA_PLANE_COLOR: Final = 50710
CFA_LAYOUT: Final = 50711
LINEARIZATION_TABLE: Final = 50712
BLACK_LEVEL_REPEAT_DIM: Final = 50713
BLACK_LEVEL: Final = 50714
WHITE_LEVEL: Final = 50717
DEFAULT_CROP_ORIGIN: Final = 50719
ACTIVE_AREA: Final = 50829
RAW_IMAGE_DIGEST: Final = 50858
NEW_RAW_IMAGE_DIGEST: Final = 51111

COMPRESSION_NONE: Final = 1
PHOTOMETRIC_CFA: Final = 32803
PHOTOMETRIC_LINEAR_RAW: Final = 34892
PHOTOMETRIC_PREVIEW: Final = frozenset({2, 6})
"""RGB and YCbCr: the photometric interpretations of DNG preview images."""
SUBFILE_REDUCED: Final = 1
"""NewSubfileType bit of reduced-resolution (preview / thumbnail) images."""
RAW_BITS: Final = 16
RAW_SAMPLE_BYTES: Final = RAW_BITS // 8
SAMPLE_FORMAT_UINT: Final = 1
CFA_TILE: Final = (2, 2)
DEFAULT_PLANE_COLORS: Final = (0, 1, 2)
PLANE_LETTERS: Final = "RGBCMYW"
"""TIFF/EP CFA colour codes 0..6."""
ENTRY_SIZE: Final = 12
"""Bytes of one classic-TIFF IFD entry (code, type, count, value/offset)."""

logging.getLogger("tifffile").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Entry:
    """One IFD entry.

    Attributes:
        code: Tag code.
        dtype: TIFF field type (1 BYTE, 3 SHORT, 4 LONG, 7 UNDEFINED, 13 IFD, ...).
        count: Number of values.
        entry_offset: File offset of the 12-byte entry.
        value_offset: File offset of the value (inside the entry when it fits in 4 bytes).
        value: Decoded value.
    """

    code: int
    dtype: int
    count: int
    entry_offset: int
    value_offset: int
    value: Any


@dataclass(frozen=True)
class Ifd:
    """An image file directory and its entries, keyed by tag code."""

    offset: int
    entries: Mapping[int, Entry]
    in_sub_ifds: bool

    def value(self, code: int, default: Any = None) -> Any:  # noqa: ANN401 - TIFF values are heterogeneous
        """Decoded value of tag ``code``, or ``default``."""
        entry = self.entries.get(code)
        return default if entry is None else entry.value

    def ints(self, code: int) -> tuple[int, ...]:
        """Tag ``code`` as a tuple of ints (empty when absent)."""
        value = self.value(code)
        if value is None:
            return ()
        if isinstance(value, bytes):
            return tuple(value)
        if isinstance(value, tuple | list):
            return tuple(int(v) for v in value)
        return (int(value),)

    def first(self, code: int, default: int) -> int:
        """First value of tag ``code`` as int, or ``default``."""
        values = self.ints(code)
        return values[0] if values else default

    @property
    def width(self) -> int:
        """Image columns."""
        return self.first(IMAGE_WIDTH, 0)

    @property
    def height(self) -> int:
        """Image rows."""
        return self.first(IMAGE_LENGTH, 0)

    @property
    def subfile_type(self) -> int:
        """NewSubfileType (0 = main image)."""
        return self.first(NEW_SUBFILE_TYPE, 0)

    @property
    def photometric(self) -> int:
        """PhotometricInterpretation."""
        return self.first(PHOTOMETRIC, 0)

    @property
    def compression(self) -> int:
        """Compression (1 = none)."""
        return self.first(COMPRESSION, COMPRESSION_NONE)

    @property
    def is_preview(self) -> bool:
        """A reduced-resolution RGB/YCbCr image (thumbnail or preview)."""
        return bool(self.subfile_type & SUBFILE_REDUCED) and self.photometric in PHOTOMETRIC_PREVIEW

    def grid(self, bytes_per_pixel: int) -> SampleGrid:
        """Byte layout of the image data (strips or tiles)."""
        if TILE_OFFSETS in self.entries:
            tile_h, tile_w = self.first(TILE_LENGTH, 0), self.first(TILE_WIDTH, 0)
            offsets, counts = self.ints(TILE_OFFSETS), self.ints(TILE_BYTE_COUNTS)
        else:
            tile_h, tile_w = min(self.first(ROWS_PER_STRIP, self.height), self.height), self.width
            offsets, counts = self.ints(STRIP_OFFSETS), self.ints(STRIP_BYTE_COUNTS)
        grid = SampleGrid(
            self.height, self.width, tile_h, tile_w, bytes_per_pixel, tiled=TILE_OFFSETS in self.entries,
            offsets=np.asarray(offsets, dtype=np.int64), byte_counts=np.asarray(counts, dtype=np.int64),
        )  # fmt: skip
        grid.validate()
        return grid


@dataclass(frozen=True)
class SampleGrid:
    """Strips (one tile column as wide as the image) or tiles of uncompressed samples.

    Attributes:
        height: Image rows.
        width: Image columns.
        tile_h: Rows per strip / tile.
        tile_w: Columns per tile (the image width for strips).
        bytes_per_pixel: Bytes of one pixel (all samples, chunky).
        tiled: Whether the data is tiled (edge tiles are padded).
        offsets: File offset of each segment, row-major.
        byte_counts: Stored bytes of each segment.
    """

    height: int
    width: int
    tile_h: int
    tile_w: int
    bytes_per_pixel: int
    tiled: bool
    offsets: NDArray[np.int64]
    byte_counts: NDArray[np.int64]

    @property
    def across(self) -> int:
        """Segments per row of segments."""
        return -(-self.width // self.tile_w)

    @property
    def down(self) -> int:
        """Rows of segments."""
        return -(-self.height // self.tile_h)

    def segment_rows(self, index: int) -> int:
        """Rows actually stored in segment ``index`` (edge tiles are padded to ``tile_h``)."""
        if self.tiled:
            return self.tile_h
        return min(self.tile_h, self.height - (index // self.across) * self.tile_h)

    def validate(self) -> None:
        """Raise :class:`DngFormatError` unless every segment holds its samples."""
        if self.tile_h <= 0 or self.tile_w <= 0 or self.width <= 0 or self.height <= 0:
            msg = "invalid image, strip or tile size"
            raise DngFormatError(msg)
        expected = self.across * self.down
        if self.offsets.size != expected or self.byte_counts.size != expected:
            found = f"{self.offsets.size} offsets and {self.byte_counts.size} counts"
            msg = f"expected {expected} strips/tiles, found {found}"
            raise DngFormatError(msg)
        needed = np.asarray([self.segment_rows(i) * self.tile_w * self.bytes_per_pixel for i in range(expected)])
        if np.any(self.byte_counts < needed):
            msg = "a strip/tile holds fewer bytes than its samples need (compressed or truncated data)"
            raise DngFormatError(msg)

    def byte_offsets(self, ys: NDArray[np.intp], xs: NDArray[np.intp]) -> NDArray[np.int64]:
        """File offset of the first byte of each pixel ``(ys[i], xs[i])``."""
        if np.any((ys < 0) | (ys >= self.height) | (xs < 0) | (xs >= self.width)):
            msg = "photosite outside the raw image"
            raise DngFormatError(msg)
        ty, tx = ys // self.tile_h, xs // self.tile_w
        local = (ys - ty * self.tile_h).astype(np.int64) * self.tile_w + (xs - tx * self.tile_w)
        offsets: NDArray[np.int64] = self.offsets[ty * self.across + tx] + local * self.bytes_per_pixel
        return offsets

    def read(self, path: Path, dtype: np.dtype[Any], samples: int = 1) -> NDArray[Any]:
        """Read the whole image as ``(H, W)`` or ``(H, W, samples)`` of ``dtype``."""
        out = np.empty((self.height, self.width, samples), dtype=dtype)
        with path.open("rb") as fh:
            for index in range(self.offsets.size):
                rows = self.segment_rows(index)
                fh.seek(int(self.offsets[index]))
                flat = np.fromfile(fh, dtype=dtype, count=rows * self.tile_w * samples)
                seg = flat.reshape(rows, self.tile_w, samples)
                y0, x0 = (index // self.across) * self.tile_h, (index % self.across) * self.tile_w
                h, w = min(rows, self.height - y0), min(self.tile_w, self.width - x0)
                out[y0 : y0 + h, x0 : x0 + w] = seg[:h, :w]
        return out[..., 0] if samples == 1 else out

    def write(self, fh: Any, image: NDArray[Any]) -> None:  # noqa: ANN401 - binary file object
        """Overwrite every stored pixel with ``image`` (same shape and dtype as stored)."""
        for index in range(self.offsets.size):
            rows = self.segment_rows(index)
            y0, x0 = (index // self.across) * self.tile_h, (index % self.across) * self.tile_w
            h, w = min(rows, self.height - y0), min(self.tile_w, self.width - x0)
            for r in range(h):
                fh.seek(int(self.offsets[index]) + r * self.tile_w * self.bytes_per_pixel)
                fh.write(np.ascontiguousarray(image[y0 + r, x0 : x0 + w]).tobytes())


@dataclass(frozen=True)
class DngLayout:
    """The IFDs of a DNG that ``fix-raw`` reads or edits.

    Attributes:
        byteorder: ``"<"`` or ``">"``.
        ifd0: The first IFD (DNG and EXIF metadata, often the thumbnail).
        raw: The main raw IFD (NewSubfileType 0, CFA photometric).
        sub_ifds: IFDs listed in IFD0's SubIFDs, in order.
    """

    byteorder: str
    ifd0: Ifd
    raw: Ifd
    sub_ifds: tuple[Ifd, ...]

    @property
    def previews(self) -> tuple[Ifd, ...]:
        """IFD0 (when it is a thumbnail) and the preview SubIFDs."""
        return tuple(ifd for ifd in (self.ifd0, *self.sub_ifds) if ifd.is_preview)

    @property
    def raw_dtype(self) -> np.dtype[np.uint16]:
        """Stored sample dtype in the file's byte order."""
        return np.dtype(f"{self.byteorder}u2")

    def raw_grid(self) -> SampleGrid:
        """Byte layout of the raw image data."""
        return self.raw.grid(RAW_SAMPLE_BYTES)


def _ifd(page: tifffile.TiffPage, *, in_sub_ifds: bool) -> Ifd:
    entries = {
        int(tag.code): Entry(
            int(tag.code), int(tag.dtype), int(tag.count), int(tag.offset), int(tag.valueoffset), tag.value
        )
        for tag in page.tags.values()
    }
    return Ifd(int(page.offset), entries, in_sub_ifds)


def _page_at(tif: tifffile.TiffFile, offset: int) -> tifffile.TiffPage:
    tif.filehandle.seek(offset)
    return tifffile.TiffPage(tif, index=0)


def read_layout(path: Path) -> DngLayout:
    """Parse the IFDs of the DNG at ``path`` and locate the main raw IFD.

    Raises:
        DngFormatError: Not a classic-TIFF DNG, or no NewSubfileType-0 CFA IFD.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.is_bigtiff:
                msg = f"{path.name}: BigTIFF DNGs are not supported"
                raise DngFormatError(msg)
            ifd0 = _ifd(tif.pages.first, in_sub_ifds=False)
            byteorder = tif.byteorder
            subs = tuple(_ifd(_page_at(tif, off), in_sub_ifds=True) for off in ifd0.ints(SUB_IFDS))
    except DngFormatError:
        raise
    except Exception as exc:  # tifffile raises many types on malformed input
        msg = f"{path.name}: not a readable TIFF/DNG: {exc}"
        raise DngFormatError(msg) from exc
    if DNG_VERSION not in ifd0.entries:
        msg = f"{path.name}: not a DNG (no DNGVersion tag)"
        raise DngFormatError(msg)
    main = [
        ifd
        for ifd in (ifd0, *subs)
        if ifd.subfile_type == 0 and ifd.photometric in {PHOTOMETRIC_CFA, PHOTOMETRIC_LINEAR_RAW}
    ]
    if not main:
        msg = f"{path.name}: no main raw IFD (NewSubfileType 0 with CFA photometric)"
        raise DngFormatError(msg)
    return DngLayout(byteorder, ifd0, main[0], subs)


def check_patchable(layout: DngLayout, name: str) -> None:
    """Raise :class:`DngFormatError` unless the raw IFD is uncompressed 16-bit CFA that can be patched in place."""
    raw = layout.raw
    problems = []
    if raw.photometric == PHOTOMETRIC_LINEAR_RAW:
        problems.append("linear (demosaiced) DNG; convert without the 'linear' option")
    elif raw.photometric != PHOTOMETRIC_CFA:
        problems.append(f"photometric {raw.photometric} is not CFA")
    if raw.compression != COMPRESSION_NONE:
        problems.append(f"raw data is compressed (Compression={raw.compression}); convert uncompressed")
    if set(raw.ints(BITS_PER_SAMPLE)) != {RAW_BITS}:
        problems.append(f"raw samples are {raw.ints(BITS_PER_SAMPLE)} bits, only {RAW_BITS} is supported")
    if raw.first(SAMPLES_PER_PIXEL, 1) != 1 or raw.first(PLANAR_CONFIGURATION, 1) != 1:
        problems.append("raw IFD must hold one sample per pixel")
    if raw.first(SAMPLE_FORMAT, SAMPLE_FORMAT_UINT) != SAMPLE_FORMAT_UINT:
        problems.append("raw samples are not unsigned integers")
    if raw.ints(CFA_REPEAT_PATTERN_DIM) not in {(), CFA_TILE} or raw.first(CFA_LAYOUT, 1) != 1:
        problems.append("only a rectangular 2x2 Bayer CFA is supported")
    if LINEARIZATION_TABLE in raw.entries:
        problems.append("raw data uses a LinearizationTable")
    if problems:
        msg = f"{name}: cannot patch in place: " + "; ".join(problems)
        raise DngFormatError(msg)
    layout.raw_grid()


def cfa_letters(raw: Ifd) -> str:
    """Colour letters of the 2x2 CFA tile at the stored origin, row-major (e.g. ``"RGGB"``)."""
    pattern = raw.ints(CFA_PATTERN)
    planes = raw.ints(CFA_PLANE_COLOR) or DEFAULT_PLANE_COLORS
    if len(pattern) != CFA_TILE[0] * CFA_TILE[1]:
        msg = "CFAPattern is missing or not 2x2"
        raise DngFormatError(msg)
    try:
        return "".join(PLANE_LETTERS[planes[code]] if code < len(planes) else "?" for code in pattern)
    except IndexError as exc:
        msg = f"unsupported CFA colours {pattern}"
        raise DngFormatError(msg) from exc


def read_raw(path: Path, layout: DngLayout | None = None) -> DngRawImage:
    """Return the stored CFA of the main raw IFD of the DNG at ``path``."""
    layout = layout or read_layout(path)
    check_patchable(layout, path.name)
    raw = layout.raw
    area = raw.ints(ACTIVE_AREA)
    crop = raw.ints(DEFAULT_CROP_ORIGIN)
    data = layout.raw_grid().read(path, layout.raw_dtype).astype(np.uint16)
    return DngRawImage(
        black_level=_black_levels(raw),
        white_level=raw.first(WHITE_LEVEL, (1 << RAW_BITS) - 1),
        data=data,
        cfa_pattern=cfa_letters(raw),
        active_area=(area[0], area[1], area[2], area[3]) if len(area) == 4 else None,  # noqa: PLR2004 - top, left, bottom, right
        default_crop_origin=(crop[0], crop[1]) if len(crop) == 2 else None,  # noqa: PLR2004 - x, y
    )


RATIONAL_TYPES: Final = frozenset({5, 10})


def _black_levels(raw: Ifd) -> tuple[int, ...]:
    """BlackLevel as ints per 2x2 tile position (rationals rounded); ``(0,)`` when absent."""
    entry = raw.entries.get(BLACK_LEVEL)
    if entry is None:
        return (0,)
    values = raw.ints(BLACK_LEVEL)
    if entry.dtype in RATIONAL_TYPES:
        values = tuple(round(n / d) if d else 0 for n, d in zip(values[::2], values[1::2], strict=True))
    if len(set(values)) == 1:
        return (values[0],)
    if raw.ints(BLACK_LEVEL_REPEAT_DIM) != CFA_TILE or len(values) != CFA_TILE[0] * CFA_TILE[1]:
        msg = "BlackLevel must be a single value or a 2x2 repeat"
        raise DngFormatError(msg)
    return values
