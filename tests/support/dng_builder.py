"""Minimal synthetic DNG writer for tests: classic TIFF, IFD0 thumbnail + SubIFDs (raw, JPEG preview)."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from raw_pipeline.io.dng.digest import legacy_raw_digest, new_raw_digest

BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED = 1, 2, 3, 4, 5, 7
FORMATS = {BYTE: "B", ASCII: "s", SHORT: "H", LONG: "I", RATIONAL: "I", UNDEFINED: "B"}
SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1}


def raw_frame(height: int = 48, width: int = 64, seed: int = 0) -> NDArray[np.uint16]:
    """Noisy Bayer data; values are unique enough that only the true offset matches."""
    rng = np.random.default_rng(seed)
    return (1000 + rng.integers(0, 3000, (height, width))).astype(np.uint16)


@dataclass
class DngSpec:
    """What :func:`write_dng` writes. Defaults mimic a DNGLab EOS 800D file."""

    raw: NDArray[np.uint16] = field(default_factory=raw_frame)
    byteorder: str = "<"
    rows_per_strip: int = 16
    tile: tuple[int, int] | None = None
    compression: int = 1
    photometric: int = 32803
    bits: int = 16
    cfa: bytes = b"\x00\x01\x01\x02"
    active_area: tuple[int, int, int, int] | None = (2, 4, 48, 64)
    crop_origin: tuple[int, int] | None = (2, 2)
    black_level: tuple[int, ...] = (500,)
    white_level: int = 12000
    thumbnail: tuple[int, int] | None = (6, 8)
    jpeg_preview: tuple[int, int] | None = (12, 16)
    digests: str | None = None
    """``None``, ``"valid"`` (SDK digests of ``raw``) or ``"stale"`` (wrong values)."""
    linearization: bool = False
    gps: bool = False
    """Add an EXIF GPS IFD (version, latitude ref and an out-of-line latitude) to IFD0."""
    xmp: bytes | None = None
    original_raw: bytes | None = None
    """Embedded original raw file (``OriginalRawFileData``)."""


class _Buffer:
    def __init__(self, bo: str) -> None:
        self.bo = bo
        self.data = bytearray(b"II*\x00" if bo == "<" else b"MM\x00*") + bytes(4)

    def add(self, blob: bytes) -> int:
        if len(self.data) % 2:
            self.data += b"\0"
        offset = len(self.data)
        self.data += blob
        return offset

    def ifd(self, entries: list[tuple[int, int, list[int] | bytes]]) -> int:
        packed = []
        for code, dtype, values in sorted(entries, key=lambda e: e[0]):
            if isinstance(values, bytes):
                body, count = values, len(values)
            else:
                count = len(values) // 2 if dtype == RATIONAL else len(values)
                body = struct.pack(f"{self.bo}{len(values)}{FORMATS[dtype]}", *values)
            value = body.ljust(4, b"\0") if len(body) <= 4 else struct.pack(f"{self.bo}I", self.add(body))
            packed.append(struct.pack(f"{self.bo}HHI", code, dtype, count) + value)
        return self.add(struct.pack(f"{self.bo}H", len(packed)) + b"".join(packed) + bytes(4))


def _segments(
    buf: _Buffer, image: NDArray[np.generic], spec_tile: tuple[int, int] | None, rows: int
) -> tuple[list[int], list[int]]:
    offsets, counts = [], []
    height, width = image.shape[:2]
    if spec_tile is None:
        for y in range(0, height, rows):
            blob = np.ascontiguousarray(image[y : y + rows]).tobytes()
            offsets.append(buf.add(blob))
            counts.append(len(blob))
        return offsets, counts
    th, tw = spec_tile
    for y in range(0, height, th):
        for x in range(0, width, tw):
            tile = np.zeros((th, tw, *image.shape[2:]), dtype=image.dtype)
            part = image[y : y + th, x : x + tw]
            tile[: part.shape[0], : part.shape[1]] = part
            blob = tile.tobytes()
            offsets.append(buf.add(blob))
            counts.append(len(blob))
    return offsets, counts


def jpeg_bytes(height: int, width: int, value: int = 200) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, np.uint8)).save(out, format="JPEG")
    return out.getvalue()


def write_dng(path: Path, spec: DngSpec | None = None) -> Path:
    """Write a synthetic DNG described by ``spec`` to ``path``."""
    s = spec or DngSpec()
    buf = _Buffer(s.byteorder)
    height, width = s.raw.shape
    stored = s.raw.astype(np.dtype(f"{s.byteorder}u2")) if s.bits == 16 else s.raw.astype(np.uint8)
    offsets, counts = _segments(buf, stored, s.tile, s.rows_per_strip)
    raw_entries: list[tuple[int, int, list[int] | bytes]] = [
        (254, LONG, [0]), (256, LONG, [width]), (257, LONG, [height]), (258, SHORT, [s.bits]),
        (259, SHORT, [s.compression]), (262, SHORT, [s.photometric]), (277, SHORT, [1]),
        (33421, SHORT, [2, 2]), (33422, BYTE, s.cfa), (50717, LONG, [s.white_level]),
        (50714, SHORT, list(s.black_level)),
    ]  # fmt: skip
    if len(s.black_level) > 1:
        raw_entries.append((50713, SHORT, [2, 2]))
    if s.tile is None:
        raw_entries += [(273, LONG, offsets), (278, LONG, [s.rows_per_strip]), (279, LONG, counts)]
    else:
        raw_entries += [(322, LONG, [s.tile[1]]), (323, LONG, [s.tile[0]]), (324, LONG, offsets), (325, LONG, counts)]
    if s.active_area is not None:
        raw_entries.append((50829, LONG, list(s.active_area)))
    if s.crop_origin is not None:
        raw_entries.append((50719, LONG, list(s.crop_origin)))
    if s.linearization:
        raw_entries.append((50712, SHORT, list(range(16))))
    subs = [buf.ifd(raw_entries)]
    if s.jpeg_preview is not None:
        blob = jpeg_bytes(*s.jpeg_preview)
        h, w = s.jpeg_preview
        subs.append(buf.ifd([
            (254, LONG, [1]), (256, LONG, [w]), (257, LONG, [h]), (258, SHORT, [8, 8, 8]), (259, SHORT, [7]),
            (262, SHORT, [6]), (273, LONG, [buf.add(blob)]), (277, SHORT, [3]), (278, LONG, [h]),
            (279, LONG, [len(blob)]),
        ]))  # fmt: skip
    ifd0: list[tuple[int, int, list[int] | bytes]] = [
        (330, LONG, subs),
        (50706, BYTE, b"\x01\x04\x00\x00"),
        (271, ASCII, b"Canon\0"),
        (272, ASCII, b"EOS 800D\0"),
    ]
    if s.thumbnail is not None:
        h, w = s.thumbnail
        thumb = np.full((h, w, 3), 100, np.uint8)
        ifd0 += [
            (254, LONG, [1]), (256, LONG, [w]), (257, LONG, [h]), (258, SHORT, [8, 8, 8]), (259, SHORT, [1]),
            (262, SHORT, [2]), (273, LONG, [buf.add(thumb.tobytes())]), (277, SHORT, [3]), (278, LONG, [h]),
            (279, LONG, [thumb.nbytes]),
        ]  # fmt: skip
    else:
        ifd0 += [(254, LONG, [0]), (256, LONG, [1]), (257, LONG, [1]), (262, SHORT, [2])]
    if s.digests is not None:
        legacy, new = legacy_raw_digest(s.raw), new_raw_digest(s.raw)
        if s.digests == "stale":
            legacy, new = bytes(16), bytes(range(16))
        ifd0 += [(50858, BYTE, legacy), (51111, BYTE, new)]
    ifd0 += _metadata(buf, s)
    first = buf.ifd(ifd0)
    buf.data[4:8] = struct.pack(f"{s.byteorder}I", first)
    path.write_bytes(bytes(buf.data))
    return path


def _metadata(buf: _Buffer, s: DngSpec) -> list[tuple[int, int, list[int] | bytes]]:
    """Optional GPS IFD pointer and XMP packet entries of IFD0."""
    entries: list[tuple[int, int, list[int] | bytes]] = []
    if s.gps:
        gps_ifd = buf.ifd([(0, BYTE, b"\x02\x03\x00\x00"), (1, ASCII, b"S\0"), (2, RATIONAL, [34, 1, 36, 1, 7, 1])])
        entries.append((34853, LONG, [gps_ifd]))
    if s.xmp is not None:
        entries.append((700, BYTE, s.xmp))
    if s.original_raw is not None:
        entries.append((50817, UNDEFINED, s.original_raw))
    return entries
