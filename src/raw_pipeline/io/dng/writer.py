"""In-place raw patching of uncompressed CFA DNGs (ADR 0003)."""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.dng import DigestHandling, DngRawImage, DngWriteResult, PreviewMode, RawPatch
from raw_pipeline.core.errors import DngFormatError, OutputExistsError
from raw_pipeline.io.dng import preview as previews
from raw_pipeline.io.dng.digest import legacy_raw_digest, new_raw_digest
from raw_pipeline.io.dng.tiff import (
    BITS_PER_SAMPLE,
    COMPRESSION_NONE,
    ENTRY_SIZE,
    NEW_RAW_IMAGE_DIGEST,
    PLANAR_CONFIGURATION,
    RAW_IMAGE_DIGEST,
    SAMPLES_PER_PIXEL,
    STRIP_BYTE_COUNTS,
    STRIP_OFFSETS,
    SUB_IFDS,
    TILE_OFFSETS,
    DngLayout,
    Entry,
    Ifd,
    check_patchable,
    read_layout,
    read_raw,
)

logger = logging.getLogger(__name__)

COMPRESSION_JPEG: Final = 7
PHOTOMETRIC_RGB: Final = 2
RGB_SAMPLES: Final = 3
RGB_BITS: Final = (8, 8, 8)
INLINE_BYTES: Final = 4
OUTPUT_MODE: Final = 0o666
"""Permissions of the written DNG before the umask (mkstemp would leave 0600)."""
TYPE_SIZES: Final = {1: 1, 3: 2, 4: 4, 7: 1, 13: 4}
"""Bytes per value of the TIFF field types edited here."""
TYPE_FORMATS: Final = {1: "B", 3: "H", 4: "I", 7: "B", 13: "I"}
LONG_TYPES: Final = frozenset({4, 13})
DIGESTS: Final = ((RAW_IMAGE_DIGEST, legacy_raw_digest), (NEW_RAW_IMAGE_DIGEST, new_raw_digest))

Renderer = Callable[[Path], NDArray[np.uint8]]
"""Renders a quick unrotated sRGB 8-bit development of a DNG (for regenerated previews)."""


class _Editor:
    """Low-level edits of classic-TIFF IFD entries in an open file."""

    def __init__(self, fh: BinaryIO, byteorder: str) -> None:
        self._fh = fh
        self._bo = byteorder

    def write_at(self, offset: int, data: bytes) -> None:
        self._fh.seek(offset)
        self._fh.write(data)

    def zero(self, offset: int, count: int) -> None:
        self.write_at(offset, bytes(count))

    def append(self, data: bytes) -> int:
        """Write ``data`` at the end of the file (word aligned); return its offset."""
        end = self._fh.seek(0, os.SEEK_END)
        if end % 2:
            self._fh.write(b"\0")
            end += 1
        self._fh.write(data)
        return end

    def set_values(self, entry: Entry, values: list[int]) -> None:
        """Rewrite ``entry`` with ``values`` (count may shrink, never grow beyond the old storage)."""
        size = TYPE_SIZES[entry.dtype]
        if len(values) > entry.count and len(values) * size > INLINE_BYTES:
            msg = f"tag {entry.code}: cannot grow from {entry.count} values in place"
            raise DngFormatError(msg)
        packed = struct.pack(f"{self._bo}{len(values)}{TYPE_FORMATS[entry.dtype]}", *values)
        self.write_at(entry.entry_offset + 4, struct.pack(f"{self._bo}I", len(values)))
        if len(packed) <= INLINE_BYTES:
            self.write_at(entry.entry_offset + 8, packed.ljust(INLINE_BYTES, b"\0"))
        else:
            self.write_at(entry.value_offset, packed)

    def remove_entries(self, ifd: Ifd, codes: set[int]) -> None:
        """Drop the entries ``codes`` from ``ifd``, shifting the remaining ones (sorted order kept)."""
        self._fh.seek(ifd.offset)
        (count,) = struct.unpack(f"{self._bo}H", self._fh.read(2))
        table = self._fh.read(count * ENTRY_SIZE + 4)
        entries = [table[i * ENTRY_SIZE : (i + 1) * ENTRY_SIZE] for i in range(count)]
        kept = [e for e in entries if struct.unpack(f"{self._bo}H", e[:2])[0] not in codes]
        body = struct.pack(f"{self._bo}H", len(kept)) + b"".join(kept) + table[count * ENTRY_SIZE :]
        self.write_at(ifd.offset, body.ljust(2 + len(table), b"\0"))


def _patch_raw(editor: _Editor, layout: DngLayout, patch: RawPatch) -> int:
    if not len(patch):
        return 0
    offsets = layout.raw_grid().byte_offsets(patch.ys, patch.xs)
    order = np.argsort(offsets, kind="stable")
    offsets, values = offsets[order], patch.values[order].astype(layout.raw_dtype)
    breaks = np.flatnonzero(np.diff(offsets) != layout.raw_dtype.itemsize) + 1
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, offsets.size], strict=True):
        editor.write_at(int(offsets[start]), values[start:stop].tobytes())
    return len(patch)


def _segments(ifd: Ifd) -> list[tuple[int, int]]:
    codes = (TILE_OFFSETS, TILE_OFFSETS + 1) if TILE_OFFSETS in ifd.entries else (STRIP_OFFSETS, STRIP_BYTE_COUNTS)
    return list(zip(ifd.ints(codes[0]), ifd.ints(codes[1]), strict=False))


def _is_plain_rgb(ifd: Ifd) -> bool:
    return (
        ifd.compression == COMPRESSION_NONE
        and ifd.photometric == PHOTOMETRIC_RGB
        and ifd.ints(BITS_PER_SAMPLE) == RGB_BITS
        and ifd.first(SAMPLES_PER_PIXEL, 1) == RGB_SAMPLES
        and ifd.first(PLANAR_CONFIGURATION, 1) == 1
    )


def _is_single_jpeg(ifd: Ifd) -> bool:
    return (
        ifd.compression == COMPRESSION_JPEG
        and TILE_OFFSETS not in ifd.entries
        and len(ifd.ints(STRIP_OFFSETS)) == 1
        and ifd.entries[STRIP_OFFSETS].dtype in LONG_TYPES
        and ifd.entries[STRIP_BYTE_COUNTS].dtype in LONG_TYPES
    )


def _replace_image(editor: _Editor, fh: BinaryIO, ifd: Ifd, rgb: NDArray[np.uint8]) -> None:
    """Replace the pixels of preview ``ifd`` with ``rgb`` (already at its size)."""
    if _is_plain_rgb(ifd):
        ifd.grid(RGB_SAMPLES).write(fh, rgb)
        return
    data = previews.encode_jpeg(rgb)
    ((old_offset, old_count),) = _segments(ifd)
    editor.zero(old_offset, old_count)
    offset = editor.append(data)
    editor.set_values(ifd.entries[STRIP_OFFSETS], [offset])
    editor.set_values(ifd.entries[STRIP_BYTE_COUNTS], [len(data)])


def _replaceable(ifd: Ifd) -> bool:
    return _is_plain_rgb(ifd) or _is_single_jpeg(ifd)


@dataclass(frozen=True)
class TiffDngRawWriter:
    """:class:`~raw_pipeline.core.protocols.DngRawWriter` over :mod:`tifffile` parsing and in-place byte edits.

    Only the raw photosites of the patch, the digest tags and (by
    ``preview``) the preview images change; every other byte of the base DNG
    is copied unchanged and no IFD is rebuilt.

    Args:
        renderer: Develops the patched DNG for regenerated previews
            (injected by tests; LibRaw by default).
    """

    renderer: Renderer = previews.render_preview

    def read_raw(self, path: Path) -> DngRawImage:
        """See :meth:`~raw_pipeline.core.protocols.DngRawWriter.read_raw`."""
        return read_raw(path)

    def write(
        self,
        base: Path,
        destination: Path,
        patch: RawPatch,
        *,
        preview: PreviewMode,
        overwrite: bool,
        check: Callable[[Path], None] | None = None,
    ) -> DngWriteResult:
        """See :meth:`~raw_pipeline.core.protocols.DngRawWriter.write`."""
        layout = read_layout(base)
        check_patchable(layout, base.name)
        if destination.exists() and not overwrite:
            msg = f"{destination} already exists; pass --overwrite to replace it"
            raise OutputExistsError(msg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        tmp = Path(name)
        try:
            tmp.chmod(OUTPUT_MODE & ~_umask())
            with os.fdopen(fd, "r+b") as fh, base.open("rb") as src:
                shutil.copyfileobj(src, fh)
                editor = _Editor(fh, layout.byteorder)
                written = _patch_raw(editor, layout, patch)
                digest = self._digests(editor, layout, base, patch)
                fh.flush()
                applied = self._previews(editor, fh, layout, tmp, preview) if len(patch) else PreviewMode.KEEP
                fh.flush()
                os.fsync(fh.fileno())
            if check is not None:
                check(tmp)
            _publish(tmp, destination, overwrite=overwrite)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return DngWriteResult(destination, written, digest, applied)

    @staticmethod
    def _digests(editor: _Editor, layout: DngLayout, base: Path, patch: RawPatch) -> DigestHandling:
        present = [(code, fn) for code, fn in DIGESTS if code in layout.ifd0.entries]
        if not present:
            return DigestHandling.ABSENT
        original = layout.raw_grid().read(base, layout.raw_dtype).astype(np.uint16)
        corrected = original.copy()
        corrected[patch.ys, patch.xs] = patch.values
        stale: set[int] = set()
        for code, digest in present:
            entry = layout.ifd0.entries[code]
            if bytes(entry.value) != digest(original):
                logger.warning("%s: stored tag %d does not match the DNG-SDK digest; removing it", base.name, code)
                stale.add(code)
            elif len(patch):
                editor.write_at(entry.value_offset, digest(corrected))
        if stale:
            editor.remove_entries(layout.ifd0, stale)
            return DigestHandling.REMOVED
        return DigestHandling.RECOMPUTED if len(patch) else DigestHandling.UNCHANGED

    def _previews(self, editor: _Editor, fh: BinaryIO, layout: DngLayout, path: Path, mode: PreviewMode) -> PreviewMode:
        targets = layout.previews
        if mode is PreviewMode.KEEP or not targets:
            return mode
        if mode is PreviewMode.REGENERATE:
            if not all(_replaceable(ifd) for ifd in targets):
                logger.warning("%s: a preview format cannot be regenerated in place; keeping previews", path.name)
                return PreviewMode.KEEP
            rgb = self.renderer(path)
            for ifd in targets:
                _replace_image(editor, fh, ifd, previews.resized(rgb, ifd.width, ifd.height))
            return mode
        return _remove_previews(editor, fh, layout, path.name)


def _remove_previews(editor: _Editor, fh: BinaryIO, layout: DngLayout, name: str) -> PreviewMode:
    dropped = [ifd for ifd in layout.previews if ifd.in_sub_ifds]
    if layout.ifd0.is_preview:
        if not _replaceable(layout.ifd0):
            logger.warning("%s: the IFD0 thumbnail cannot be blanked in place; keeping previews", name)
            return PreviewMode.KEEP
        _replace_image(editor, fh, layout.ifd0, previews.grey(layout.ifd0.width, layout.ifd0.height))
    for ifd in dropped:
        for offset, count in _segments(ifd):
            editor.zero(offset, count)
    if dropped:
        kept = [ifd.offset for ifd in layout.sub_ifds if ifd not in dropped]
        if kept:
            editor.set_values(layout.ifd0.entries[SUB_IFDS], kept)
        else:
            editor.remove_entries(layout.ifd0, {SUB_IFDS})
    return PreviewMode.REMOVE


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def _publish(tmp: Path, destination: Path, *, overwrite: bool) -> None:
    """Move the finished file into place; never replace an existing file unless ``overwrite``."""
    if overwrite:
        tmp.replace(destination)
        return
    try:
        os.link(tmp, destination)
    except FileExistsError as exc:
        msg = f"{destination} already exists; pass --overwrite to replace it"
        raise OutputExistsError(msg) from exc
    except OSError:  # filesystem without hard links
        if destination.exists():
            msg = f"{destination} already exists; pass --overwrite to replace it"
            raise OutputExistsError(msg) from None
        tmp.replace(destination)
        return
    tmp.unlink()
