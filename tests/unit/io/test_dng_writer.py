from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import rawpy

from raw_pipeline.core.dng import DigestHandling, PreviewMode, RawPatch
from raw_pipeline.core.errors import DngFormatError, OutputExistsError
from raw_pipeline.io.dng.digest import legacy_raw_digest, new_raw_digest
from raw_pipeline.io.dng.tiff import NEW_RAW_IMAGE_DIGEST, RAW_IMAGE_DIGEST, SUB_IFDS, read_layout, read_raw
from raw_pipeline.io.dng.writer import TiffDngRawWriter
from tests.support.dng_builder import DngSpec, raw_frame, write_dng


def _render(_: Path) -> np.ndarray:
    return np.full((24, 32, 3), 77, np.uint8)


WRITER = TiffDngRawWriter(renderer=_render)


def _column_patch(x: int = 10, value: int = 9) -> RawPatch:
    ys = np.arange(0, 48, dtype=np.intp)
    return RawPatch(ys, np.full_like(ys, x), np.full(ys.size, value, np.uint16))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "spec",
    [DngSpec(), DngSpec(byteorder=">"), DngSpec(tile=(16, 16)), DngSpec(tile=(32, 48)), DngSpec(rows_per_strip=7)],
    ids=["strips", "big-endian", "tiles", "padded-tiles", "uneven-strips"],
)
def test_patch_changes_only_patched_photosites(tmp_path: Path, spec: DngSpec) -> None:
    base = write_dng(tmp_path / "base.dng", spec)
    before = _sha(base)
    result = WRITER.write(base, tmp_path / "out.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    original, written = read_raw(base).data, read_raw(result.path).data
    mask = np.zeros(original.shape, bool)
    mask[:, 10] = True
    np.testing.assert_array_equal(written[~mask], original[~mask])
    assert (written[mask] == 9).all()
    assert result.photosites_written == 48
    assert _sha(base) == before


def test_only_raw_bytes_differ_with_keep(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    out = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False).path
    a, b = np.frombuffer(base.read_bytes(), np.uint8), np.frombuffer(out.read_bytes(), np.uint8)
    assert a.size == b.size
    changed = np.flatnonzero(a != b)
    grid = read_layout(base).raw_grid()
    expected = grid.byte_offsets(np.arange(48, dtype=np.intp), np.full(48, 10, np.intp))
    assert set(changed) <= set(expected) | set(expected + 1)


def test_empty_patch_is_byte_identical(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(digests="valid"))
    empty = RawPatch(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0, np.uint16))
    result = WRITER.write(base, tmp_path / "o.dng", empty, preview=PreviewMode.KEEP, overwrite=False)
    assert result.path.read_bytes() == base.read_bytes()
    assert result.digest is DigestHandling.UNCHANGED


def test_valid_digests_are_recomputed(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(digests="valid"))
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert result.digest is DigestHandling.RECOMPUTED
    ifd0 = read_layout(result.path).ifd0
    data = read_raw(result.path).data
    assert bytes(ifd0.value(RAW_IMAGE_DIGEST)) == legacy_raw_digest(data)
    assert bytes(ifd0.value(NEW_RAW_IMAGE_DIGEST)) == new_raw_digest(data)


def test_unverifiable_digests_are_removed(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(digests="stale"))
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert result.digest is DigestHandling.REMOVED
    entries = read_layout(result.path).ifd0.entries
    assert RAW_IMAGE_DIGEST not in entries
    assert NEW_RAW_IMAGE_DIGEST not in entries
    assert SUB_IFDS in entries


def test_no_digests_reported_absent(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    assert WRITER.write(
        base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False
    ).digest is (DigestHandling.ABSENT)


def test_new_digest_uses_256_tiles() -> None:
    data = raw_frame(300, 520)
    import hashlib as h  # noqa: PLC0415

    tiles = [data[y : y + 256, x : x + 256] for y in (0, 256) for x in (0, 256, 512)]
    expected = h.md5(b"".join(h.md5(t.astype("<u2").tobytes()).digest() for t in tiles)).digest()
    assert new_raw_digest(data) == expected
    assert legacy_raw_digest(data) == h.md5(data.astype("<u2").tobytes()).digest()


def test_regenerate_replaces_thumbnail_and_jpeg(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.REGENERATE, overwrite=False)
    assert result.preview is PreviewMode.REGENERATE
    layout = read_layout(result.path)
    thumb, jpeg = layout.previews
    assert (thumb.grid(3).read(result.path, np.dtype(np.uint8), 3) == 77).all()
    from PIL import Image  # noqa: PLC0415

    with result.path.open("rb") as fh:
        offset, count = jpeg.ints(273)[0], jpeg.ints(279)[0]
        fh.seek(offset)
        import io  # noqa: PLC0415

        image = np.asarray(Image.open(io.BytesIO(fh.read(count))))
    assert image.shape == (12, 16, 3)
    assert abs(float(image.mean()) - 77) < 3


def test_remove_drops_preview_and_blanks_thumbnail(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.REMOVE, overwrite=False)
    layout = read_layout(result.path)
    assert result.preview is PreviewMode.REMOVE
    assert len(layout.sub_ifds) == 1
    assert layout.previews == (layout.ifd0,)
    assert (layout.ifd0.grid(3).read(result.path, np.dtype(np.uint8), 3) == 128).all()
    np.testing.assert_array_equal(read_raw(result.path).data[:, 11], read_raw(base).data[:, 11])


def test_keep_leaves_previews(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert len(read_layout(result.path).previews) == 2


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (DngSpec(compression=7), "compressed"),
        (DngSpec(photometric=34892), "linear"),
        (DngSpec(bits=8, raw=raw_frame() % 255), "bits"),
        (DngSpec(linearization=True), "LinearizationTable"),
    ],
    ids=["compressed", "linear", "8-bit", "linearization"],
)
def test_rejects_unpatchable(tmp_path: Path, spec: DngSpec, match: str) -> None:
    base = write_dng(tmp_path / "base.dng", spec)
    with pytest.raises(DngFormatError, match=match):
        WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert not (tmp_path / "o.dng").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["base.dng"]


def test_rejects_non_dng(tmp_path: Path) -> None:
    bogus = tmp_path / "x.dng"
    bogus.write_bytes(b"not a tiff")
    with pytest.raises(DngFormatError, match="not a readable"):
        WRITER.read_raw(bogus)


def test_refuses_existing_output_unless_overwrite(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    out = tmp_path / "o.dng"
    out.write_bytes(b"keep me")
    with pytest.raises(OutputExistsError):
        WRITER.write(base, out, _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert out.read_bytes() == b"keep me"
    WRITER.write(base, out, _column_patch(), preview=PreviewMode.KEEP, overwrite=True)
    assert read_raw(out).data[0, 10] == 9
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.dng", "o.dng"]


def test_patch_outside_raw_fails_without_output(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    bad = RawPatch(np.array([0], np.intp), np.array([999], np.intp), np.array([1], np.uint16))
    with pytest.raises(DngFormatError, match="outside"):
        WRITER.write(base, tmp_path / "o.dng", bad, preview=PreviewMode.KEEP, overwrite=False)
    assert not (tmp_path / "o.dng").exists()


def test_reads_layout_tags(tmp_path: Path) -> None:
    raw = read_raw(write_dng(tmp_path / "b.dng", DngSpec(black_level=(500, 501, 502, 503), cfa=b"\x01\x00\x02\x01")))
    assert raw.cfa_pattern == "GRBG"
    assert raw.active_area == (2, 4, 48, 64)
    assert raw.default_crop_origin == (2, 2)
    assert raw.black_level == (500, 501, 502, 503)
    assert raw.white_level == 12000


def test_rawpy_opens_patched_real_layout(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(raw=raw_frame(64, 96), active_area=(0, 0, 64, 96)))
    out = TiffDngRawWriter().write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    with rawpy.imread(str(out.path)) as raw:
        assert raw.raw_image.shape == (64, 96)
        assert int(raw.raw_image[5, 10]) == 9
