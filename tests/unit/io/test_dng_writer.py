from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
import rawpy

from raw_pipeline.core.dng import DigestHandling, GpsHandling, PreviewMode, RawPatch, UniqueIdHandling
from raw_pipeline.core.errors import DngFormatError, OutputExistsError
from raw_pipeline.io.dng.digest import legacy_raw_digest, new_raw_digest
from raw_pipeline.io.dng.tiff import (
    NEW_RAW_IMAGE_DIGEST,
    RAW_DATA_UNIQUE_ID,
    RAW_IMAGE_DIGEST,
    SUB_IFDS,
    read_layout,
    read_raw,
)
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


def test_empty_patch_skips_preview_regeneration(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    empty = RawPatch(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0, np.uint16))
    result = WRITER.write(base, tmp_path / "o.dng", empty, preview=PreviewMode.REGENERATE, overwrite=False)
    assert result.path.read_bytes() == base.read_bytes()
    assert result.preview is PreviewMode.KEEP


def test_failed_check_publishes_nothing(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng")
    out = tmp_path / "o.dng"
    out.write_bytes(b"previous")
    checked: list[Path] = []

    def check(path: Path) -> None:
        checked.append(path)
        assert read_raw(path).data[0, 10] == 9
        raise DngFormatError("bad")

    with pytest.raises(DngFormatError, match="bad"):
        WRITER.write(base, out, _column_patch(), preview=PreviewMode.KEEP, overwrite=True, check=check)
    assert out.read_bytes() == b"previous"
    assert checked
    assert checked[0] != out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.dng", "o.dng"]


def test_render_failure_is_a_dng_error(tmp_path: Path) -> None:
    from raw_pipeline.io.dng.preview import render_preview  # noqa: PLC0415

    junk = tmp_path / "junk.dng"
    junk.write_bytes(b"II*\x00junk")
    with pytest.raises(DngFormatError, match="preview"):
        render_preview(junk)


XMP_WITH_GPS = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
    b' exif:GPSLatitude="34,36.0S" xmp:Rating="3">'
    b"<exif:GPSLongitude>58,24.0W</exif:GPSLongitude><exif:GPSVersionID/>"
    b"</rdf:Description></rdf:RDF></x:xmpmeta>"
)
GPS_LATITUDE = struct.pack("<6I", 34, 1, 36, 1, 7, 1)


def _gps_tags(path: Path) -> tuple[bool, bytes]:
    import tifffile  # noqa: PLC0415

    with tifffile.TiffFile(path) as tif:
        tags = tif.pages.first.tags
        return 34853 in tags, bytes(tags[700].value) if 700 in tags else b""


@pytest.mark.parametrize("empty_patch", [False, True])
def test_gps_is_stripped_without_touching_raw_or_digests(tmp_path: Path, empty_patch: bool) -> None:
    spec = DngSpec(gps=True, xmp=XMP_WITH_GPS, digests="valid")
    base = write_dng(tmp_path / "base.dng", spec)
    assert GPS_LATITUDE in base.read_bytes()
    patch = (
        RawPatch(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0, np.uint16)) if empty_patch else _column_patch()
    )
    result = WRITER.write(base, tmp_path / "o.dng", patch, preview=PreviewMode.KEEP, overwrite=False)
    assert result.gps is GpsHandling.STRIPPED
    data = result.path.read_bytes()
    has_gps, xmp = _gps_tags(result.path)
    assert not has_gps
    assert GPS_LATITUDE not in data
    assert b"GPS" not in xmp
    assert b'xmp:Rating="3"' in xmp
    assert len(xmp) == len(XMP_WITH_GPS)
    expected = spec.raw.copy()
    expected[patch.ys, patch.xs] = patch.values
    np.testing.assert_array_equal(read_raw(result.path).data, expected)
    layout = read_layout(result.path)
    assert bytes(layout.ifd0.entries[RAW_IMAGE_DIGEST].value) == legacy_raw_digest(expected)
    assert bytes(layout.ifd0.entries[NEW_RAW_IMAGE_DIGEST].value) == new_raw_digest(expected)
    assert result.digest is (DigestHandling.UNCHANGED if empty_patch else DigestHandling.RECOMPUTED)
    with rawpy.imread(str(result.path)) as raw:
        assert raw.raw_image.shape == spec.raw.shape


def test_keep_gps_copies_gps_metadata(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(gps=True, xmp=XMP_WITH_GPS))
    result = WRITER.write(
        base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False, keep_gps=True
    )
    assert result.gps is GpsHandling.KEPT
    has_gps, xmp = _gps_tags(result.path)
    assert has_gps
    assert xmp == XMP_WITH_GPS
    assert GPS_LATITUDE in result.path.read_bytes()


def test_self_closing_xmp_gps_element_does_not_swallow_later_metadata(tmp_path: Path) -> None:
    xmp = (
        b'<rdf:Description xmlns:ns1="http://ns.adobe.com/exif/1.0/"><ns1:GPSLatitude/>'
        b"<dc:title>keep me</dc:title><ns1:GPSLatitude>34,36.0S</ns1:GPSLatitude></rdf:Description>"
    )
    base = write_dng(tmp_path / "base.dng", DngSpec(xmp=xmp))
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    packet = _gps_tags(result.path)[1]
    assert b"<dc:title>keep me</dc:title>" in packet
    assert b"GPS" not in packet


def test_xmp_gps_attribute_value_with_tag_delimiters_is_stripped(tmp_path: Path) -> None:
    xmp = b'<rdf:Description><exif:GPSLatitude note="a/>b">34,36.0S</exif:GPSLatitude></rdf:Description>'
    base = write_dng(tmp_path / "base.dng", DngSpec(xmp=xmp))
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    packet = _gps_tags(result.path)[1]
    assert b"34,36.0S" not in packet
    assert b"GPS" not in packet


@pytest.mark.parametrize("keep_gps", [False, True])
def test_embedded_original_is_rejected_by_default(tmp_path: Path, keep_gps: bool) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(original_raw=b"CR2 with the defect" * 4))
    with pytest.raises(DngFormatError, match="uncorrected defect"):
        WRITER.write(
            base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False, keep_gps=keep_gps
        )
    assert list(tmp_path.iterdir()) == [base]


def test_allowed_embedded_original_still_needs_keep_gps(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(original_raw=b"CR2 with GPS" * 4))
    with pytest.raises(DngFormatError, match="keep_gps"):
        WRITER.write(
            base,
            tmp_path / "o.dng",
            _column_patch(),
            preview=PreviewMode.KEEP,
            overwrite=False,
            allow_embedded_original=True,
        )
    assert list(tmp_path.iterdir()) == [base]


def test_allowed_embedded_original_is_kept_and_reported(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(original_raw=b"CR2 with GPS" * 4))
    result = WRITER.write(
        base,
        tmp_path / "o.dng",
        _column_patch(),
        preview=PreviewMode.KEEP,
        overwrite=False,
        keep_gps=True,
        allow_embedded_original=True,
    )
    assert result.embedded_original
    assert b"CR2 with GPS" * 4 in result.path.read_bytes()


def test_xmp_gps_under_another_prefix_is_stripped(tmp_path: Path) -> None:
    xmp = b'<rdf:Description xmlns:ns1="http://ns.adobe.com/exif/1.0/" ns1:GPSLatitude="34,36.0S"/>'
    base = write_dng(tmp_path / "base.dng", DngSpec(xmp=xmp))
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert result.gps is GpsHandling.STRIPPED
    assert b"GPS" not in _gps_tags(result.path)[1]


@pytest.mark.parametrize("keep", [False, True])
def test_no_gps_is_reported_absent(tmp_path: Path, keep: bool) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(xmp=b'<x:xmpmeta xmp:Rating="1"/>'))
    result = WRITER.write(
        base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False, keep_gps=keep
    )
    assert result.gps is GpsHandling.ABSENT


@pytest.mark.parametrize("preview", [PreviewMode.REMOVE, PreviewMode.REGENERATE])
def test_gps_stripping_after_preview_edits_keeps_raw_intact(tmp_path: Path, preview: PreviewMode) -> None:
    spec = DngSpec(gps=True)
    base = write_dng(tmp_path / "base.dng", spec)
    patch = _column_patch()
    result = WRITER.write(base, tmp_path / "o.dng", patch, preview=preview, overwrite=False)
    assert result.gps is GpsHandling.STRIPPED
    expected = spec.raw.copy()
    expected[patch.ys, patch.xs] = patch.values
    np.testing.assert_array_equal(read_raw(result.path).data, expected)
    assert not _gps_tags(result.path)[0]
    assert GPS_LATITUDE not in result.path.read_bytes()


UNIQUE_ID = bytes(range(16))


def test_raw_data_unique_id_is_replaced_when_photosites_change(tmp_path: Path) -> None:
    spec = DngSpec(unique_id=UNIQUE_ID, digests="valid")
    base = write_dng(tmp_path / "base.dng", spec)
    patch = _column_patch()
    first = WRITER.write(base, tmp_path / "a.dng", patch, preview=PreviewMode.KEEP, overwrite=False)
    second = WRITER.write(base, tmp_path / "b.dng", patch, preview=PreviewMode.KEEP, overwrite=False)
    expected = spec.raw.copy()
    expected[patch.ys, patch.xs] = patch.values
    new_id = hashlib.md5(UNIQUE_ID + new_raw_digest(expected), usedforsecurity=False).digest()
    assert first.unique_id is UniqueIdHandling.REPLACED
    assert first.unique_id_value == second.unique_id_value == new_id.hex()
    layout = read_layout(first.path)
    assert bytes(layout.ifd0.entries[RAW_DATA_UNIQUE_ID].value) == new_id
    assert bytes(layout.ifd0.entries[NEW_RAW_IMAGE_DIGEST].value) == new_raw_digest(expected)
    np.testing.assert_array_equal(read_raw(first.path).data, expected)
    assert len(first.path.read_bytes()) == len(base.read_bytes())


def test_raw_data_unique_id_is_unchanged_in_passthrough(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(unique_id=UNIQUE_ID))
    empty = RawPatch(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0, np.uint16))
    result = WRITER.write(base, tmp_path / "o.dng", empty, preview=PreviewMode.KEEP, overwrite=False)
    assert (result.unique_id, result.unique_id_value) == (UniqueIdHandling.UNCHANGED, UNIQUE_ID.hex())
    assert result.path.read_bytes() == base.read_bytes()


def test_raw_data_unique_id_absent(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec())
    result = WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert (result.unique_id, result.unique_id_value) == (UniqueIdHandling.ABSENT, "")


def test_raw_data_unique_id_of_another_type_is_rejected(tmp_path: Path) -> None:
    base = write_dng(tmp_path / "base.dng", DngSpec(unique_id=UNIQUE_ID))
    layout = read_layout(base)
    entry = layout.ifd0.entries[RAW_DATA_UNIQUE_ID]
    data = bytearray(base.read_bytes())
    data[entry.entry_offset + 2 : entry.entry_offset + 12] = struct.pack(layout.byteorder + "HII", 4, 1, 16)
    base.write_bytes(bytes(data))
    with pytest.raises(DngFormatError, match="RawDataUniqueID"):
        WRITER.write(base, tmp_path / "o.dng", _column_patch(), preview=PreviewMode.KEEP, overwrite=False)
    assert not (tmp_path / "o.dng").exists()
