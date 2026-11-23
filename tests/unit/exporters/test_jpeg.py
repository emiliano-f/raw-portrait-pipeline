from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from raw_pipeline.core.protocols import Exporter
from raw_pipeline.core.registries import exporter_registry
from raw_pipeline.exporters.encoding import encode_srgb_uint8, srgb_icc_profile
from raw_pipeline.exporters.jpeg import JpegExporter
from tests.contract.exporter_contract import ExporterContract
from tests.support.synthetic import decoded_context, flat_image, gradient_image
from tests.unit.exporters.helpers import CANON_EXIF, build_env


class TestJpegExporterContract(ExporterContract):
    def make_exporter(self, *, overwrite: bool) -> Exporter:
        return JpegExporter(quality=92, long_edge_px=None, keep_gps=False, overwrite=overwrite)


def test_registered_and_built_from_config(tmp_path: Path) -> None:
    exporter = exporter_registry.create(
        "jpeg", build_env(tmp_path, overwrite=True, export={"jpeg_quality": 75, "long_edge_px": 100, "keep_gps": True})
    )
    assert isinstance(exporter, JpegExporter)
    assert (exporter.quality, exporter.long_edge_px, exporter.keep_gps, exporter.overwrite) == (75, 100, True, True)


def test_pixels_are_srgb_encoded(tmp_path: Path) -> None:
    ctx = decoded_context(flat_image(0.18, 32, 48))
    path = JpegExporter(quality=100, long_edge_px=None, keep_gps=False, overwrite=False).export(ctx, tmp_path / "o")
    assert path.name == "o.jpg"
    with Image.open(path) as img:
        assert img.info["icc_profile"] == srgb_icc_profile()
        decoded = np.asarray(img)
    expected = int(encode_srgb_uint8(flat_image(0.18, 1, 1))[0, 0, 0])
    assert np.abs(decoded.astype(int) - expected).max() <= 1


def test_resizes_to_long_edge(tmp_path: Path) -> None:
    ctx = decoded_context(gradient_image(60, 90))
    path = JpegExporter(quality=90, long_edge_px=64, keep_gps=False, overwrite=False).export(ctx, tmp_path / "o")
    with Image.open(path) as img:
        assert img.size == (64, 43)


def test_writes_exif_without_gps(tmp_path: Path) -> None:
    ctx = decoded_context(gradient_image(32, 48))
    ctx = ctx.with_metadata(ctx.metadata.with_exif(CANON_EXIF))
    path = JpegExporter(quality=90, long_edge_px=None, keep_gps=False, overwrite=False).export(ctx, tmp_path / "o")
    with Image.open(path) as img:
        exif = img.getexif()
    assert exif[ExifTags.Base.Model] == "Canon EOS 800D"
    assert exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.ISOSpeedRatings] == 200
    assert not exif.get_ifd(ExifTags.IFD.GPSInfo)
