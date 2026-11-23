from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile

from raw_pipeline.core.protocols import Exporter
from raw_pipeline.core.registries import exporter_registry
from raw_pipeline.exporters.encoding import encode_srgb_uint16, srgb_icc_profile
from raw_pipeline.exporters.tiff import TiffExporter
from tests.contract.exporter_contract import ExporterContract
from tests.support.synthetic import decoded_context, gradient_image
from tests.unit.exporters.helpers import CANON_EXIF, build_env


class TestTiffExporterContract(ExporterContract):
    def make_exporter(self, *, overwrite: bool) -> Exporter:
        return TiffExporter(long_edge_px=None, overwrite=overwrite)


def test_sixteen_bit_pixels_and_icc(tmp_path: Path) -> None:
    image = gradient_image(32, 48)
    path = TiffExporter(long_edge_px=None, overwrite=False).export(decoded_context(image), tmp_path / "o")
    assert path.name == "o.tif"
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        data = page.asarray()
        assert page.tags["InterColorProfile"].value == srgb_icc_profile()
        assert "Software" in page.tags
        assert "DateTime" not in page.tags
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, encode_srgb_uint16(image))


def test_descriptive_tags_without_gps(tmp_path: Path) -> None:
    ctx = decoded_context(gradient_image(32, 48))
    ctx = ctx.with_metadata(ctx.metadata.with_exif(CANON_EXIF))
    path = TiffExporter(long_edge_px=24, overwrite=False).export(ctx, tmp_path / "o")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {tag.name: tag.value for tag in page.tags}
        assert page.shape == (16, 24, 3)
    assert tags["Make"] == "Canon"
    assert tags["Model"] == "Canon EOS 800D"
    assert tags["DateTime"] == "2026:03:14 15:09:26"
    assert tags["Orientation"] == 1
    assert tags["ExposureTime"] == (1, 250)
    assert tags["FNumber"] == (28, 5)
    assert not any("GPS" in name for name in tags)


def test_built_from_config_and_warns_about_gps(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        exporter = exporter_registry.create("tiff", build_env(tmp_path, export={"keep_gps": True, "long_edge_px": 640}))
    assert isinstance(exporter, TiffExporter)
    assert exporter.long_edge_px == 640
    assert "JPEG outputs only" in caplog.text
    assert "TIFF outputs never contain GPS" in caplog.text
