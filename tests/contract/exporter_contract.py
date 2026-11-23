"""Contract every :class:`~raw_pipeline.core.protocols.Exporter` must satisfy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raw_pipeline.core.errors import OutputExistsError
from raw_pipeline.core.protocols import Exporter
from tests.support.synthetic import decoded_context, gradient_image


class ExporterContract:
    """Reusable suite. Override ``make_exporter(overwrite)``."""

    def make_exporter(self, *, overwrite: bool) -> Exporter:
        raise NotImplementedError

    def test_writes_readable_file_with_icc(self, tmp_path: Path) -> None:
        ctx = decoded_context(gradient_image(48, 64))
        path = self.make_exporter(overwrite=False).export(ctx, tmp_path / "out")
        assert path.exists()
        assert path.parent == tmp_path
        with Image.open(path) as img:
            assert img.size == (64, 48)
            assert img.info.get("icc_profile")

    def test_does_not_mutate_input(self, tmp_path: Path) -> None:
        ctx = decoded_context(gradient_image(48, 64))
        before = ctx.image.copy()
        self.make_exporter(overwrite=False).export(ctx, tmp_path / "out")
        np.testing.assert_array_equal(ctx.image, before)

    def test_refuses_silent_overwrite(self, tmp_path: Path) -> None:
        ctx = decoded_context(gradient_image(48, 64))
        self.make_exporter(overwrite=False).export(ctx, tmp_path / "out")
        with pytest.raises(OutputExistsError):
            self.make_exporter(overwrite=False).export(ctx, tmp_path / "out")
        self.make_exporter(overwrite=True).export(ctx, tmp_path / "out")

    def test_byte_identical(self, tmp_path: Path) -> None:
        ctx = decoded_context(gradient_image(48, 64))
        a = self.make_exporter(overwrite=False).export(ctx, tmp_path / "a")
        b = self.make_exporter(overwrite=False).export(ctx, tmp_path / "b")
        assert a.read_bytes() == b.read_bytes()
