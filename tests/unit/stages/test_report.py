from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.core.artifacts import BEFORE_IMAGE, DEFECT_CROPS, EXPOSURE, REPORT_THUMBNAILS, DefectCrops
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import StageError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.values import ExposureMeasurement
from raw_pipeline.exporters.encoding import encode_srgb_uint8
from raw_pipeline.stages.report import ReportStage
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context, flat_image, gradient_image
from tests.unit.exporters.helpers import build_env


class TestReportStageContract(StageContract):
    def make_stage(self) -> Stage:
        return ReportStage(thumbnail_long_edge_px=32)

    def make_context(self) -> ImageContext:
        return decoded_context().with_artifact(BEFORE_IMAGE, gradient_image(32, 48))


def _crops() -> DefectCrops:
    return DefectCrops(
        before=flat_image(0.9, 20, 20), after=flat_image(0.045, 20, 20), sensor_xy=(10, 10), image_xy=(10, 10)
    )


def test_before_after_thumbnails() -> None:
    ctx = decoded_context(gradient_image(60, 90)).with_artifact(BEFORE_IMAGE, gradient_image(30, 45))
    out = ReportStage(48).process(ctx)
    thumbs = out.artifact(REPORT_THUMBNAILS)
    assert thumbs is not None
    assert set(thumbs) == {"before", "after"}
    assert thumbs["before"].shape == (30, 45, 3)  # never upscaled
    assert thumbs["after"].shape == (32, 48, 3)
    assert thumbs["after"].dtype == np.uint8
    assert out.metadata.records["report"] == {
        "defect_crop_ev": None,
        "thumbnail_long_edge_px": 48,
        "thumbnails": "before,after",
    }
    assert out.image is ctx.image


def test_defect_crops_full_size_with_exposure_applied() -> None:
    exposure = ExposureMeasurement(measured_luminance=0.045, target_luminance=0.18, correction_ev=2.0, method="x")
    ctx = decoded_context().with_artifact(DEFECT_CROPS, _crops()).with_artifact(EXPOSURE, exposure)
    out = ReportStage(16).process(ctx)
    thumbs = out.artifact(REPORT_THUMBNAILS)
    assert thumbs is not None
    assert "before" not in thumbs
    assert thumbs["defect_before"].shape == (20, 20, 3)
    np.testing.assert_array_equal(thumbs["defect_after"], encode_srgb_uint8(flat_image(0.18, 20, 20)))
    assert int(thumbs["defect_before"].min()) == 255
    assert out.metadata.records["report"]["defect_crop_ev"] == 2.0


def test_built_from_config(tmp_path: Path) -> None:
    stage = stage_registry.create("report", build_env(tmp_path, report={"thumbnail_long_edge_px": 256}))
    assert isinstance(stage, ReportStage)
    assert stage.thumbnail_long_edge_px == 256


def test_requires_decoded_image() -> None:
    with pytest.raises(StageError):
        ReportStage(32).process(ImageContext.initial(Path("a.CR2")))
