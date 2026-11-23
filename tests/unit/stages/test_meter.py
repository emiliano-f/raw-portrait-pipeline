from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from raw_pipeline.config import AppConfig
from raw_pipeline.core.artifacts import EXPOSURE, FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import DetectionError, MeteringError, StageError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, ExposureMeasurement, FaceRegion
from raw_pipeline.detection.mediapipe_detector import MediaPipeDetector
from raw_pipeline.detection.opencv_detector import OpenCVDetector
from raw_pipeline.metering.center_weighted import CenterWeightedMetering
from raw_pipeline.metering.face_weighted import FaceWeightedMetering
from raw_pipeline.stages.meter import MeterStage, build_preview, preview_gain_ev
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context, face_image, flat_image


@dataclass
class FakeDetector:
    """Returns fixed preview-space boxes; records the preview it saw."""

    boxes: tuple[BoundingBox, ...] = ()
    confidence: float = 0.9
    fail: bool = False
    name: str = "fake"
    seen: list[tuple[int, ...]] = field(default_factory=list)

    def detect(self, preview: NDArray[np.uint8]) -> list[FaceRegion]:
        assert preview.dtype == np.uint8
        self.seen.append(preview.shape)
        if self.fail:
            raise DetectionError("backend unavailable")
        return [FaceRegion(b, self.confidence, self.name) for b in self.boxes]


@dataclass
class FailingStrategy:
    name: ClassVar[str] = "failing"

    def measure(self, ctx: ImageContext) -> ExposureMeasurement:  # noqa: ARG002
        raise MeteringError("cannot measure")


def _stage(detector: FakeDetector, fallback: FakeDetector | None = None, long_edge: int = 100) -> MeterStage:
    return MeterStage(
        detector,
        fallback,
        FaceWeightedMetering(0.18, 2.0),
        CenterWeightedMetering(0.18, 2.0, 0.25),
        detection_long_edge_px=long_edge,
        min_face_confidence=0.6,
    )


class TestMeterStageContract(StageContract):
    def make_stage(self) -> Stage:
        return _stage(FakeDetector((BoundingBox(30, 20, 70, 60),)))


class TestMeterStageNoFaceContract(StageContract):
    def make_stage(self) -> Stage:
        return _stage(FakeDetector(), FakeDetector())


def _face_ctx() -> tuple[ImageContext, FaceRegion]:
    image, face = face_image(face_luminance=0.05, background=0.6)
    return decoded_context(image), face


def _preview_box(face: FaceRegion, scale: float) -> BoundingBox:
    b = face.box
    return BoundingBox(b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale)


def test_faces_mapped_to_full_resolution_and_metered() -> None:
    ctx, face = _face_ctx()
    scale = 100 / max(ctx.image.shape[:2])
    detector = FakeDetector((_preview_box(face, scale),))
    out = _stage(detector).process(ctx)
    assert max(detector.seen[0][:2]) == 100
    (found,) = out.artifact(FACES)
    assert found.box.x0 == pytest.approx(face.box.x0, abs=1.0)
    assert found.box.x1 == pytest.approx(face.box.x1, abs=1.0)
    exposure = out.artifact(EXPOSURE)
    assert exposure is not None
    assert exposure.method == "face_weighted"
    assert exposure.measured_luminance == pytest.approx(0.05, rel=1e-3)
    assert exposure.correction_ev == pytest.approx(math.log2(0.18 / 0.05), rel=1e-3)
    record = out.metadata.records["meter"]
    assert record["detector_used"] == "fake"
    assert record["faces_found"] == 1
    assert record["fallback_used"] is False
    np.testing.assert_array_equal(out.image, ctx.image)


def test_fallback_detector_used_when_primary_fails() -> None:
    ctx, face = _face_ctx()
    scale = 100 / max(ctx.image.shape[:2])
    fallback = FakeDetector((_preview_box(face, scale),), name="backup")
    out = _stage(FakeDetector(fail=True), fallback).process(ctx)
    record = out.metadata.records["meter"]
    assert record["detector_used"] == "backup"
    assert "backend unavailable" in record["detection_errors"]
    assert out.artifact(EXPOSURE).method == "face_weighted"  # type: ignore[union-attr]


def test_low_confidence_faces_dropped_then_center_weighted() -> None:
    ctx, face = _face_ctx()
    out = _stage(FakeDetector((face.box,), confidence=0.3)).process(ctx)
    assert out.artifact(FACES) == ()
    exposure = out.artifact(EXPOSURE)
    assert exposure is not None
    assert exposure.method == "center_weighted"
    assert exposure.faces_used == 0
    assert out.metadata.records["meter"]["fallback_used"] is True


def test_strategy_failure_falls_back() -> None:
    ctx, face = _face_ctx()
    stage = MeterStage(
        FakeDetector((face.box,)),
        None,
        FailingStrategy(),
        CenterWeightedMetering(0.18, 2.0, 0.25),
        detection_long_edge_px=1024,
        min_face_confidence=0.6,
    )
    assert stage.process(ctx).artifact(EXPOSURE).method == "center_weighted"  # type: ignore[union-attr]


def test_undecoded_context_rejected(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        _stage(FakeDetector()).process(ImageContext.initial(tmp_path / "x.CR2"))


def test_from_env_resolves_registered_components(tmp_path: Path) -> None:
    stage = stage_registry.create("meter", BuildEnv(AppConfig(), tmp_path))
    assert isinstance(stage, MeterStage)
    assert isinstance(stage.detector, MediaPipeDetector)
    assert isinstance(stage.fallback_detector, OpenCVDetector)
    assert isinstance(stage.strategy, FaceWeightedMetering)
    assert isinstance(stage.fallback_strategy, CenterWeightedMetering)
    assert stage.detection_long_edge_px == AppConfig().meter.detection_long_edge_px


def test_default_stage_on_blank_image_uses_center_weighted(tmp_path: Path) -> None:
    stage = stage_registry.create("meter", BuildEnv(AppConfig(), tmp_path))
    out = stage.process(decoded_context(flat_image(0.09)))
    exposure = out.artifact(EXPOSURE)
    assert exposure is not None
    assert exposure.method == "center_weighted"
    assert exposure.correction_ev == pytest.approx(1.0, abs=1e-4)


def test_preview_gain_and_encoding() -> None:
    assert preview_gain_ev(flat_image(0.045)) == pytest.approx(2.0, abs=1e-4)
    assert preview_gain_ev(flat_image(0.0)) == 4.0
    ctx = decoded_context(flat_image(0.18, 40, 60))
    from raw_pipeline.core.geometry import PreviewGeometry  # noqa: PLC0415

    preview = build_preview(ctx.image, PreviewGeometry.for_long_edge(60, 40, 30))
    assert preview.shape == (20, 30, 3)
    assert preview.dtype == np.uint8
    assert int(preview[0, 0, 0]) == pytest.approx(118, abs=2)
