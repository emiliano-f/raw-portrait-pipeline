from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.core.artifacts import FACES
from raw_pipeline.core.color import luminance
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import MeteringError
from raw_pipeline.core.protocols import MeteringStrategy
from raw_pipeline.core.registries import metering_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, FaceRegion
from raw_pipeline.metering.center_weighted import CenterWeightedMetering, center_weights
from raw_pipeline.metering.face_weighted import FaceWeightedMetering, metered_faces, skin_region
from raw_pipeline.metering.stats import correction_ev, subsampled_luminance, trimmed_mean, weighted_median
from tests.contract.metering_contract import MeteringContract
from tests.support.synthetic import decoded_context, face_image, flat_image


def _whole_frame_face(ctx: ImageContext) -> ImageContext:
    h, w = ctx.image.shape[:2]
    return ctx.with_artifact(FACES, (FaceRegion(BoundingBox(0, 0, w, h), 0.99, "synthetic"),))


class TestFaceWeightedContract(MeteringContract):
    def make_strategy(self) -> MeteringStrategy:
        return FaceWeightedMetering(self.target, self.max_correction_ev)

    def make_context(self, value: float) -> ImageContext:
        return _whole_frame_face(super().make_context(value))


class TestCenterWeightedContract(MeteringContract):
    def make_strategy(self) -> MeteringStrategy:
        return CenterWeightedMetering(self.target, self.max_correction_ev, 0.25)


@pytest.mark.parametrize(
    ("name", "cls"), [("face_weighted", FaceWeightedMetering), ("center_weighted", CenterWeightedMetering)]
)
def test_registered_and_built_from_config(name: str, cls: type, tmp_path: Path) -> None:
    strategy = metering_registry.create(name, BuildEnv(AppConfig(), tmp_path))
    assert isinstance(strategy, cls)
    assert strategy.name == name


def test_face_patch_lands_on_target() -> None:
    image, face = face_image(face_luminance=0.05, background=0.6)
    ctx = decoded_context(image).with_artifact(FACES, (face,))
    m = FaceWeightedMetering(0.18, 2.0).measure(ctx)
    assert m.method == "face_weighted"
    assert m.faces_used == 1
    assert math.isclose(m.measured_luminance, 0.05, rel_tol=1e-5)
    corrected = image * np.float32(2.0**m.correction_ev)
    patch = corrected[int(face.box.y0) + 5 : int(face.box.y1) - 5, int(face.box.x0) + 5 : int(face.box.x1) - 5]
    assert math.isclose(float(luminance(patch).mean()), 0.18, rel_tol=1e-3)


def test_hair_and_background_inside_box_are_ignored() -> None:
    image, face = face_image(face_luminance=0.1, background=0.1)
    box = face.box
    image[int(box.y0) : int(box.y0 + 0.15 * box.height), int(box.x0) : int(box.x1)] = 0.005
    image[int(box.y0) : int(box.y1), int(box.x0) : int(box.x0 + 0.2 * box.width)] = 0.9
    m = FaceWeightedMetering(0.18, 2.0).measure(decoded_context(image).with_artifact(FACES, (face,)))
    assert math.isclose(m.measured_luminance, 0.1, rel_tol=1e-5)


def test_specular_highlights_and_eyes_are_trimmed() -> None:
    image, face = face_image(face_luminance=0.1, background=0.1)
    rng = np.random.default_rng(3)
    region = skin_region(face.box)
    ys = rng.integers(int(region.y0), int(region.y1), 40)
    xs = rng.integers(int(region.x0), int(region.x1), 40)
    image[ys[:20], xs[:20]] = 4.0
    image[ys[20:], xs[20:]] = 0.0
    m = FaceWeightedMetering(0.18, 2.0).measure(decoded_context(image).with_artifact(FACES, (face,)))
    assert math.isclose(m.measured_luminance, 0.1, rel_tol=1e-5)


def test_small_background_faces_are_not_metered() -> None:
    big = FaceRegion(BoundingBox(0, 0, 100, 100), 0.9, "x")
    tiny = FaceRegion(BoundingBox(0, 0, 10, 10), 0.9, "x")
    assert metered_faces([big, tiny]) == [big]
    assert metered_faces([]) == []


def test_faces_combined_by_area() -> None:
    image = flat_image(0.3, 100, 200)
    image[:, :100] = 0.1
    image[:, 100:] = 0.2
    left = FaceRegion(BoundingBox(0, 0, 100, 100), 0.9, "x")
    right = FaceRegion(BoundingBox(120, 20, 180, 80), 0.9, "x")
    m = FaceWeightedMetering(0.18, 2.0).measure(decoded_context(image).with_artifact(FACES, (left, right)))
    expected = (0.1 * 100 * 100 + 0.2 * 60 * 60) / (100 * 100 + 60 * 60)
    assert m.faces_used == 2
    assert math.isclose(m.measured_luminance, expected, rel_tol=1e-5)


def test_face_weighted_without_faces_raises() -> None:
    with pytest.raises(MeteringError):
        FaceWeightedMetering(0.18, 2.0).measure(decoded_context(flat_image()))
    empty = decoded_context(flat_image()).with_artifact(FACES, (FaceRegion(BoundingBox(5, 5, 5, 5), 0.9, "x"),))
    with pytest.raises(MeteringError):
        FaceWeightedMetering(0.18, 2.0).measure(empty)


def test_center_dominates_center_weighted() -> None:
    image = flat_image(0.8, 120, 160)
    image[20:100, 25:135] = 0.09
    m = CenterWeightedMetering(0.18, 2.0, 0.25).measure(decoded_context(image))
    assert m.method == "center_weighted"
    assert m.faces_used == 0
    assert math.isclose(m.measured_luminance, 0.09, rel_tol=1e-5)
    assert math.isclose(m.correction_ev, 1.0, abs_tol=1e-4)


def test_center_weights_peak_at_centre() -> None:
    w = center_weights(5, 7, 0.25)
    assert w.shape == (5, 7)
    assert np.unravel_index(int(np.argmax(w)), w.shape) == (2, 3)
    assert math.isclose(float(w.max()), 1.0)


@pytest.mark.parametrize(
    ("measured", "expected"), [(0.18, 0.0), (0.09, 1.0), (0.36, -1.0), (0.001, 2.0), (50.0, -2.0), (0.0, 2.0)]
)
def test_correction_ev_clamped(measured: float, expected: float) -> None:
    assert math.isclose(correction_ev(measured, 0.18, 2.0), expected, abs_tol=1e-9)


def test_clamp_both_strategies() -> None:
    bright = decoded_context(flat_image(10.0))
    for strategy, ctx in (
        (FaceWeightedMetering(0.18, 1.5), _whole_frame_face(bright)),
        (CenterWeightedMetering(0.18, 1.5, 0.25), bright),
    ):
        assert strategy.measure(ctx).correction_ev == -1.5


def test_stats_helpers() -> None:
    assert trimmed_mean(np.asarray([0, 1, 1, 1, 100], np.float32), 0.2) == 1.0
    assert trimmed_mean(np.asarray([np.nan, 2.0], np.float32), 0.4) == 2.0
    assert weighted_median(np.asarray([1, 2, 3], np.float32), np.asarray([0, 0, 1], np.float32)) == 3.0
    with pytest.raises(MeteringError):
        trimmed_mean(np.asarray([np.nan], np.float32), 0.1)
    with pytest.raises(MeteringError):
        weighted_median(np.asarray([1.0], np.float32), np.asarray([0.0], np.float32))
    with pytest.raises(MeteringError):
        subsampled_luminance(np.zeros((0, 0, 3), np.float32))
    assert subsampled_luminance(np.zeros((3000, 2000, 3), np.float32)).shape == (1000, 667)
