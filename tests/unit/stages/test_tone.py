from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import ToneConfig
from raw_pipeline.core.artifacts import EXPOSURE
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import ExposureMeasurement
from raw_pipeline.stages.tone import MAX_SHADOW_LIFT_EV, ToneCurve, ToneStage
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context, flat_image, gradient_image

DEFAULT = ToneCurve(-1.0, 1.5, 0.3, 0.05)


def curve(**kw: float) -> ToneCurve:
    params = {
        "highlight_rolloff_start_ev": -1.0,
        "highlight_rolloff_ev": 1.5,
        "shadow_lift_ev": 0.3,
        "shadow_pivot_luminance": 0.05,
    }
    params.update(kw)
    return ToneCurve(**params)


def with_ev(ctx: ImageContext, ev: float) -> ImageContext:
    return ctx.with_artifact(EXPOSURE, ExposureMeasurement(0.09, 0.18, ev, "face_weighted", 1))


class TestToneStageContract(StageContract):
    def make_stage(self) -> Stage:
        return ToneStage(DEFAULT)


class TestToneStageWithExposureContract(StageContract):
    def make_stage(self) -> Stage:
        return ToneStage(DEFAULT)

    def make_context(self) -> ImageContext:
        return with_ev(decoded_context(gradient_image(peak=4.0)), 1.3)


def ramp(lo: float = -0.01, hi: float = 20.0, n: int = 200_001) -> np.ndarray:
    return np.linspace(lo, hi, n, dtype=np.float64).astype(np.float32)


@pytest.mark.parametrize(
    "c",
    [DEFAULT, curve(shadow_lift_ev=0.0), curve(shadow_lift_ev=2.8), curve(highlight_rolloff_start_ev=-0.1)],
)
def test_curve_is_monotonic_and_finite(c: ToneCurve) -> None:
    y = c.apply(ramp())
    assert np.isfinite(y).all()
    assert np.all(np.diff(y) >= 0)


def test_curve_is_identity_in_mid_tones() -> None:
    x = np.linspace(0.05, 0.5, 1000, dtype=np.float32)
    np.testing.assert_array_equal(DEFAULT.apply(x), x)


def test_zero_lift_is_identity_below_shoulder() -> None:
    x = np.linspace(0.0, 0.5, 1000, dtype=np.float32)
    np.testing.assert_array_equal(curve(shadow_lift_ev=0.0).apply(x), x)


def test_shoulder_maps_scene_white_to_one_and_keeps_headroom() -> None:
    c = DEFAULT
    assert math.isclose(float(c.apply(np.float32([2**1.5]))[0]), 1.0, rel_tol=1e-5)
    beyond = float(c.apply(np.float32([100.0]))[0])
    assert 1.0 < beyond < c.shoulder_start + c.shoulder_scale


def test_shadow_lift_at_black_and_pivot() -> None:
    c = curve(shadow_lift_ev=0.5)
    tiny = np.float32([1e-6])
    assert math.isclose(float(c.apply(tiny)[0] / tiny[0]), 2**0.5, rel_tol=1e-4)
    assert float(c.apply(np.float32([0.05]))[0]) == pytest.approx(0.05, rel=1e-6)
    assert float(c.gain(np.float32([0.0]))[0]) == pytest.approx(2**0.5)
    assert float(c.gain(np.float32([-0.01]))[0]) == pytest.approx(2**0.5)


@pytest.mark.parametrize("knee", [DEFAULT.shoulder_start, DEFAULT.shadow_pivot_luminance])
def test_curve_is_c1_at_knees(knee: float) -> None:
    eps = 1e-4
    x = np.float64(knee)
    f = lambda v: float(DEFAULT.apply(np.float32([v]))[0])  # noqa: E731
    left = (f(x) - f(x - eps)) / eps
    right = (f(x + eps) - f(x)) / eps
    assert left == pytest.approx(1.0, abs=5e-3)
    assert right == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize(
    "kw",
    [
        {"highlight_rolloff_start_ev": 0.0},
        {"highlight_rolloff_ev": 0.0},
        {"shadow_lift_ev": MAX_SHADOW_LIFT_EV},
        {"shadow_lift_ev": -0.1},
        {"shadow_pivot_luminance": 0.0},
    ],
)
def test_invalid_curves_rejected(kw: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        curve(**kw)


def test_ev_scaling_exact_in_mid_tones() -> None:
    ctx = with_ev(decoded_context(flat_image(0.1)), 1.0)
    out = ToneStage(DEFAULT).process(ctx)
    np.testing.assert_array_equal(out.image, np.float32(0.2))


def test_no_exposure_means_zero_ev_identity_in_mid_tones() -> None:
    img = flat_image(0.18)
    out = ToneStage(DEFAULT).process(decoded_context(img))
    np.testing.assert_array_equal(out.image, img)
    rec = out.metadata.records["tone"]
    assert rec["exposure_ev"] == 0.0
    assert rec["exposure_method"] == "none"


def test_hue_preserving_scaling() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[..., :] = [2.0, 1.2, 0.6]
    out = ToneStage(DEFAULT).process(decoded_context(img)).image
    np.testing.assert_allclose(out[..., 0] / out[..., 1], 2.0 / 1.2, rtol=1e-5)
    np.testing.assert_allclose(out[..., 2] / out[..., 1], 0.6 / 1.2, rtol=1e-5)
    assert float(out[0, 0, 1]) < 1.2


def test_highlights_recovered_and_values_above_one_preserved() -> None:
    img = flat_image(3.0)
    out = ToneStage(DEFAULT).process(decoded_context(img))
    assert 1.0 < float(out.image.max()) < 3.0
    assert out.metadata.records["tone"]["clipped_fraction"] == 1.0


def test_records_all_parameters() -> None:
    out = ToneStage(DEFAULT).process(with_ev(decoded_context(), 0.5))
    assert dict(out.metadata.records["tone"]) == {
        "clipped_fraction": pytest.approx(0.0),
        "exposure_ev": 0.5,
        "exposure_method": "face_weighted",
        "highlight_rolloff_ev": 1.5,
        "highlight_rolloff_start_ev": -1.0,
        "shadow_lift_ev": 0.3,
        "shadow_pivot_luminance": 0.05,
    }


def test_registered_and_built_from_config() -> None:
    cfg = AppConfig(tone=ToneConfig(shadow_lift_ev=0.2))
    stage = stage_registry.create("tone", BuildEnv(config=cfg, output_dir=Path("out")))
    assert isinstance(stage, ToneStage)
    assert stage.curve.shadow_lift_ev == 0.2
