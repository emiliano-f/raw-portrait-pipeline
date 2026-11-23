from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import RetouchConfig
from raw_pipeline.core.registries import retoucher_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, FaceRegion
from raw_pipeline.stages.retouchers import FrequencySeparationRetoucher, NullRetoucher
from raw_pipeline.stages.retouchers.skin_mask import SkinMaskBuilder
from tests.unit.stages.retouchers.faces import mesh_face, skin_image


def retoucher(strength: float = 0.5, feather: int = 6) -> FrequencySeparationRetoucher:
    return FrequencySeparationRetoucher(strength, 0.05, SkinMaskBuilder(feather_px=feather))


def skin_of(face: FaceRegion, feather: int = 6) -> np.ndarray:
    return SkinMaskBuilder(feather).build(face, (0, 0), (200, 200)).hard


def band_std(img: np.ndarray, mask: np.ndarray, lo: float, hi: float | None) -> float:
    g = img[..., 1]
    low = cv2.GaussianBlur(g, (0, 0), lo)
    band = low - cv2.GaussianBlur(g, (0, 0), hi) if hi else g - low
    return float(band[mask].std())


def test_changes_nothing_outside_mask_and_on_features() -> None:
    synth = mesh_face()
    img = skin_image()
    result = retoucher().retouch(img, [synth.face])
    untouched = ~skin_of(synth.face)
    np.testing.assert_array_equal(result.image[untouched], img[untouched])
    features = synth.feature_mask(200, 200)
    np.testing.assert_array_equal(result.image[features], img[features])
    assert not np.array_equal(result.image, img)


def test_does_not_mutate_input_and_keeps_dtype() -> None:
    img = skin_image()
    before = img.copy()
    out = retoucher().retouch(img, [mesh_face().face]).image
    np.testing.assert_array_equal(img, before)
    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_smooths_blotches_and_keeps_texture() -> None:
    synth = mesh_face()
    img = skin_image()
    out = retoucher(strength=1.0).retouch(img, [synth.face]).image
    core = cv2.erode(skin_of(synth.face).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert band_std(out, core, 1.0, 12.0) < 0.9 * band_std(img, core, 1.0, 12.0)
    assert band_std(out, core, 1.0, None) > 0.95 * band_std(img, core, 1.0, None)


def test_strength_scales_effect() -> None:
    img = skin_image()
    face = mesh_face().face
    weak = retoucher(0.2).retouch(img, [face]).image - img
    strong = retoucher(0.8).retouch(img, [face]).image - img
    np.testing.assert_allclose(strong, 4.0 * weak, atol=1e-6)


@pytest.mark.parametrize("strength", [0.0])
def test_zero_strength_is_passthrough(strength: float) -> None:
    img = skin_image()
    result = retoucher(strength).retouch(img, [mesh_face().face])
    assert result.image is img
    assert result.params["strength"] == 0.0


def test_no_faces_is_passthrough() -> None:
    img = skin_image()
    result = retoucher().retouch(img, [])
    assert result.image is img
    assert result.params["faces_retouched"] == 0
    assert result.params["mask_coverage"] == 0.0


def test_face_outside_image_is_ignored() -> None:
    img = skin_image()
    face = FaceRegion(box=BoundingBox(500, 500, 600, 600), confidence=0.9, detector="x")
    assert retoucher().retouch(img, [face]).image is img


def test_ellipse_fallback_and_partial_face_at_border() -> None:
    img = skin_image()
    face = FaceRegion(box=BoundingBox(-30, 40, 90, 190), confidence=0.9, detector="opencv")
    result = retoucher().retouch(img, [face])
    assert result.params["mask_source"] == "ellipse"
    assert 0.0 < float(result.params["mask_coverage"]) < 0.3  # type: ignore[arg-type]
    np.testing.assert_array_equal(result.image[:, 120:], img[:, 120:])


def test_params_recorded_and_deterministic() -> None:
    img = skin_image()
    face = mesh_face().face
    a = retoucher().retouch(img, [face])
    b = retoucher().retouch(img, [face])
    np.testing.assert_array_equal(a.image, b.image)
    assert dict(a.params) == dict(b.params)
    assert a.params["mask_source"] == "landmarks"
    assert a.params["faces_retouched"] == 1
    assert a.params["strength"] == 0.5
    assert a.params["blur_radius_px"] == pytest.approx(6.0)
    assert a.params["mask_feather_px"] == 6


def test_two_faces_mixed_sources() -> None:
    img = np.concatenate([skin_image(), skin_image(seed=1)], axis=1)
    left = mesh_face().face
    right = FaceRegion(box=BoundingBox(240, 15, 360, 165), confidence=0.9, detector="opencv")
    result = retoucher().retouch(img, [left, right])
    assert result.params["mask_source"] == "mixed"
    assert result.params["faces_retouched"] == 2


def test_overlapping_face_features_are_protected() -> None:
    img = np.concatenate([skin_image(), skin_image(seed=1)], axis=1)
    left = mesh_face(cx=100.0)
    right = mesh_face(cx=170.0)
    result = retoucher().retouch(img, [left.face, right.face])
    features = left.feature_mask(200, 400) | right.feature_mask(200, 400)
    np.testing.assert_array_equal(result.image[features], img[features])
    assert not np.array_equal(result.image, img)


def test_registered_and_built_from_config() -> None:
    cfg = AppConfig(retouch=RetouchConfig(strength=0.1, mask_feather_px=3, blur_radius_face_fraction=0.02))
    env = BuildEnv(config=cfg, output_dir=Path("out"))
    built = retoucher_registry.create("frequency_separation", env)
    assert built == FrequencySeparationRetoucher(0.1, 0.02, SkinMaskBuilder(3))
    assert isinstance(retoucher_registry.create("none", env), NullRetoucher)


def test_null_retoucher_returns_same_array() -> None:
    img = skin_image()
    result = NullRetoucher().retouch(img, [mesh_face().face])
    assert result.image is img
    assert result.params["mask_coverage"] == 0.0
