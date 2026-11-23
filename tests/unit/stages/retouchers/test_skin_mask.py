from __future__ import annotations

from dataclasses import replace

import numpy as np

from raw_pipeline.core.values import BoundingBox, FaceRegion
from raw_pipeline.stages.retouchers.skin_mask import SkinMaskBuilder, has_face_mesh
from tests.unit.stages.retouchers.faces import mesh_face

SHAPE = (200, 200)


def test_landmark_mask_excludes_features_and_outside() -> None:
    synth = mesh_face()
    mask = SkinMaskBuilder(feather_px=6).build(synth.face, (0, 0), SHAPE)
    assert mask.source == "landmarks"
    assert not mask.hard[synth.feature_mask(*SHAPE)].any()
    outer = replace(synth.oval, rx=synth.oval.rx + 1, ry=synth.oval.ry + 1)
    assert not mask.hard[~outer.mask(*SHAPE)].any()
    assert mask.hard.mean() > 0.1


def test_soft_mask_is_feathered_and_zero_outside_hard() -> None:
    mask = SkinMaskBuilder(feather_px=6).build(mesh_face().face, (0, 0), SHAPE)
    assert mask.soft.dtype == np.float32
    assert float(mask.soft.min()) >= 0.0
    assert float(mask.soft.max()) == 1.0
    assert not (mask.soft[~mask.hard] != 0).any()
    edge = mask.hard & (mask.soft < 1.0)
    assert edge.any()


def test_zero_feather_gives_hard_mask() -> None:
    mask = SkinMaskBuilder(feather_px=0).build(mesh_face().face, (0, 0), SHAPE)
    np.testing.assert_array_equal(mask.soft, mask.hard.astype(np.float32))


def test_region_origin_offsets_coordinates() -> None:
    synth = mesh_face()
    full = SkinMaskBuilder(feather_px=4).build(synth.face, (0, 0), SHAPE)
    part = SkinMaskBuilder(feather_px=4).build(synth.face, (30, 10), (150, 140))
    np.testing.assert_array_equal(part.hard, full.hard[10:160, 30:170])


def test_ellipse_fallback_without_landmarks() -> None:
    face = FaceRegion(box=BoundingBox(40, 15, 160, 165), confidence=0.9, detector="opencv")
    assert not has_face_mesh(face)
    mask = SkinMaskBuilder(feather_px=4).build(face, (0, 0), SHAPE)
    assert mask.source == "ellipse"
    assert mask.hard.any()
    assert not mask.hard[:15].any()
    assert not mask.hard[15 + int(0.15 * 150) + 1 : 15 + int(0.55 * 150)].any()
    mouth = mask.hard[15 + int(0.62 * 150) + 1 : 15 + int(0.92 * 150), 40 + int(0.22 * 120) + 1 : 40 + int(0.78 * 120)]
    assert not mouth.any()


def test_wrong_scheme_or_shape_falls_back() -> None:
    face = mesh_face().face
    assert has_face_mesh(face)
    assert not has_face_mesh(replace(face, landmark_scheme="other"))
    assert not has_face_mesh(replace(face, landmarks=face.landmarks[:10]))  # type: ignore[index]
    bad = face.landmarks.copy()  # type: ignore[union-attr]
    bad[0] = np.nan
    assert not has_face_mesh(replace(face, landmarks=bad))
