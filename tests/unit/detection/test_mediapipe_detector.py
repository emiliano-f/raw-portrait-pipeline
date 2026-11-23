from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from raw_pipeline.config import AppConfig
from raw_pipeline.core.errors import DetectionError
from raw_pipeline.core.protocols import FaceDetector
from raw_pipeline.core.registries import face_detector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox
from raw_pipeline.detection import mediapipe_detector
from raw_pipeline.detection.mediapipe_detector import (
    FACE_DETECTOR_MODEL,
    FACE_LANDMARKER_MODEL,
    LANDMARK_SCHEME,
    MediaPipeDetector,
    MediaPipeTasksBackend,
    RawDetection,
    assign_meshes,
    default_model_dir,
)
from tests.contract.face_detector_contract import FaceDetectorContract


def _mesh(cx: float, cy: float) -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    return (np.asarray([cx, cy]) + rng.normal(0, 3, (478, 2))).astype(np.float32)


class _FakeBackend:
    def __init__(self, boxes: list[RawDetection], meshes: list[NDArray[np.float32]]) -> None:
        self.boxes = boxes
        self.meshes = meshes

    def detect_boxes(self, preview: NDArray[np.uint8]) -> list[RawDetection]:  # noqa: ARG002
        return self.boxes

    def detect_meshes(self, preview: NDArray[np.uint8]) -> list[NDArray[np.float32]]:  # noqa: ARG002
        return self.meshes


def _detector(backend: _FakeBackend, min_confidence: float = 0.5) -> MediaPipeDetector:
    return MediaPipeDetector(min_confidence, backend_factory=lambda *_: backend)


PREVIEW = np.zeros((240, 320, 3), dtype=np.uint8)


def test_models_are_vendored() -> None:
    for name in (FACE_DETECTOR_MODEL, FACE_LANDMARKER_MODEL):
        path = default_model_dir() / name
        assert path.is_file()
        assert path.stat().st_size > 100_000


def test_registered_and_built_from_config(tmp_path: Path) -> None:
    detector = face_detector_registry.create("mediapipe", BuildEnv(AppConfig(), tmp_path))
    assert isinstance(detector, MediaPipeDetector)
    assert detector.min_confidence == AppConfig().meter.min_face_confidence


def test_boxes_sorted_clipped_filtered_and_meshed() -> None:
    small = RawDetection(BoundingBox(200, 50, 240, 90), 0.9)
    large = RawDetection(BoundingBox(-10, 100, 110, 260), 0.8)
    weak = RawDetection(BoundingBox(10, 10, 60, 60), 0.3)
    faces = _detector(_FakeBackend([small, large, weak], [_mesh(220, 70)])).detect(PREVIEW)
    assert [f.confidence for f in faces] == [0.8, 0.9]
    assert faces[0].box == BoundingBox(0, 100, 110, 240)
    assert faces[0].landmarks is None
    assert faces[0].landmark_scheme is None
    assert faces[1].landmarks is not None
    assert faces[1].landmarks.shape == (478, 2)
    assert faces[1].landmark_scheme == LANDMARK_SCHEME
    assert all(f.detector == "mediapipe" for f in faces)


def test_no_boxes_skips_landmarker() -> None:
    class _NoMesh(_FakeBackend):
        def detect_meshes(self, preview: NDArray[np.uint8]) -> list[NDArray[np.float32]]:  # noqa: ARG002
            raise AssertionError

    assert _detector(_NoMesh([], [])).detect(PREVIEW) == []


def test_assign_meshes_by_centroid_largest_first() -> None:
    outer = BoundingBox(0, 0, 200, 200)
    inner = BoundingBox(50, 50, 100, 100)
    a, b = _mesh(75, 75), _mesh(150, 150)
    assigned = assign_meshes([outer, inner], [a, b])
    assert assigned[0] is a
    assert assigned[1] is None
    assert assign_meshes([outer], [np.zeros((10, 2), np.float32)]) == [None]


def test_missing_mediapipe_raises_detection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unavailable() -> tuple[object, object, object]:
        raise DetectionError("MediaPipe is not available")

    monkeypatch.setattr(mediapipe_detector, "_import_mediapipe", _unavailable)
    with pytest.raises(DetectionError, match="not available"):
        MediaPipeDetector(0.5).detect(PREVIEW)


def test_missing_model_raises_detection_error(tmp_path: Path) -> None:
    pytest.importorskip("mediapipe")
    with pytest.raises(DetectionError, match="model not found"):
        MediaPipeDetector(0.5, model_dir=tmp_path).detect(PREVIEW)


def test_real_backend_clips_landmarks_to_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    vision = pytest.importorskip("mediapipe.tasks.python.vision")

    class _Point:
        def __init__(self, x: float, y: float) -> None:
            self.x, self.y = x, y

    class _Result:
        face_landmarks: ClassVar = [[_Point(-0.1, 0.5), _Point(0.5, 1.2), _Point(0.25, 0.75)]]

    class _Task:
        def __enter__(self) -> _Task:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def detect(self, _image: object) -> _Result:
            return _Result()

    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", lambda *_: _Task())
    monkeypatch.setattr(MediaPipeTasksBackend, "_image", lambda *_: None)  # mp.Image needs the native library
    meshes = MediaPipeTasksBackend(default_model_dir(), 0.5, 1).detect_meshes(PREVIEW)
    np.testing.assert_allclose(meshes[0], [[0, 120], [160, 240], [80, 180]])


def _real_mediapipe() -> MediaPipeDetector:
    """The real detector, or skip when MediaPipe or its native libraries cannot load here."""
    detector = MediaPipeDetector(0.6)
    try:
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    except DetectionError as exc:
        pytest.skip(f"MediaPipe unusable on this host: {exc}")
    return detector


def test_native_library_failure_raises_detection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    vision = pytest.importorskip("mediapipe.tasks.python.vision")

    def _no_egl(*_: object) -> object:
        raise OSError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(vision.FaceDetector, "create_from_options", _no_egl)
    with pytest.raises(DetectionError, match="native library"):
        MediaPipeDetector(0.5).detect(PREVIEW)


class TestMediaPipeDetectorContract(FaceDetectorContract):
    def make_detector(self) -> FaceDetector:
        return _real_mediapipe()


def _face_samples() -> list[Path]:
    root = os.environ.get("RAW_PIPELINE_FACE_SAMPLES")
    if not root:
        return []
    return sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


@pytest.mark.slow
@pytest.mark.parametrize("path", _face_samples() or [pytest.param(None, marks=pytest.mark.skip("no face samples"))])
def test_real_face_found_with_mesh(path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    preview = np.asarray(Image.open(path).convert("RGB"))
    faces = _real_mediapipe().detect(preview)
    assert faces
    assert faces[0].landmarks is not None
    assert faces[0].landmarks.shape == (478, 2)
