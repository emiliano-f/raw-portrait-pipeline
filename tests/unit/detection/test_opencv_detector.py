from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.core.errors import DetectionError
from raw_pipeline.core.protocols import FaceDetector
from raw_pipeline.core.registries import face_detector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.detection.opencv_detector import OpenCVDetector, default_cascade_path
from tests.contract.face_detector_contract import FaceDetectorContract


class TestOpenCVDetectorContract(FaceDetectorContract):
    def make_detector(self) -> FaceDetector:
        return OpenCVDetector()


def test_cascade_ships_with_opencv() -> None:
    assert default_cascade_path().is_file()


def test_registered_and_picklable(tmp_path: Path) -> None:
    detector = face_detector_registry.create("opencv", BuildEnv(AppConfig(), tmp_path))
    assert isinstance(detector, OpenCVDetector)
    assert pickle.loads(pickle.dumps(detector)).cascade_path == detector.cascade_path


def test_bad_cascade_raises(tmp_path: Path) -> None:
    with pytest.raises(DetectionError):
        OpenCVDetector(cascade_path=tmp_path / "missing.xml").detect(np.zeros((32, 32, 3), np.uint8))


def _face_samples() -> list[Path]:
    root = os.environ.get("RAW_PIPELINE_FACE_SAMPLES")
    if not root:
        return []
    return sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


@pytest.mark.slow
@pytest.mark.parametrize("path", _face_samples() or [pytest.param(None, marks=pytest.mark.skip("no face samples"))])
def test_real_face_found(path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    faces = OpenCVDetector().detect(np.asarray(Image.open(path).convert("RGB")))
    assert faces
    assert faces[0].confidence == 1.0
