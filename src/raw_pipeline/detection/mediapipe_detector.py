"""MediaPipe Tasks face detector with the 478-point face mesh.

MediaPipe is imported lazily: it is not installed on platforms without a
wheel (Intel macOS), and the meter stage then falls back to the OpenCV
detector. Models are vendored in ``raw_pipeline/detection/models`` so the
pipeline runs offline (see the README there for source, licence and hashes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.errors import DetectionError
from raw_pipeline.core.registries import face_detector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import BoundingBox, FaceRegion

FACE_DETECTOR_MODEL: Final = "blaze_face_short_range.tflite"
"""BlazeFace short-range detector (boxes and scores)."""
FACE_LANDMARKER_MODEL: Final = "face_landmarker.task"
"""Face Landmarker v2 bundle (478-point mesh)."""
LANDMARK_SCHEME: Final = "mediapipe_face_mesh_478"
"""``FaceRegion.landmark_scheme`` of the meshes produced here."""
MESH_POINTS: Final = 478
DEFAULT_MAX_FACES: Final = 8
"""Upper bound on faces meshed per image."""


def default_model_dir() -> Path:
    """Directory holding the vendored MediaPipe models (inside the installed package)."""
    return Path(str(resources.files("raw_pipeline.detection") / "models"))


@dataclass(frozen=True)
class RawDetection:
    """A box straight out of a backend, in preview pixels (unclipped)."""

    box: BoundingBox
    score: float


class FaceBackend(Protocol):
    """What :class:`MediaPipeDetector` needs from the MediaPipe Tasks runtime."""

    def detect_boxes(self, preview: NDArray[np.uint8]) -> list[RawDetection]:
        """Return face boxes in preview pixels."""
        ...

    def detect_meshes(self, preview: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """Return one ``(478, 2)`` array of ``(x, y)`` preview pixels per face."""
        ...


BackendFactory = Callable[[Path, float, int], FaceBackend]
"""Builds a backend from ``(model_dir, min_confidence, max_faces)``."""


class MediaPipeTasksBackend:
    """:class:`FaceBackend` over the MediaPipe Tasks ``FaceDetector`` and ``FaceLandmarker``.

    Each call opens the task, runs it once and closes it, so no native
    handles outlive a call (MediaPipe tasks are neither picklable nor safe to
    finalize at interpreter shutdown). Opening costs a few tens of milliseconds.

    Raises:
        DetectionError: MediaPipe is not installed or a model cannot be loaded.
    """

    def __init__(self, model_dir: Path, min_confidence: float, max_faces: int) -> None:
        self._model_dir = model_dir
        self._min_confidence = min_confidence
        self._max_faces = max_faces
        self._mp, self._vision, self._base_options = _import_mediapipe()

    def _model(self, name: str) -> Any:  # noqa: ANN401 - MediaPipe BaseOptions is untyped
        path = self._model_dir / name
        if not path.is_file():
            msg = f"MediaPipe model not found: {path}"
            raise DetectionError(msg)
        return self._base_options(model_asset_path=str(path))

    def _image(self, preview: NDArray[np.uint8]) -> Any:  # noqa: ANN401 - mp.Image is untyped
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(preview))

    def detect_boxes(self, preview: NDArray[np.uint8]) -> list[RawDetection]:
        """Run BlazeFace on ``preview``."""
        options = self._vision.FaceDetectorOptions(
            base_options=self._model(FACE_DETECTOR_MODEL), min_detection_confidence=self._min_confidence
        )
        try:
            with self._vision.FaceDetector.create_from_options(options) as task:
                result = task.detect(self._image(preview))
        except OSError as exc:
            raise _native_library_error(exc) from exc
        except (RuntimeError, ValueError) as exc:
            msg = f"MediaPipe face detection failed: {exc}"
            raise DetectionError(msg) from exc
        detections = []
        for det in result.detections:
            bb = det.bounding_box
            score = float(det.categories[0].score) if det.categories else 0.0
            box = BoundingBox(
                float(bb.origin_x), float(bb.origin_y), float(bb.origin_x + bb.width), float(bb.origin_y + bb.height)
            )
            detections.append(RawDetection(box=box, score=score))
        return detections

    def detect_meshes(self, preview: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """Run the Face Landmarker on ``preview``."""
        options = self._vision.FaceLandmarkerOptions(
            base_options=self._model(FACE_LANDMARKER_MODEL),
            num_faces=self._max_faces,
            min_face_detection_confidence=self._min_confidence,
        )
        try:
            with self._vision.FaceLandmarker.create_from_options(options) as task:
                result = task.detect(self._image(preview))
        except OSError as exc:
            raise _native_library_error(exc) from exc
        except (RuntimeError, ValueError) as exc:
            msg = f"MediaPipe face landmarker failed: {exc}"
            raise DetectionError(msg) from exc
        height, width = preview.shape[:2]
        scale = np.asarray([width, height], dtype=np.float32)
        return [
            (np.asarray([(p.x, p.y) for p in face], dtype=np.float32) * scale).astype(np.float32)
            for face in result.face_landmarks
        ]


def _native_library_error(exc: OSError) -> DetectionError:
    # The Tasks C library is loaded lazily on first task creation and needs system
    # libraries (e.g. libEGL/libGLES on Linux) that minimal hosts may lack.
    return DetectionError(f"MediaPipe native library could not be loaded ({exc}); use the 'opencv' detector")


def _import_mediapipe() -> tuple[Any, Any, Any]:
    try:
        import mediapipe as mp  # noqa: PLC0415 - optional, platform-dependent dependency
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: PLC0415
    except ImportError as exc:
        msg = f"MediaPipe is not available on this platform ({exc}); use the 'opencv' detector"
        raise DetectionError(msg) from exc
    return mp, vision, BaseOptions


@face_detector_registry.register("mediapipe")
class MediaPipeDetector:
    """Face boxes from BlazeFace plus the 478-point mesh from the Face Landmarker.

    Each mesh is attached to the detected box that contains its centroid
    (largest box first); boxes without a mesh keep ``landmarks=None``.

    Args:
        min_confidence: Minimum detection score in ``[0, 1]``.
        max_faces: Maximum number of faces meshed.
        model_dir: Directory with the vendored models.
        backend_factory: Builds the MediaPipe backend (injected fakes in tests).
    """

    name: ClassVar[str] = "mediapipe"

    def __init__(
        self,
        min_confidence: float,
        max_faces: int = DEFAULT_MAX_FACES,
        model_dir: Path | None = None,
        backend_factory: BackendFactory = MediaPipeTasksBackend,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_faces = max_faces
        self.model_dir = model_dir or default_model_dir()
        self._backend_factory = backend_factory

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from ``meter.min_face_confidence``."""
        return cls(min_confidence=app_config(env).meter.min_face_confidence)

    def detect(self, preview: NDArray[np.uint8]) -> list[FaceRegion]:
        """Detect faces on an sRGB uint8 ``(h, w, 3)`` preview, in preview pixels.

        Raises:
            DetectionError: MediaPipe is unavailable or failed.
        """
        backend = self._backend_factory(self.model_dir, self.min_confidence, self.max_faces)
        detections = [d for d in backend.detect_boxes(preview) if d.score >= self.min_confidence]
        if not detections:
            return []
        height, width = preview.shape[:2]
        detections.sort(key=lambda d: (-d.box.clipped(width, height).area, d.box.x0, d.box.y0))
        meshes = assign_meshes([d.box for d in detections], backend.detect_meshes(preview))
        return [
            FaceRegion(
                box=d.box.clipped(width, height),
                confidence=min(max(d.score, 0.0), 1.0),
                detector=self.name,
                landmarks=mesh,
                landmark_scheme=LANDMARK_SCHEME if mesh is not None else None,
            )
            for d, mesh in zip(detections, meshes, strict=True)
            if d.box.clipped(width, height).area > 0
        ]


def assign_meshes(
    boxes: Sequence[BoundingBox], meshes: Sequence[NDArray[np.float32]]
) -> list[NDArray[np.float32] | None]:
    """Attach each mesh to the first box (in order) containing its centroid.

    Args:
        boxes: Face boxes, in priority order (largest first).
        meshes: ``(N, 2)`` landmark arrays in the boxes' pixel grid.

    Returns:
        One entry per box: its mesh, or ``None`` when no mesh fell inside it.
    """
    assigned: list[NDArray[np.float32] | None] = [None] * len(boxes)
    for mesh in meshes:
        if mesh.shape != (MESH_POINTS, 2):
            continue
        cx, cy = (float(v) for v in mesh.mean(axis=0))
        for i, box in enumerate(boxes):
            if assigned[i] is None and box.x0 <= cx < box.x1 and box.y0 <= cy < box.y1:
                assigned[i] = mesh
                break
    return assigned
