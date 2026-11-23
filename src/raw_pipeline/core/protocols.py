"""Extension-point protocols: the stable contracts of the pipeline.

Every component is registered in one of the registries in
:mod:`raw_pipeline.core.registries` and must satisfy the matching protocol
below and its contract test suite in ``tests/contract/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.raw import (
    CFAFrame,
    DefectCorrectionResult,
    DefectMap,
    DetectedDefect,
    DevelopedImage,
)
from raw_pipeline.core.values import (
    ExifInfo,
    ExposureMeasurement,
    FaceRegion,
    ImageMetadata,
    QAResult,
    RetouchResult,
)


@runtime_checkable
class Stage(Protocol):
    """One step of the per-image pipeline.

    Guarantees every implementation must keep (checked by the contract suite):

    * ``process`` returns a new context and never mutates ``ctx`` or its arrays.
    * Output image is float32 ``(H, W, 3)``; stages other than ``decode`` keep
      the input shape unless documented otherwise.
    * Parameters applied are recorded with ``ctx.with_record(self.name, ...)``.
    * Deterministic: same input and configuration give identical output.
    * Picklable and stateless between images.
    """

    name: ClassVar[str]

    def process(self, ctx: ImageContext) -> ImageContext:
        """Transform ``ctx`` and return the resulting context."""
        ...


class RawHandle(Protocol):
    """An opened RAW file. Obtained from :meth:`RawDecoder.open`."""

    def read_cfa(self) -> CFAFrame:
        """Return a copy of the visible undemosaiced Bayer data."""
        ...

    def develop(self, cfa: CFAFrame) -> DevelopedImage:
        """Demosaic ``cfa`` (possibly defect-corrected) with camera white balance.

        Returns scene-linear float32 in the working space; never clipped.
        """
        ...

    def sensor_to_image(self, x: int, y: int) -> tuple[int, int]:
        """Map raw sensor coordinates to developed-image coordinates (rotation/flip)."""
        ...


class RawDecoder(Protocol):
    """Adapter over a RAW library (LibRaw via rawpy by default)."""

    def open(self, path: Path) -> AbstractContextManager[RawHandle]:
        """Open ``path`` read-only. Raises :class:`~raw_pipeline.core.errors.DecodeError`."""
        ...


class MetadataReader(Protocol):
    """Reads camera EXIF from a RAW file without modifying it."""

    def read(self, path: Path) -> ExifInfo:
        """Return the EXIF of ``path``; missing tags are ``None``."""
        ...


class DefectCorrector(Protocol):
    """Raw-level sensor defect correction, applied *before* demosaicing (ADR 0002).

    Guarantees: the input frame is never mutated; only photosites inside the
    defect mask may change; the output has the input's shape, dtype and layout;
    replacement values come from same-colour CFA neighbours; the ``none``
    implementation returns the input data unchanged.
    """

    name: ClassVar[str]

    def correct(self, frame: CFAFrame, defects: DefectMap) -> DefectCorrectionResult:
        """Return a corrected copy of ``frame`` and the number of photosites replaced."""
        ...


class DefectDetector(Protocol):
    """Finds rows, columns and pixels that are consistent outliers across frames."""

    def detect(self, frames: Iterable[CFAFrame]) -> list[DetectedDefect]:
        """Analyse ``frames`` (dark frames or a normal batch) and return defects found.

        Frames are consumed one at a time; implementations must not keep all of
        them in memory.
        """
        ...


class FaceDetector(Protocol):
    """Detects faces on a downscaled, gamma-encoded preview."""

    name: ClassVar[str]

    def detect(self, preview: NDArray[np.uint8]) -> list[FaceRegion]:
        """Return faces in *preview* pixel coordinates.

        Args:
            preview: ``(h, w, 3)`` uint8 sRGB-encoded RGB preview.

        Returns:
            Faces sorted by decreasing area; an empty list when none are found.
        """
        ...


class MeteringStrategy(Protocol):
    """Computes the exposure correction for one image."""

    name: ClassVar[str]

    def measure(self, ctx: ImageContext) -> ExposureMeasurement:
        """Measure ``ctx.image`` (scene-linear) and return the correction in EV."""
        ...


class Retoucher(Protocol):
    """Skin retouch restricted to a skin mask. ``none`` is the Null Object.

    Must not touch eyes, eyebrows, lips or teeth, must not move pixels
    geometrically, and must not use generative models.
    """

    name: ClassVar[str]

    def retouch(self, image: NDArray[np.float32], faces: Sequence[FaceRegion]) -> RetouchResult:
        """Return the retouched scene-linear image and the parameters applied."""
        ...


class Exporter(Protocol):
    """Encodes the final image to a file.

    Exporters are the only place where the transfer function (gamma) is
    applied and values are clipped.
    """

    name: ClassVar[str]

    def export(self, ctx: ImageContext, destination: Path) -> Path:
        """Write ``ctx.image`` and return the path written.

        Args:
            ctx: Final context; ``ctx.image`` is scene-linear working space.
            destination: Output path *without* suffix; the exporter appends its own.

        Raises:
            OutputExistsError: The file exists and overwriting is disabled.
        """
        ...


class QAProvider(Protocol):
    """Vision-LLM review of a downscaled JPEG preview. Flags only, never edits."""

    name: ClassVar[str]

    def review(self, preview_jpeg: bytes) -> QAResult:
        """Review a JPEG preview (long edge <= 1024 px) and return flags.

        Implementations must never raise for API or network problems; they
        return ``QAResult(skipped=True)`` and log a warning instead.
        """
        ...


@dataclass(frozen=True)
class ImageOutcome:
    """What the batch keeps of one processed image (no full-resolution pixels).

    Attributes:
        source_path: Input RAW file.
        metadata: Final metadata (EXIF plus every stage record).
        outputs: Files written for this image.
        thumbnails: Display-ready sRGB uint8 thumbnails (see ``REPORT_THUMBNAILS``).
        error: Error message when the image failed, else ``None``.
        error_type: Exception class name when the image failed.
    """

    source_path: Path
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    outputs: tuple[Path, ...] = ()
    thumbnails: Mapping[str, NDArray[np.uint8]] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` when the image was processed without error."""
        return self.error is None


class BatchReporter(Protocol):
    """Aggregates outcomes into batch-level reports (CSV, contact sheet)."""

    name: ClassVar[str]

    def add(self, outcome: ImageOutcome) -> None:
        """Record one image outcome (successful or failed)."""
        ...

    def finalize(self, output_dir: Path) -> list[Path]:
        """Write the report files into ``output_dir`` and return their paths."""
        ...


class PipelineObserver(Protocol):
    """Receives progress and timing events. Stages never print."""

    def on_batch_start(self, total: int) -> None:
        """Called once before the first image."""
        ...

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        """Called after each stage of each image with its wall time in seconds."""
        ...

    def on_image_end(self, outcome: ImageOutcome) -> None:
        """Called once per image, successful or failed."""
        ...

    def on_batch_end(self, outcomes: Sequence[ImageOutcome]) -> None:
        """Called once after the last image."""
        ...
