"""Well-known artifact keys shared between stages.

Producers and consumers agree on these keys and value types; everything else
in :attr:`ImageContext.artifacts` is private to the stage that wrote it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.context import ArtifactKey
from raw_pipeline.core.values import ExposureMeasurement, FaceRegion, QAResult


@dataclass(frozen=True)
class DefectCrops:
    """100 % crops around the first defect, for the contact sheet.

    Both crops cover the same output-image pixels (after the decoder's
    rotation/flip), developed with the same settings.

    Attributes:
        before: ``(h, w, 3)`` float32 scene-linear crop developed *without* correction.
        after: ``(h, w, 3)`` float32 scene-linear crop developed *with* correction.
        sensor_xy: Centre of the crop in raw sensor coordinates.
        image_xy: Centre of the crop in developed-image coordinates.
    """

    before: NDArray[np.float32]
    after: NDArray[np.float32]
    sensor_xy: tuple[int, int]
    image_xy: tuple[int, int]


FACES: Final = ArtifactKey[tuple[FaceRegion, ...]]("faces")
"""Faces in full-resolution image coordinates (written by the meter stage)."""

EXPOSURE: Final = ArtifactKey[ExposureMeasurement]("exposure")
"""Exposure measurement used by the tone stage (written by the meter stage)."""

BEFORE_IMAGE: Final = ArtifactKey[NDArray[np.float32]]("before_image")
"""Downscaled scene-linear image as decoded with camera WB and no defect
correction, for the contact sheet "before" column (written by the decode stage)."""

DEFECT_CROPS: Final = ArtifactKey[DefectCrops]("defect_crops")
"""Before/after 100 % defect crops (decode stage, only when correction is on)."""

OUTPUT_PATHS: Final = ArtifactKey[tuple[Path, ...]]("output_paths")
"""Files written for this image (export stage)."""

REPORT_THUMBNAILS: Final = ArtifactKey[Mapping[str, NDArray[np.uint8]]]("report_thumbnails")
"""Display-ready sRGB uint8 thumbnails for the contact sheet, keyed by
``"before"``, ``"after"``, ``"defect_before"``, ``"defect_after"`` (report stage)."""

QA_RESULT: Final = ArtifactKey[QAResult]("qa_result")
"""QA flags for this image (QA stage)."""
