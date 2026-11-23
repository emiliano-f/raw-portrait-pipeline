"""Report stage: prepares the display-ready thumbnails the batch reporters use."""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import BEFORE_IMAGE, DEFECT_CROPS, EXPOSURE, REPORT_THUMBNAILS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import StageError
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import Scalar
from raw_pipeline.exporters.encoding import encode_srgb_uint8, srgb_thumbnail


@stage_registry.register("report")
class ReportStage:
    """Builds :data:`~raw_pipeline.core.artifacts.REPORT_THUMBNAILS` (sRGB uint8).

    * ``before``: the uncorrected decode (``BEFORE_IMAGE``), as shot with camera WB.
    * ``after``: the final image.
    * ``defect_before`` / ``defect_after``: the 100 % defect crops (no resize),
      scaled by the image's exposure correction so they look like the final
      output. Present only when the decode stage produced ``DEFECT_CROPS``.

    Pixels are not changed; the stage returns the same image.

    Args:
        thumbnail_long_edge_px: Long edge of the before/after thumbnails, in pixels.
    """

    name: ClassVar[str] = "report"

    def __init__(self, thumbnail_long_edge_px: int) -> None:
        self.thumbnail_long_edge_px = thumbnail_long_edge_px

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``report`` config section."""
        return cls(app_config(env).report.thumbnail_long_edge_px)

    def process(self, ctx: ImageContext) -> ImageContext:
        """Encode the thumbnails and record which ones were produced."""
        if not ctx.is_decoded:
            msg = "report needs a decoded image"
            raise StageError(msg)
        thumbs: dict[str, NDArray[np.uint8]] = {}
        before = ctx.artifact(BEFORE_IMAGE)
        if before is not None:
            thumbs["before"] = srgb_thumbnail(before, self.thumbnail_long_edge_px)
        thumbs["after"] = srgb_thumbnail(ctx.image, self.thumbnail_long_edge_px)
        exposure = ctx.artifact(EXPOSURE)
        crop_ev = exposure.correction_ev if exposure is not None else 0.0
        crops = ctx.artifact(DEFECT_CROPS)
        if crops is not None:
            gain = np.float32(2.0**crop_ev)
            thumbs["defect_before"] = encode_srgb_uint8(crops.before * gain)
            thumbs["defect_after"] = encode_srgb_uint8(crops.after * gain)
        record: dict[str, Scalar] = {
            "thumbnails": ",".join(thumbs),
            "thumbnail_long_edge_px": self.thumbnail_long_edge_px,
            "defect_crop_ev": crop_ev if crops is not None else None,
        }
        return ctx.with_artifact(REPORT_THUMBNAILS, MappingProxyType(thumbs)).with_record(self.name, record)
