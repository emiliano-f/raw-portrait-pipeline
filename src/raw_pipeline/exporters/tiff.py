"""16-bit sRGB TIFF exporter."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, ClassVar, Final, Self

import tifffile

from raw_pipeline.config import app_config
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.registries import exporter_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.exporters.encoding import encode_srgb_uint16, resize_long_edge, srgb_icc_profile
from raw_pipeline.exporters.exif import ExifField, exif_fields
from raw_pipeline.exporters.output import write_output

logger = logging.getLogger(__name__)

TIFF_SUFFIX: Final = ".tif"
SOFTWARE: Final = "raw-pipeline"
TIFF_ASCII: Final = 2
TIFF_SHORT: Final = 3
TIFF_RATIONAL: Final = 5

TiffTag = tuple[int, int, int, object, bool]
"""``tifffile`` extra tag: ``(code, dtype, count, value, writeonce)``."""


def _tiff_tag(field: ExifField) -> TiffTag:
    value = field.value
    if isinstance(value, str):
        return (field.tag, TIFF_ASCII, 0, value, True)
    if isinstance(value, Fraction):
        return (field.tag, TIFF_RATIONAL, 1, (value.numerator, value.denominator), True)
    return (field.tag, TIFF_SHORT, 1, value, True)


@exporter_registry.register("tiff")
class TiffExporter:
    """Writes a 16-bit-per-channel sRGB TIFF (deflate) with embedded ICC profile.

    Descriptive EXIF tags are stored in IFD0 (TIFF/EP style), which common
    readers (exiftool, Pillow, Lightroom) understand. GPS is never written to
    TIFF files: a GPS sub-IFD is not supported by the encoder.

    Args:
        long_edge_px: Downscale so the long edge is at most this many pixels; ``None`` keeps full size.
        overwrite: Replace an existing file instead of raising ``OutputExistsError``.
    """

    name: ClassVar[str] = "tiff"

    def __init__(self, *, long_edge_px: int | None, overwrite: bool) -> None:
        self.long_edge_px = long_edge_px
        self.overwrite = overwrite

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``export`` config section and the run's overwrite flag."""
        cfg = app_config(env).export
        if cfg.keep_gps:
            logger.warning(
                "export.keep_gps is set: GPS is kept in JPEG outputs only; TIFF outputs never contain "
                "GPS tags (tifffile cannot write a GPS sub-IFD)"
            )
        return cls(long_edge_px=cfg.long_edge_px, overwrite=env.overwrite)

    def export(self, ctx: ImageContext, destination: Path) -> Path:
        """Encode ``ctx.image`` (scene-linear) and write ``destination`` + ``.tif``."""
        pixels = encode_srgb_uint16(resize_long_edge(ctx.image, self.long_edge_px))
        tags = [_tiff_tag(field) for field in exif_fields(ctx.metadata.exif)]

        def write(handle: BinaryIO) -> None:
            tifffile.imwrite(
                handle,
                pixels,
                photometric="rgb",
                compression="zlib",
                predictor=True,
                iccprofile=srgb_icc_profile(),
                software=SOFTWARE,
                metadata=None,
                extratags=tags,
            )

        return write_output(destination.with_name(destination.name + TIFF_SUFFIX), write, overwrite=self.overwrite)
