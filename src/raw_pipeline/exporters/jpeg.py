"""8-bit sRGB JPEG exporter."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, ClassVar, Final, Self

from PIL import Image

from raw_pipeline.config import app_config
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.registries import exporter_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.exporters.encoding import encode_srgb_uint8, resize_long_edge, srgb_icc_profile
from raw_pipeline.exporters.exif import jpeg_exif_bytes
from raw_pipeline.exporters.output import write_output

JPEG_SUFFIX: Final = ".jpg"
CHROMA_SUBSAMPLING_444: Final = 0
"""Pillow ``subsampling`` value for 4:4:4 (no chroma subsampling): keeps skin and hair colour edges clean."""


@exporter_registry.register("jpeg")
class JpegExporter:
    """Writes an sRGB JPEG with embedded ICC profile and the source's descriptive EXIF.

    Args:
        quality: JPEG quality, 1-100.
        long_edge_px: Downscale so the long edge is at most this many pixels; ``None`` keeps full size.
        keep_gps: Copy GPS tags from the source EXIF.
        overwrite: Replace an existing file instead of raising ``OutputExistsError``.
    """

    name: ClassVar[str] = "jpeg"

    def __init__(self, *, quality: int, long_edge_px: int | None, keep_gps: bool, overwrite: bool) -> None:
        self.quality = quality
        self.long_edge_px = long_edge_px
        self.keep_gps = keep_gps
        self.overwrite = overwrite

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``export`` config section and the run's overwrite flag."""
        cfg = app_config(env).export
        return cls(
            quality=cfg.jpeg_quality, long_edge_px=cfg.long_edge_px, keep_gps=cfg.keep_gps, overwrite=env.overwrite
        )

    def export(self, ctx: ImageContext, destination: Path) -> Path:
        """Encode ``ctx.image`` (scene-linear) and write ``destination`` + ``.jpg``."""
        pixels = encode_srgb_uint8(resize_long_edge(ctx.image, self.long_edge_px))
        exif = jpeg_exif_bytes(ctx.metadata.exif, keep_gps=self.keep_gps)

        def write(handle: BinaryIO) -> None:
            Image.fromarray(pixels, mode="RGB").save(
                handle,
                format="JPEG",
                quality=self.quality,
                subsampling=CHROMA_SUBSAMPLING_444,
                optimize=False,
                icc_profile=srgb_icc_profile(),
                exif=exif,
            )

        return write_output(destination.with_name(destination.name + JPEG_SUFFIX), write, overwrite=self.overwrite)
