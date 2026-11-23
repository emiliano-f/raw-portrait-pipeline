"""Export stage: writes the final image with every configured exporter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import OUTPUT_PATHS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import StageError
from raw_pipeline.core.protocols import Exporter
from raw_pipeline.core.registries import exporter_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import Scalar
from raw_pipeline.exporters.encoding import output_size

OUTPUT_SEPARATOR = ";"
"""Separator of output file names in the ``export.files`` record."""


@stage_registry.register("export")
class ExportStage:
    """Runs the exporters in order and records the files written.

    Outputs are named ``<output_dir>/<input stem><filename_suffix><exporter suffix>``.
    Pixels are not changed; the stage returns the same image.

    Args:
        exporters: Exporters to run, in order.
        output_dir: Directory receiving the files.
        filename_suffix: Appended to the input file stem.
        params: Export settings recorded verbatim (quality, size limit, GPS, colour space).
        long_edge_px: Long-edge limit the exporters apply, used to record the output size.
    """

    name: ClassVar[str] = "export"

    def __init__(
        self,
        exporters: Sequence[Exporter],
        output_dir: Path,
        *,
        filename_suffix: str = "",
        long_edge_px: int | None = None,
        params: Mapping[str, Scalar] | None = None,
    ) -> None:
        self.exporters = tuple(exporters)
        self.output_dir = output_dir
        self.filename_suffix = filename_suffix
        self.long_edge_px = long_edge_px
        self.params = dict(params or {})

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build the exporters named in ``export.formats`` from the exporter registry."""
        cfg = app_config(env).export
        return cls(
            [exporter_registry.create(name, env) for name in cfg.formats],
            env.output_dir,
            filename_suffix=cfg.filename_suffix,
            long_edge_px=cfg.long_edge_px,
            params={
                "jpeg_quality": cfg.jpeg_quality,
                "long_edge_px": cfg.long_edge_px,
                "color_space": cfg.color_space,
                "keep_gps": cfg.keep_gps,
            },
        )

    def destination(self, source_path: Path) -> Path:
        """Output path without suffix for ``source_path``."""
        return self.output_dir / f"{source_path.stem}{self.filename_suffix}"

    def process(self, ctx: ImageContext) -> ImageContext:
        """Export ``ctx.image`` and set :data:`~raw_pipeline.core.artifacts.OUTPUT_PATHS`."""
        if not ctx.is_decoded:
            msg = "export needs a decoded image"
            raise StageError(msg)
        destination = self.destination(ctx.source_path)
        paths = tuple(exporter.export(ctx, destination) for exporter in self.exporters)
        height, width = ctx.image.shape[:2]
        out_width, out_height = output_size(width, height, self.long_edge_px)
        record: dict[str, Scalar] = {
            **self.params,
            "formats": ",".join(exporter.name for exporter in self.exporters),
            "files": OUTPUT_SEPARATOR.join(path.name for path in paths),
            "output_width": out_width,
            "output_height": out_height,
        }
        return ctx.with_artifact(OUTPUT_PATHS, paths).with_record(self.name, record)
