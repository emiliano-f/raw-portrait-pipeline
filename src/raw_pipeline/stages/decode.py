"""The ``decode`` stage: read CFA → raw-level defect correction → demosaic (ADR 0002)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.config.defects import to_defect_map
from raw_pipeline.core.artifacts import BEFORE_IMAGE, DEFECT_CROPS, DefectCrops
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import DefectCorrectionError
from raw_pipeline.core.geometry import PreviewGeometry
from raw_pipeline.core.protocols import DefectCorrector, MetadataReader, RawDecoder, RawHandle
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, Defect, DefectCorrectionResult, DefectMap, PixelDefect
from raw_pipeline.core.registries import defect_corrector_registry, raw_decoder_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.io.exif import ExifReader

NULL_CORRECTOR: Final = "none"
REPORT_STAGE: Final = "report"
DEFECTS_RECORD: Final = "defects"


def defect_center(defect: Defect, height: int, width: int) -> tuple[int, int]:
    """Return the ``(x, y)`` sensor photosite at the centre of ``defect``, clipped to the frame."""
    if isinstance(defect, PixelDefect):
        x, y = defect.x, defect.y
    else:
        across = defect.index + (defect.width - 1) // 2
        length = height if isinstance(defect, ColumnDefect) else width
        lo = max(defect.start or 0, 0)
        hi = min(defect.end if defect.end is not None else length, length)
        along = (lo + hi - 1) // 2
        x, y = (across, along) if isinstance(defect, ColumnDefect) else (along, across)
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def crop_window(center: int, size: int, length: int) -> slice:
    """Return a ``size``-long window around ``center`` that stays inside ``[0, length)``."""
    size = min(size, length)
    start = min(max(center - size // 2, 0), length - size)
    return slice(start, start + size)


def downscale(image: NDArray[np.float32], long_edge_px: int) -> NDArray[np.float32]:
    """Area-downscale a scene-linear image so its long edge is at most ``long_edge_px`` (never upscales)."""
    height, width = image.shape[:2]
    geometry = PreviewGeometry.for_long_edge(width, height, long_edge_px)
    if (geometry.preview_width, geometry.preview_height) == (width, height):
        return image.copy()
    size = (geometry.preview_width, geometry.preview_height)
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.float32)


@dataclass(frozen=True)
class _Developed:
    after: NDArray[np.float32]
    before: NDArray[np.float32] | None
    crops: DefectCrops | None
    params: dict[str, str | int | float | bool | None]
    correction: DefectCorrectionResult


@stage_registry.register("decode")
class DecodeStage:
    """Decodes a RAW file into the scene-linear working image.

    Order: :meth:`RawHandle.read_cfa` → :class:`DefectCorrector` →
    :meth:`RawHandle.develop`, so defects never reach the demosaic step. For the
    report it also keeps an uncorrected "before" preview (a second development
    only when the corrector actually replaced photosites) and 100 % defect
    crops of the first defect.

    Args:
        decoder: RAW decoder adapter.
        corrector: Raw-level defect corrector (``none`` is the Null Object).
        defect_map: Known defects in raw sensor coordinates.
        metadata_reader: Reads camera EXIF.
        before_preview_long_edge_px: Long edge of ``BEFORE_IMAGE`` (px).
        defect_crop_px: Edge of each 100 % defect crop (px).
        produce_report_artifacts: Build ``BEFORE_IMAGE`` and ``DEFECT_CROPS``
            (true when the ``report`` stage is configured).
    """

    name: ClassVar[str] = "decode"

    def __init__(
        self,
        decoder: RawDecoder,
        corrector: DefectCorrector,
        defect_map: DefectMap,
        metadata_reader: MetadataReader,
        *,
        before_preview_long_edge_px: int = 1024,
        defect_crop_px: int = 192,
        produce_report_artifacts: bool = True,
    ) -> None:
        self._decoder = decoder
        self._corrector = corrector
        self._defect_map = defect_map
        self._metadata_reader = metadata_reader
        self._before_long_edge_px = before_preview_long_edge_px
        self._crop_px = defect_crop_px
        self._report = produce_report_artifacts

    @classmethod
    def from_env(cls, env: BuildEnv) -> DecodeStage:
        """Build from the ``decode``, ``defects``, ``report`` and ``pipeline`` config sections."""
        cfg = app_config(env)
        return cls(
            decoder=raw_decoder_registry.create(cfg.decode.decoder, env),
            corrector=defect_corrector_registry.create(cfg.defects.method, env),
            defect_map=to_defect_map(cfg.defects.map),
            metadata_reader=ExifReader(),
            before_preview_long_edge_px=cfg.decode.before_preview_long_edge_px,
            defect_crop_px=cfg.report.defect_crop_px,
            produce_report_artifacts=REPORT_STAGE in cfg.pipeline.stages,
        )

    @property
    def correction_enabled(self) -> bool:
        """``True`` when a real corrector is configured and the defect map is not empty."""
        return self._corrector.name != NULL_CORRECTOR and len(self._defect_map) > 0

    def process(self, ctx: ImageContext) -> ImageContext:
        """Decode ``ctx.source_path`` (read-only) and return the decoded context."""
        with self._decoder.open(ctx.source_path) as handle:
            developed = self._develop(handle, ctx)
        exif = self._metadata_reader.read(ctx.source_path)
        out = ctx.with_metadata(ctx.metadata.with_exif(exif)).with_image(developed.after)
        out = out.with_record(self.name, {**developed.params, "before_preview": developed.before is not None})
        out = out.with_record(
            DEFECTS_RECORD,
            {
                "method": self._corrector.name,
                "corrected_pixel_count": developed.correction.corrected_pixel_count,
                "count": len(self._defect_map),
            },
        )
        if developed.before is not None:
            out = out.with_artifact(BEFORE_IMAGE, developed.before)
        if developed.crops is not None:
            out = out.with_artifact(DEFECT_CROPS, developed.crops)
        return out

    def _develop(self, handle: RawHandle, ctx: ImageContext) -> _Developed:
        raw = handle.read_cfa()
        correction = self._correct(raw, ctx)
        developed = handle.develop(correction.frame)
        after = developed.image
        if not self._report:
            return _Developed(after, None, None, developed.params, correction)
        before_full = handle.develop(raw).image if correction.corrected_pixel_count > 0 else after
        crops = self._crops(handle, raw, before_full, after) if self.correction_enabled else None
        before = downscale(before_full, self._before_long_edge_px)
        return _Developed(after, before, crops, developed.params, correction)

    def _correct(self, raw: CFAFrame, ctx: ImageContext) -> DefectCorrectionResult:
        if len(self._defect_map) == 0:
            return self._corrector.correct(raw, self._defect_map)
        try:
            self._defect_map.mask(raw.height, raw.width)
            return self._corrector.correct(raw, self._defect_map)
        except ValueError as exc:
            msg = f"{ctx.source_path.name}: defect map does not fit the sensor: {exc}"
            raise DefectCorrectionError(msg) from exc

    def _crops(
        self, handle: RawHandle, raw: CFAFrame, before: NDArray[np.float32], after: NDArray[np.float32]
    ) -> DefectCrops:
        sensor_xy = defect_center(self._defect_map.defects[0], raw.height, raw.width)
        image_xy = handle.sensor_to_image(*sensor_xy)
        rows = crop_window(image_xy[1], self._crop_px, after.shape[0])
        cols = crop_window(image_xy[0], self._crop_px, after.shape[1])
        return DefectCrops(
            before=before[rows, cols].copy(),
            after=after[rows, cols].copy(),
            sensor_xy=sensor_xy,
            image_xy=image_xy,
        )
