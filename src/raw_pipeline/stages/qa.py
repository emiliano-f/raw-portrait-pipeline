"""The ``qa`` stage: optional vision-LLM review of a downscaled preview. Never changes pixels."""

from __future__ import annotations

import logging
from typing import ClassVar

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import QA_RESULT
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import StageError
from raw_pipeline.core.protocols import QAProvider
from raw_pipeline.core.registries import qa_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import QAResult, Scalar
from raw_pipeline.qa.preview import MAX_PREVIEW_LONG_EDGE_PX, encode_preview
from raw_pipeline.qa.response import KNOWN_FLAGS
from raw_pipeline.qa.target import require_endpoint_and_model

logger = logging.getLogger(__name__)


@stage_registry.register("qa")
class QAStage:
    """Encodes an EXIF-free JPEG preview, asks the QA provider for flags and records them.

    The preview is the only data handed to the provider: long edge at most
    ``preview_long_edge_px`` and never above 1024 px, sRGB-encoded from the
    scene-linear image, no metadata. The stage never fails an image because of
    QA: provider exceptions become a skipped result with a warning.

    Args:
        provider: QA provider (``NullQAProvider`` when QA is disabled).
        preview_long_edge_px: Requested preview long edge in pixels (capped at 1024).
        preview_jpeg_quality: JPEG quality of the preview (1-100).
    """

    name: ClassVar[str] = "qa"

    def __init__(self, provider: QAProvider, preview_long_edge_px: int = 1024, preview_jpeg_quality: int = 85) -> None:
        self.provider = provider
        self.preview_long_edge_px = min(preview_long_edge_px, MAX_PREVIEW_LONG_EDGE_PX)
        self.preview_jpeg_quality = preview_jpeg_quality

    @classmethod
    def from_env(cls, env: BuildEnv) -> QAStage:
        """Build from ``qa.*``, resolving ``qa.provider`` in the QA registry.

        Raises:
            ConfigError: If ``qa.provider`` is not ``none`` and ``qa.endpoint`` or ``qa.model`` is null.
        """
        cfg = app_config(env).qa
        if cfg.provider != "none":
            require_endpoint_and_model(cfg)
        return cls(
            qa_registry.create(cfg.provider, env),
            preview_long_edge_px=cfg.preview_long_edge_px,
            preview_jpeg_quality=cfg.preview_jpeg_quality,
        )

    def process(self, ctx: ImageContext) -> ImageContext:
        """Review ``ctx.image`` and return the context with ``QA_RESULT`` and a ``qa`` record.

        Raises:
            StageError: If the image has not been decoded yet.
        """
        if not ctx.is_decoded:
            msg = f"{ctx.source_path.name}: qa stage needs a decoded image"
            raise StageError(msg)
        preview = encode_preview(ctx.image, self.preview_long_edge_px, self.preview_jpeg_quality)
        result = self._review(ctx, preview)
        if result.skipped:
            logger.debug("%s: QA skipped (%s)", ctx.source_path.name, result.notes)
        elif result.flags:
            logger.info("%s: QA flags %s", ctx.source_path.name, ", ".join(sorted(result.flags)))
        return ctx.with_artifact(QA_RESULT, result).with_record(self.name, qa_record(result))

    def _review(self, ctx: ImageContext, preview: bytes) -> QAResult:
        try:
            return self.provider.review(preview)
        except Exception as exc:  # QA must never fail an image, whatever a plugin raises
            logger.warning("%s: QA provider %s failed: %s", ctx.source_path.name, self.provider.name, exc)
            return QAResult(provider=self.provider.name, skipped=True, notes=f"provider error: {type(exc).__name__}")


def qa_record(result: QAResult) -> dict[str, Scalar]:
    """Flatten a :class:`QAResult` into CSV-ready parameters.

    One boolean column per known flag (``None`` when QA was skipped, so "not
    checked" is distinguishable from "no issue"), plus every flag joined by ``;``.
    """
    record: dict[str, Scalar] = {
        "provider": result.provider,
        "model": result.model,
        "skipped": result.skipped,
        "flags": ";".join(sorted(result.flags)),
        "notes": result.notes,
    }
    for flag in sorted(KNOWN_FLAGS):
        record[flag] = None if result.skipped else flag in result.flags
    return record
