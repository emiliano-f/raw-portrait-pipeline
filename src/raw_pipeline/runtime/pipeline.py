"""The per-image pipeline: an ordered chain of stages (Chain of Responsibility)."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from raw_pipeline.core.artifacts import OUTPUT_PATHS, REPORT_THUMBNAILS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import ImageError
from raw_pipeline.core.protocols import ImageOutcome, Stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageTiming:
    """Wall time of one stage on one image.

    Attributes:
        stage: Stage name.
        seconds: Wall time in seconds.
    """

    stage: str
    seconds: float


@dataclass(frozen=True)
class ImageRun:
    """Result of running the pipeline on one image.

    Attributes:
        outcome: What the batch keeps of the image (no full-resolution pixels).
        timings: Per-stage timings, in execution order (stages run before a failure).
    """

    outcome: ImageOutcome
    timings: tuple[StageTiming, ...]


class Pipeline:
    """Runs one image through an ordered list of stages.

    The pipeline knows nothing about what the stages do. A failure is
    isolated to the image: it is logged, captured into the outcome (with the
    metadata recorded so far) and the stages after it are skipped.

    Args:
        stages: Stages in execution order. Must be picklable (the batch runner
            ships the pipeline to worker processes).
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = tuple(stages)

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Names of the stages, in execution order."""
        return tuple(stage.name for stage in self._stages)

    def run(self, source_path: Path) -> ImageRun:
        """Process ``source_path`` (never modified) and return its outcome and timings."""
        ctx = ImageContext.initial(source_path)
        timings: list[StageTiming] = []
        for stage in self._stages:
            started = time.perf_counter()
            try:
                ctx = stage.process(ctx)
            except ImageError as exc:
                logger.error("%s: stage %s failed: %s", source_path.name, stage.name, exc)
                return ImageRun(_failed(ctx, exc), tuple(timings))
            except Exception as exc:
                logger.exception("%s: unexpected error in stage %s", source_path.name, stage.name)
                return ImageRun(_failed(ctx, exc), tuple(timings))
            timings.append(StageTiming(stage.name, time.perf_counter() - started))
        thumbnails = ctx.artifact(REPORT_THUMBNAILS) or {}
        outcome = ImageOutcome(
            source_path=source_path,
            metadata=ctx.metadata,
            outputs=ctx.artifact(OUTPUT_PATHS) or (),
            thumbnails=dict(thumbnails),
        )
        return ImageRun(outcome, tuple(timings))


def _failed(ctx: ImageContext, exc: Exception) -> ImageOutcome:
    return ImageOutcome(
        source_path=ctx.source_path,
        metadata=ctx.metadata,
        outputs=ctx.artifact(OUTPUT_PATHS) or (),
        error=str(exc) or type(exc).__name__,
        error_type=type(exc).__name__,
    )
