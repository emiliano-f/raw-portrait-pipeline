"""Pipeline observers: logging, CLI progress and fan-out (Observer pattern)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from raw_pipeline.core.protocols import ImageOutcome, PipelineObserver

logger = logging.getLogger(__name__)


class NullObserver:
    """Ignores every event (Null Object)."""

    def on_batch_start(self, total: int) -> None:
        """Ignore the event."""

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        """Ignore the event."""

    def on_image_end(self, outcome: ImageOutcome) -> None:
        """Ignore the event."""

    def on_batch_end(self, outcomes: Sequence[ImageOutcome]) -> None:
        """Ignore the event."""


class LoggingObserver:
    """Logs per-image results at INFO and per-stage timings at DEBUG."""

    def on_batch_start(self, total: int) -> None:
        """Log the batch size."""
        logger.info("processing %d image(s)", total)

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        """Log the wall time of one stage, in seconds."""
        logger.debug("%s: stage %s took %.3f s", source_path.name, stage, seconds)

    def on_image_end(self, outcome: ImageOutcome) -> None:
        """Log the result of one image."""
        if outcome.ok:
            logger.info("%s: done (%d output file(s))", outcome.source_path.name, len(outcome.outputs))
        else:
            logger.info("%s: failed (%s)", outcome.source_path.name, outcome.error_type)

    def on_batch_end(self, outcomes: Sequence[ImageOutcome]) -> None:
        """Log the batch summary."""
        failed = sum(1 for o in outcomes if not o.ok)
        logger.info("batch finished: %d ok, %d failed", len(outcomes) - failed, failed)


class ProgressObserver:
    """Writes one progress line per image and a final summary to a text stream.

    Args:
        stream: Where to write (the CLI passes ``sys.stderr``).
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._total = 0
        self._done = 0

    def on_batch_start(self, total: int) -> None:
        """Remember the batch size."""
        self._total = total
        self._done = 0

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        """Per-stage timing is not shown as progress."""

    def on_image_end(self, outcome: ImageOutcome) -> None:
        """Write ``[i/N] name: ok|FAILED``."""
        self._done += 1
        status = "ok" if outcome.ok else f"FAILED: {outcome.error}"
        self._stream.write(f"[{self._done}/{self._total}] {outcome.source_path.name}: {status}\n")
        self._stream.flush()

    def on_batch_end(self, outcomes: Sequence[ImageOutcome]) -> None:
        """Write the summary line."""
        failed = sum(1 for o in outcomes if not o.ok)
        self._stream.write(f"{len(outcomes) - failed} ok, {failed} failed\n")
        self._stream.flush()


class CompositeObserver:
    """Forwards every event to several observers, in order.

    Args:
        observers: Observers to notify.
    """

    def __init__(self, observers: Sequence[PipelineObserver]) -> None:
        self._observers = tuple(observers)

    def on_batch_start(self, total: int) -> None:
        """Forward the event."""
        for observer in self._observers:
            observer.on_batch_start(total)

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        """Forward the event."""
        for observer in self._observers:
            observer.on_stage_end(source_path, stage, seconds)

    def on_image_end(self, outcome: ImageOutcome) -> None:
        """Forward the event."""
        for observer in self._observers:
            observer.on_image_end(outcome)

    def on_batch_end(self, outcomes: Sequence[ImageOutcome]) -> None:
        """Forward the event."""
        for observer in self._observers:
            observer.on_batch_end(outcomes)
