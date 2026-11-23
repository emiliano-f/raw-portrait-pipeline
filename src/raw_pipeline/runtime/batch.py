"""Batch execution: parallel worker processes, ordered outcomes, batch reporters."""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Final

from raw_pipeline.core.errors import PipelineError
from raw_pipeline.core.protocols import BatchReporter, ImageOutcome, PipelineObserver
from raw_pipeline.core.values import ImageMetadata, Scalar
from raw_pipeline.runtime.observers import NullObserver
from raw_pipeline.runtime.pipeline import ImageRun, Pipeline, StageTiming

logger = logging.getLogger(__name__)


START_METHOD: Final = "spawn"
"""Worker start method: ``spawn`` behaves the same on Linux and macOS and is safe
with libraries that start threads (MediaPipe, OpenCV), unlike ``fork``."""


def default_workers() -> int:
    """CPU count - 1, at least 1."""
    return max(1, (os.cpu_count() or 1) - 1)


@dataclass(frozen=True)
class BatchResult:
    """Everything a batch run produced.

    Attributes:
        outcomes: One outcome per input, in input order.
        report_paths: Files written by the batch reporters.
        reporter_errors: ``reporter name -> error message`` for reporters that failed.
    """

    outcomes: tuple[ImageOutcome, ...]
    report_paths: tuple[Path, ...] = ()
    reporter_errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def failed(self) -> tuple[ImageOutcome, ...]:
        """Outcomes of the images that failed."""
        return tuple(o for o in self.outcomes if not o.ok)

    @property
    def ok(self) -> bool:
        """``True`` when every image and every reporter succeeded."""
        return not self.failed and not self.reporter_errors

    @property
    def exit_code(self) -> int:
        """Process exit code: 0 when everything succeeded, 1 otherwise."""
        return 0 if self.ok else 1


class BatchRunner:
    """Runs a :class:`Pipeline` over many images, one image per task.

    With ``workers > 1`` images are processed in a pool of worker processes,
    each holding its own copy of the pipeline; otherwise they run in this
    process. Either way outcomes are returned, observed and fed to the
    reporters in input order, so parallel and serial runs produce the same
    reports. Only outcomes (metadata and small thumbnails) travel back to
    this process, never full-resolution pixels.

    Args:
        pipeline: The per-image pipeline (must be picklable when ``workers > 1``).
        workers: Number of worker processes (``>= 1``).
        observer: Receives progress and per-stage timing events.
    """

    def __init__(self, pipeline: Pipeline, workers: int = 1, observer: PipelineObserver | None = None) -> None:
        if workers < 1:
            msg = f"workers must be >= 1, got {workers}"
            raise ValueError(msg)
        self._pipeline = pipeline
        self._workers = workers
        self._observer: PipelineObserver = observer or NullObserver()

    def run(self, sources: Sequence[Path], reporters: Sequence[BatchReporter], output_dir: Path) -> BatchResult:
        """Process ``sources`` and write the batch reports into ``output_dir``."""
        self._observer.on_batch_start(len(sources))
        outcomes: list[ImageOutcome] = []
        errors: dict[str, str] = {}
        for run in self._runs(sources):
            for timing in run.timings:
                self._observer.on_stage_end(run.outcome.source_path, timing.stage, timing.seconds)
            self._observer.on_image_end(run.outcome)
            for reporter in reporters:
                if reporter.name not in errors:
                    _guarded(reporter, errors, partial(reporter.add, run.outcome))
            outcomes.append(run.outcome)
        report_paths: list[Path] = []
        for reporter in reporters:
            if reporter.name not in errors:
                report_paths.extend(_guarded(reporter, errors, partial(reporter.finalize, output_dir)) or ())
        self._observer.on_batch_end(outcomes)
        return BatchResult(tuple(outcomes), tuple(report_paths), errors)

    def _runs(self, sources: Sequence[Path]) -> Iterator[ImageRun]:
        workers = min(self._workers, len(sources))
        if workers <= 1:
            yield from (self._pipeline.run(path) for path in sources)
            return
        context = multiprocessing.get_context(START_METHOD)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(self._pipeline,)
        ) as pool:
            for transport in pool.map(_run_in_worker, sources):
                yield transport.to_run()


def _guarded[R](reporter: BatchReporter, errors: dict[str, str], call: Callable[[], R]) -> R | None:
    """Run ``call``; a failure is recorded in ``errors`` and disables the reporter for the rest of the batch."""
    try:
        return call()
    except (PipelineError, OSError) as exc:
        logger.error("reporter %s failed: %s", reporter.name, exc)
        errors[reporter.name] = str(exc) or type(exc).__name__
    except Exception as exc:
        logger.exception("unexpected error in reporter %s", reporter.name)
        errors[reporter.name] = str(exc) or type(exc).__name__
    return None


@dataclass(frozen=True)
class _TransportRun:
    """Picklable form of an :class:`ImageRun` (metadata records are read-only proxies)."""

    outcome: ImageOutcome
    records: dict[str, dict[str, Scalar]]
    timings: tuple[StageTiming, ...]

    @classmethod
    def from_run(cls, run: ImageRun) -> _TransportRun:
        metadata = run.outcome.metadata
        exif = metadata.exif
        plain_exif = replace(exif, gps=dict(exif.gps)) if exif.gps is not None else exif
        outcome = replace(
            run.outcome, metadata=ImageMetadata(exif=plain_exif, records={}), thumbnails=dict(run.outcome.thumbnails)
        )
        records = {stage: dict(params) for stage, params in metadata.records.items()}
        return cls(outcome, records, run.timings)

    def to_run(self) -> ImageRun:
        metadata = self.outcome.metadata
        for stage, params in self.records.items():
            metadata = metadata.with_record(stage, params)
        return ImageRun(replace(self.outcome, metadata=metadata), self.timings)


_worker_pipeline: Pipeline | None = None
"""Per-worker-process copy of the pipeline, set once by the pool initializer."""


def _init_worker(pipeline: Pipeline) -> None:
    global _worker_pipeline  # noqa: PLW0603 - process-local state of a pool worker
    _worker_pipeline = pipeline


def _run_in_worker(source: Path) -> _TransportRun:
    if _worker_pipeline is None:  # pragma: no cover - the initializer always runs first
        msg = "worker pipeline not initialised"
        raise RuntimeError(msg)
    return _TransportRun.from_run(_worker_pipeline.run(source))
