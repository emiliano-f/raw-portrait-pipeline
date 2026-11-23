"""Automatic sensor defect detection over dark frames or a whole batch."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from raw_pipeline.core.errors import ImageError
from raw_pipeline.core.protocols import DefectDetector, RawDecoder
from raw_pipeline.core.raw import CFAFrame, DefectMap, DetectedDefect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefectDetectionResult:
    """What ``detect-defects`` found.

    Attributes:
        detected: Defects with their scores, in detector order.
        frames_used: Files whose CFA data was analysed.
        frames_failed: Files that could not be read (logged and skipped).
    """

    detected: tuple[DetectedDefect, ...]
    frames_used: tuple[Path, ...]
    frames_failed: tuple[Path, ...]

    @property
    def defect_map(self) -> DefectMap:
        """The detected defects as a map, ready for ``defects.map``."""
        return DefectMap(tuple(d.defect for d in self.detected))


class DefectDetectionService:
    """Reads CFA frames through the RAW decoder and runs a defect detector on them.

    Frames are streamed one at a time (read-only), so the batch is never held
    in memory. Coordinates in the result are raw sensor coordinates (ADR 0002).

    Args:
        decoder: Opens RAW files read-only.
        detector: Finds consistent outliers across the frames.
    """

    def __init__(self, decoder: RawDecoder, detector: DefectDetector) -> None:
        self._decoder = decoder
        self._detector = detector

    def detect(self, sources: Sequence[Path]) -> DefectDetectionResult:
        """Analyse ``sources`` (one dark frame, or a batch) and return the defects found."""
        used: list[Path] = []
        failed: list[Path] = []
        detected = self._detector.detect(self._frames(sources, used, failed))
        return DefectDetectionResult(tuple(detected), tuple(used), tuple(failed))

    def _frames(self, sources: Sequence[Path], used: list[Path], failed: list[Path]) -> Iterator[CFAFrame]:
        for path in sources:
            try:
                with self._decoder.open(path) as handle:
                    frame = handle.read_cfa()
            except ImageError as exc:
                logger.warning("%s: skipped for defect detection: %s", path.name, exc)
                failed.append(path)
                continue
            used.append(path)
            yield frame
