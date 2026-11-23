from __future__ import annotations

from pathlib import Path

from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.raw import ColumnDefect, PixelDefect
from raw_pipeline.runtime.defect_detection import DefectDetectionService
from tests.unit.runtime.fakes import FakeDecoder, FakeDetector, write_inputs


def test_streams_frames_and_skips_unreadable(tmp_path: Path) -> None:
    sources = write_inputs(tmp_path, {"A.CR2": b"a", "BAD.CR2": b"BAD", "C.CR2": b"c"})
    detector = FakeDetector(DefectDetectionConfig())

    result = DefectDetectionService(FakeDecoder(), detector).detect(sources)

    assert detector.frames == 2
    assert result.frames_used == (tmp_path / "A.CR2", tmp_path / "C.CR2")
    assert result.frames_failed == (tmp_path / "BAD.CR2",)
    assert tuple(result.defect_map) == (ColumnDefect(index=21, width=2), PixelDefect(x=7, y=9))
