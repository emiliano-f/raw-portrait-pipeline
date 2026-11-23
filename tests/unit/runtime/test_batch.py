from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.protocols import ImageOutcome
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.batch import BatchRunner, default_workers
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.factory import PipelineFactory
from tests.unit.runtime.fakes import STAGES, FakeSummary, write_inputs

CONTENTS = {"C.CR2": b"c", "A.CR2": b"a", "BAD.CR2": b"BAD", "D.CR2": b"CRASH", "B.CR2": b"b"}


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def on_batch_start(self, total: int) -> None:
        self.events.append(("start", total))

    def on_stage_end(self, source_path: Path, stage: str, seconds: float) -> None:
        assert seconds >= 0
        self.events.append(("stage", source_path.name, stage))

    def on_image_end(self, outcome: ImageOutcome) -> None:
        self.events.append(("image", outcome.source_path.name, outcome.ok))

    def on_batch_end(self, outcomes: object) -> None:
        del outcomes
        self.events.append(("end",))


def _run(registries: ComponentRegistries, config: AppConfig, tmp_path: Path, name: str, workers: int):
    sources = write_inputs(tmp_path / "in", CONTENTS)
    out = tmp_path / name
    out.mkdir()
    env = BuildEnv(config=config, output_dir=out)
    factory = PipelineFactory(registries)
    observer = RecordingObserver()
    reporters = factory.build_reporters(env)
    result = BatchRunner(factory.build_pipeline(env), workers, observer).run(sources, reporters, out)
    return result, observer, out


def test_default_workers_leaves_one_cpu_free() -> None:
    assert default_workers() >= 1


def test_workers_must_be_positive(registries: ComponentRegistries, config: AppConfig, tmp_path: Path) -> None:
    pipeline = PipelineFactory(registries).build_pipeline(BuildEnv(config=config, output_dir=tmp_path))
    with pytest.raises(ValueError, match="workers"):
        BatchRunner(pipeline, 0)


def test_failures_are_isolated_and_reported(registries: ComponentRegistries, config: AppConfig, tmp_path: Path) -> None:
    result, observer, out = _run(registries, config, tmp_path, "out", workers=1)

    assert [o.source_path.name for o in result.outcomes] == ["A.CR2", "B.CR2", "BAD.CR2", "C.CR2", "D.CR2"]
    assert [o.error_type for o in result.failed] == ["DecodeError", "ValueError"]
    assert not result.ok
    assert result.exit_code == 1
    assert sorted(p.name for p in out.iterdir()) == ["A.txt", "B.txt", "C.txt", "summary.txt"]
    assert result.report_paths == (out / "summary.txt",)
    assert observer.events[0] == ("start", 5)
    assert observer.events[-1] == ("end",)
    assert observer.events[1:5] == [*(("stage", "A.CR2", s) for s in STAGES), ("image", "A.CR2", True)]
    assert ("image", "BAD.CR2", False) in observer.events


def test_parallel_matches_serial(registries: ComponentRegistries, config: AppConfig, tmp_path: Path) -> None:
    serial, serial_events, serial_out = _run(registries, config, tmp_path, "serial", workers=1)
    parallel, parallel_events, parallel_out = _run(registries, config, tmp_path, "parallel", workers=3)

    def summary(outcomes: tuple[ImageOutcome, ...]) -> list[object]:
        return [(o.source_path, o.ok, o.error_type, o.metadata.flatten(), sorted(o.thumbnails)) for o in outcomes]

    assert summary(parallel.outcomes) == summary(serial.outcomes)
    assert parallel_events.events == serial_events.events
    assert (parallel_out / "summary.txt").read_text() == (serial_out / "summary.txt").read_text()
    for name in ("A.txt", "B.txt", "C.txt"):
        assert (parallel_out / name).read_bytes() == (serial_out / name).read_bytes()


def test_reporter_failure_does_not_lose_outcomes(
    registries: ComponentRegistries, config: AppConfig, tmp_path: Path
) -> None:
    sources = write_inputs(tmp_path / "in", {"A.CR2": b"a"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.txt").write_text("previous run", encoding="utf-8")
    env = BuildEnv(config=config, output_dir=out)
    pipeline = PipelineFactory(registries).build_pipeline(env)

    result = BatchRunner(pipeline).run(sources, [FakeSummary(overwrite=False)], out)

    assert [o.ok for o in result.outcomes] == [True]
    assert set(result.reporter_errors) == {"fake_summary"}
    assert result.exit_code == 1
    assert (out / "summary.txt").read_text() == "previous run"
