from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.factory import PipelineFactory
from raw_pipeline.runtime.pipeline import Pipeline
from tests.unit.runtime.fakes import STAGES, FakeBrighten, FakeDecode, FakeDecoder, FakeSummary, write_inputs


def _pipeline(registries: ComponentRegistries, config: AppConfig, out: Path) -> Pipeline:
    return PipelineFactory(registries).build_pipeline(BuildEnv(config=config, output_dir=out))


def test_factory_builds_stages_in_config_order(
    registries: ComponentRegistries, config: AppConfig, tmp_path: Path
) -> None:
    env = BuildEnv(config=config, output_dir=tmp_path)
    factory = PipelineFactory(registries)
    assert factory.build_pipeline(env).stage_names == STAGES
    stages = config.pipeline.model_copy(update={"stages": ["fake_brighten", "fake_decode"]})
    reordered = BuildEnv(config=config.model_copy(update={"pipeline": stages}), output_dir=tmp_path)
    assert factory.build_pipeline(reordered).stage_names == ("fake_brighten", "fake_decode")
    assert [type(r) for r in factory.build_reporters(env)] == [FakeSummary]
    assert isinstance(factory.build_raw_decoder(env), FakeDecoder)


def test_factory_rejects_unknown_components_before_building(config: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="fake_decode"):
        _pipeline(ComponentRegistries.empty(), config, tmp_path)


def test_pipeline_runs_stages_in_order_and_records_everything(
    registries: ComponentRegistries, config: AppConfig, tmp_path: Path
) -> None:
    (source,) = write_inputs(tmp_path / "in", {"IMG_1.CR2": b"data"})
    out = tmp_path / "out"
    out.mkdir()
    run = _pipeline(registries, config, out).run(source)

    assert run.outcome.ok
    assert run.outcome.outputs == (out / "IMG_1.txt",)
    assert set(run.outcome.thumbnails) == {"after"}
    assert [t.stage for t in run.timings] == list(STAGES)
    assert all(t.seconds >= 0 for t in run.timings)
    flat = run.outcome.metadata.flatten()
    value = flat["fake_decode.value"]
    assert isinstance(value, float)
    assert float((out / "IMG_1.txt").read_text()) == pytest.approx(2 * value, abs=1e-5)


@pytest.mark.parametrize(
    "case",
    [(b"BAD file", "DecodeError", 0), (b"CRASH", "ValueError", 1)],
)
def test_stage_failure_is_captured_with_partial_metadata(
    registries: ComponentRegistries, config: AppConfig, tmp_path: Path, case: tuple[bytes, str, int]
) -> None:
    data, error_type, done = case
    (source,) = write_inputs(tmp_path / "in", {"IMG_1.CR2": data})
    run = _pipeline(registries, config, tmp_path).run(source)

    assert not run.outcome.ok
    assert run.outcome.error_type == error_type
    assert run.outcome.error
    assert len(run.timings) == done
    assert run.outcome.outputs == ()
    assert ("fake_decode.value" in run.outcome.metadata.flatten()) == (done == 1)


def test_pipeline_is_reusable_between_images(tmp_path: Path) -> None:
    sources = write_inputs(tmp_path, {"A.CR2": b"a", "B.CR2": b"b"})
    pipeline = Pipeline([FakeDecode(), FakeBrighten()])
    first = [pipeline.run(p).outcome.metadata.flatten() for p in sources]
    second = [pipeline.run(p).outcome.metadata.flatten() for p in sources]
    assert first == second
    assert first[0] != first[1]
