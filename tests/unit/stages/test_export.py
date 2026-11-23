from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.core.artifacts import OUTPUT_PATHS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import OutputExistsError, StageError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.exporters.jpeg import JpegExporter
from raw_pipeline.exporters.tiff import TiffExporter
from raw_pipeline.stages.export import ExportStage
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context, gradient_image
from tests.unit.exporters.helpers import build_env


class _FakeExporter:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def export(self, ctx: ImageContext, destination: Path) -> Path:  # noqa: ARG002
        self.calls.append(destination)
        return destination.with_name(destination.name + ".fake")


class TestExportStageContract(StageContract):
    @pytest.fixture(autouse=True)
    def _output_dir(self, tmp_path: Path) -> None:
        self.output_dir = tmp_path

    def make_stage(self) -> Stage:
        exporter = JpegExporter(quality=90, long_edge_px=None, keep_gps=False, overwrite=True)
        return ExportStage([exporter], self.output_dir)


def test_runs_exporters_in_order_and_sets_output_paths(tmp_path: Path) -> None:
    first, second = _FakeExporter(), _FakeExporter()
    stage = ExportStage([first, second], tmp_path, filename_suffix="_final", long_edge_px=40)
    ctx = decoded_context(gradient_image(60, 90), name="in/IMG_0042.CR2")
    out = stage.process(ctx)
    assert first.calls == second.calls == [tmp_path / "IMG_0042_final"]
    assert out.artifact(OUTPUT_PATHS) == (tmp_path / "IMG_0042_final.fake",) * 2
    record = out.metadata.records["export"]
    assert record["formats"] == "fake,fake"
    assert record["files"] == "IMG_0042_final.fake;IMG_0042_final.fake"
    assert (record["output_width"], record["output_height"]) == (40, 27)
    assert out.image is ctx.image


def test_built_from_config(tmp_path: Path) -> None:
    env = build_env(tmp_path, export={"formats": ["jpeg", "tiff"], "jpeg_quality": 80, "filename_suffix": "-x"})
    stage = stage_registry.create("export", env)
    assert isinstance(stage, ExportStage)
    assert [type(e) for e in stage.exporters] == [JpegExporter, TiffExporter]
    out = stage.process(decoded_context(gradient_image(16, 24)))
    assert [p.name for p in out.artifact(OUTPUT_PATHS) or ()] == ["IMG_0001-x.jpg", "IMG_0001-x.tif"]
    record = out.metadata.records["export"]
    assert record["jpeg_quality"] == 80
    assert record["keep_gps"] is False
    assert record["long_edge_px"] is None
    assert record["color_space"] == "srgb"


def test_requires_decoded_image(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        ExportStage([_FakeExporter()], tmp_path).process(ImageContext.initial(Path("a.CR2")))


def test_output_conflict_removes_files_written_for_the_image(tmp_path: Path) -> None:
    (tmp_path / "IMG_0001.tif").write_bytes(b"previous")
    stage = ExportStage(
        [
            JpegExporter(quality=90, long_edge_px=None, keep_gps=False, overwrite=False),
            TiffExporter(long_edge_px=None, overwrite=False),
        ],
        tmp_path,
    )
    with pytest.raises(OutputExistsError):
        stage.process(decoded_context(gradient_image(16, 24)))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["IMG_0001.tif"]
    assert (tmp_path / "IMG_0001.tif").read_bytes() == b"previous"
