from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import QAConfig
from raw_pipeline.core.artifacts import QA_RESULT
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import ConfigError, StageError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import QAResult
from raw_pipeline.qa.null import NullQAProvider
from raw_pipeline.qa.openai_compatible import OpenAICompatibleProvider
from raw_pipeline.stages.qa import QAStage, qa_record
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context, flat_image


class _FlaggingProvider:
    """Flags every image and remembers the previews it received."""

    name: ClassVar[str] = "fake"

    def __init__(self, flags: frozenset[str] = frozenset({"closed_eyes"})) -> None:
        self.flags = flags
        self.previews: list[bytes] = []

    def review(self, preview_jpeg: bytes) -> QAResult:
        self.previews.append(preview_jpeg)
        return QAResult(provider=self.name, flags=self.flags, notes="eyes closed", model="fake-1")


class _RaisingProvider:
    name: ClassVar[str] = "broken"

    def review(self, preview_jpeg: bytes) -> QAResult:
        del preview_jpeg
        msg = "plugin bug"
        raise RuntimeError(msg)


class TestQAStageContract(StageContract):
    def make_stage(self) -> Stage:
        return QAStage(_FlaggingProvider())


class TestQAStageNullContract(StageContract):
    def make_stage(self) -> Stage:
        return QAStage(NullQAProvider())


def _big_context() -> ImageContext:
    return decoded_context(flat_image(0.18, 1800, 2700))


def test_pixels_unchanged_and_result_recorded() -> None:
    ctx = _big_context()
    out = QAStage(_FlaggingProvider()).process(ctx)
    assert out.image is ctx.image
    assert out.artifact(QA_RESULT) == QAResult(
        provider="fake", flags=frozenset({"closed_eyes"}), notes="eyes closed", model="fake-1"
    )
    rec = out.metadata.records["qa"]
    assert rec["closed_eyes"] is True
    assert rec["blur"] is False
    assert rec["skin_tone"] is False
    assert rec["skipped"] is False
    assert rec["flags"] == "closed_eyes"
    assert rec["provider"] == "fake"
    assert rec["model"] == "fake-1"


@pytest.mark.parametrize(("requested", "expected_long_edge"), [(1024, 1024), (512, 512), (5000, 1024)])
def test_preview_sent_is_small_jpeg_without_exif(requested: int, expected_long_edge: int) -> None:
    provider = _FlaggingProvider()
    QAStage(provider, preview_long_edge_px=requested, preview_jpeg_quality=80).process(_big_context())
    (preview,) = provider.previews
    with Image.open(io.BytesIO(preview)) as img:
        assert img.format == "JPEG"
        assert max(img.size) == expected_long_edge
        assert "exif" not in img.info
        assert not img.getexif()


def test_null_provider_marks_flags_unknown() -> None:
    out = QAStage(NullQAProvider()).process(decoded_context())
    rec = out.metadata.records["qa"]
    assert rec["skipped"] is True
    assert rec["closed_eyes"] is None
    assert rec["provider"] == "none"


def test_provider_exception_never_fails_image(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        out = QAStage(_RaisingProvider()).process(decoded_context(name="IMG_0042.CR2"))
    result = out.artifact(QA_RESULT)
    assert result is not None
    assert result.skipped
    assert result.notes == "provider error: RuntimeError"
    assert "IMG_0042.CR2" in caplog.text


def test_requires_decoded_image() -> None:
    with pytest.raises(StageError, match="decoded"):
        QAStage(NullQAProvider()).process(ImageContext.initial(Path("IMG_0001.CR2")))


def test_qa_record_lists_every_flag() -> None:
    rec = qa_record(QAResult(provider="p", flags=frozenset({"skin_tone", "blur"})))
    assert rec["flags"] == "blur;skin_tone"
    assert rec["blur"] is True
    assert rec["closed_eyes"] is False


def test_from_env_default_is_null_provider() -> None:
    stage = stage_registry.create("qa", BuildEnv(config=AppConfig(), output_dir=Path("out")))
    assert isinstance(stage, QAStage)
    assert isinstance(stage.provider, NullQAProvider)
    assert stage.preview_long_edge_px == 1024


def test_from_env_builds_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QA_API_KEY", raising=False)
    qa = QAConfig(
        provider="openai_compatible",
        endpoint="https://vision.example/v1",
        model="m",
        preview_long_edge_px=640,
        preview_jpeg_quality=70,
    )
    cfg = AppConfig(qa=qa)
    stage = QAStage.from_env(BuildEnv(config=cfg, output_dir=Path("out")))
    assert isinstance(stage.provider, OpenAICompatibleProvider)
    assert stage.preview_long_edge_px == 640
    assert stage.preview_jpeg_quality == 70


def test_without_api_key_pipeline_image_still_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QA_API_KEY", raising=False)
    cfg = AppConfig(qa=QAConfig(provider="openai_compatible", endpoint="https://vision.example/v1", model="m"))
    ctx = decoded_context()
    out = QAStage.from_env(BuildEnv(config=cfg, output_dir=Path("out"))).process(ctx)
    np.testing.assert_array_equal(out.image, ctx.image)
    assert out.metadata.records["qa"]["skipped"] is True


@pytest.mark.parametrize("provider", ["openai_compatible", "some_plugin"])
def test_from_env_requires_endpoint_and_model_for_any_provider(provider: str) -> None:
    cfg = AppConfig(qa=QAConfig(provider=provider))
    with pytest.raises(ConfigError, match=r"qa\.endpoint and qa\.model must be set"):
        QAStage.from_env(BuildEnv(config=cfg, output_dir=Path("out")))
