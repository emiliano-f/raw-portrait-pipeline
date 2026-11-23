from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import RetouchConfig
from raw_pipeline.core.artifacts import FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import FaceRegion, RetouchResult, freeze_record
from raw_pipeline.stages.retouch import RetouchStage
from raw_pipeline.stages.retouchers import FrequencySeparationRetoucher, NullRetoucher
from raw_pipeline.stages.retouchers.skin_mask import SkinMaskBuilder
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import decoded_context
from tests.unit.stages.retouchers.faces import mesh_face, skin_image


def face_context() -> ImageContext:
    return decoded_context(skin_image()).with_artifact(FACES, (mesh_face().face,))


class TestRetouchStageNullContract(StageContract):
    def make_stage(self) -> Stage:
        return RetouchStage(NullRetoucher())


class TestRetouchStageFrequencySeparationContract(StageContract):
    def make_stage(self) -> Stage:
        return RetouchStage(FrequencySeparationRetoucher(0.25, 0.015, SkinMaskBuilder(8)))

    def make_context(self) -> ImageContext:
        return face_context()


class _Spy:
    name: ClassVar[str] = "spy"

    def __init__(self) -> None:
        self.faces: list[Sequence[FaceRegion]] = []

    def retouch(self, image: NDArray[np.float32], faces: Sequence[FaceRegion]) -> RetouchResult:
        self.faces.append(faces)
        return RetouchResult(image * np.float32(0.5), freeze_record({"strength": 0.7}))


def test_passes_faces_and_records_method() -> None:
    spy = _Spy()
    ctx = face_context()
    out = RetouchStage(spy).process(ctx)
    assert spy.faces == [ctx.artifact(FACES)]
    np.testing.assert_array_equal(out.image, ctx.image * np.float32(0.5))
    assert dict(out.metadata.records["retouch"]) == {"method": "spy", "strength": 0.7}


def test_no_faces_artifact_passes_empty_sequence() -> None:
    spy = _Spy()
    RetouchStage(spy).process(decoded_context())
    assert spy.faces == [()]


def test_null_retoucher_is_passthrough() -> None:
    ctx = face_context()
    out = RetouchStage(NullRetoucher()).process(ctx)
    np.testing.assert_array_equal(out.image, ctx.image)
    assert out.metadata.records["retouch"]["method"] == "none"


def test_built_from_config() -> None:
    env = BuildEnv(config=AppConfig(retouch=RetouchConfig(method="none")), output_dir=Path("out"))
    stage = stage_registry.create("retouch", env)
    assert isinstance(stage, RetouchStage)
    assert isinstance(stage.retoucher, NullRetoucher)
