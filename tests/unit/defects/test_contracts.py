from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.core.protocols import DefectCorrector
from raw_pipeline.core.registries import defect_corrector_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects import DirectionalCorrector, NeighborAverageCorrector, NullDefectCorrector
from tests.contract.defect_corrector_contract import DefectCorrectorContract


class TestNullDefectCorrectorContract(DefectCorrectorContract):
    is_passthrough = True

    def make_corrector(self) -> DefectCorrector:
        return NullDefectCorrector()


class TestNeighborAverageCorrectorContract(DefectCorrectorContract):
    def make_corrector(self) -> DefectCorrector:
        return NeighborAverageCorrector()


class TestDirectionalCorrectorContract(DefectCorrectorContract):
    def make_corrector(self) -> DefectCorrector:
        return DirectionalCorrector()


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("none", NullDefectCorrector),
        ("neighbor_average", NeighborAverageCorrector),
        ("directional", DirectionalCorrector),
    ],
)
def test_registered_under_plan_names(name: str, cls: type, tmp_path: Path) -> None:
    corrector = defect_corrector_registry.create(name, BuildEnv(config=AppConfig(), output_dir=tmp_path))
    assert isinstance(corrector, cls)
    assert corrector.name == name
