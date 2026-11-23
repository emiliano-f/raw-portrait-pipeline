from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.raw import DefectMap, PixelDefect
from raw_pipeline.runtime import services as services_module
from raw_pipeline.runtime.services import ComponentUnavailableError, RuntimeServices, resolve


class Detector:
    def __init__(self, config: DefectDetectionConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: DefectDetectionConfig) -> Detector:
        return cls(config)


def test_builtin_adapts_the_io_and_defects_signatures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def record(name: str, result: object) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return result

        return call

    targets: dict[tuple[str, str], Callable[..., object]] = {
        ("raw_pipeline.io.discovery", "discover_raw_files"): record("discover", [tmp_path / "A.CR2"]),
        ("raw_pipeline.io.defects_yaml", "write_defect_map"): record("write", tmp_path / "d.yaml"),
        ("raw_pipeline.io.defects_yaml", "merge_defect_map_into_config"): record("merge", tmp_path / "c.yaml"),
        ("raw_pipeline.defects.detection", "StatisticalDefectDetector"): Detector,
    }
    monkeypatch.setattr(services_module, "resolve", lambda module, attribute: targets[module, attribute])
    svc = RuntimeServices.builtin()
    defects = DefectMap((PixelDefect(x=1, y=2),))
    out = tmp_path / "out"
    config = DefectDetectionConfig()

    assert svc.discover(tmp_path, exclude_dirs=[out]) == [tmp_path / "A.CR2"]
    assert svc.write_defect_map(defects, tmp_path / "d.yaml", overwrite=True) == tmp_path / "d.yaml"
    svc.merge_defect_map(defects, tmp_path / "c.yaml")
    detector = svc.make_defect_detector(config)

    assert calls == [
        ("discover", (tmp_path,), {"exclude_dirs": (out,)}),
        ("write", (tmp_path / "d.yaml", defects), {"overwrite": True}),
        ("merge", (tmp_path / "c.yaml", defects), {}),
    ]
    assert isinstance(detector, Detector)
    assert detector.config is config


def test_resolve_reports_missing_modules() -> None:
    with pytest.raises(ComponentUnavailableError, match=r"raw_pipeline\.not_a_module"):
        resolve("raw_pipeline.not_a_module", "anything")
