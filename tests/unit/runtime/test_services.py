from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
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


def _signature_checked(real: Callable[..., object]) -> Callable[..., object]:
    signature = inspect.signature(real)

    def call(*args: object, **kwargs: object) -> object:
        bound = signature.bind(*args, **kwargs).arguments
        for name, value in bound.items():
            if name.endswith(("path", "_dir")):
                assert isinstance(value, Path), f"{name} got {value!r}"
            if name.endswith("defect_map"):
                assert isinstance(value, DefectMap), f"{name} got {value!r}"
        return None

    return call


@pytest.mark.parametrize(
    ("module", "attribute", "use"),
    [
        ("raw_pipeline.io.discovery", "discover_raw_files", lambda s, p: s.discover(p, exclude_dirs=[p / "out"])),
        (
            "raw_pipeline.io.defects_yaml",
            "write_defect_map",
            lambda s, p: s.write_defect_map(DefectMap(), p / "d.yaml", overwrite=True),
        ),
        (
            "raw_pipeline.io.defects_yaml",
            "merge_defect_map_into_config",
            lambda s, p: s.merge_defect_map(DefectMap(), p / "c.yaml"),
        ),
        (
            "raw_pipeline.defects.detection",
            "StatisticalDefectDetector",
            lambda s, _: s.make_defect_detector(DefectDetectionConfig()),
        ),
    ],
)
def test_builtin_calls_match_the_real_signatures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: str,
    attribute: str,
    use: Callable[[RuntimeServices, Path], object],
) -> None:
    real = getattr(pytest.importorskip(module), attribute)
    if isinstance(real, type):
        fake: object = SimpleNamespace(from_config=_signature_checked(real.from_config))
    else:
        fake = _signature_checked(real)
    monkeypatch.setattr(services_module, "resolve", lambda m, a: fake if (m, a) == (module, attribute) else None)
    use(RuntimeServices.builtin(), tmp_path)
