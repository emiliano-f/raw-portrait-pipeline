from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.core import registries as core_registries
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.runtime.components import ComponentRegistries, import_builtin_modules, load_components
from tests.unit.runtime.fakes import fake_config


def test_default_wraps_process_wide_registries() -> None:
    regs = ComponentRegistries.default()
    assert regs.stages is core_registries.stage_registry
    assert regs.reporters is core_registries.reporter_registry
    assert {r.kind for r in regs.all()} == {
        "stages", "raw_decoders", "defect_correctors", "detectors", "metering",
        "retouchers", "exporters", "qa", "reporters",
    }  # fmt: skip


def test_validate_reports_every_unknown_name(registries: ComponentRegistries) -> None:
    regs = ComponentRegistries.empty()
    with pytest.raises(ConfigError) as info:
        regs.validate(fake_config())
    message = str(info.value)
    assert "fake_decode" in message
    assert "fake_summary" in message
    registries.validate(fake_config())


def test_missing_builtin_packages_are_skipped() -> None:
    assert import_builtin_modules(("raw_pipeline.does_not_exist",)) == []


def test_builtin_packages_are_walked_and_broken_modules_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "rp_fake_builtins"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (package / "sub" / "good.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "broken.py").write_text("import rp_missing_optional_dependency\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    imported = import_builtin_modules(("rp_fake_builtins",))

    assert imported == ["rp_fake_builtins", "rp_fake_builtins.sub", "rp_fake_builtins.sub.good"]


def test_load_components_loads_entry_points_into_given_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    regs = ComponentRegistries.empty()
    loaded: list[ComponentRegistries] = []

    def fake_load(self: ComponentRegistries) -> list[str]:
        loaded.append(self)
        return []

    monkeypatch.setattr(ComponentRegistries, "load_entry_points", fake_load)
    assert load_components(regs) is regs
    assert loaded == [regs]
