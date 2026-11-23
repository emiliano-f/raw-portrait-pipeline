from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from typing import ClassVar

import pytest

from raw_pipeline.core import registry as registry_module
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import RegistryError
from raw_pipeline.core.registries import ALL_REGISTRIES, stage_registry
from raw_pipeline.core.registry import BuildEnv, Registry


class _Echo:
    name: ClassVar[str] = "echo"

    def __init__(self, factor: int) -> None:
        self.factor = factor

    @classmethod
    def from_env(cls, env: BuildEnv) -> _Echo:
        return cls(int(str(env.options.get("factor", 1))))

    def process(self, ctx: ImageContext) -> ImageContext:
        return ctx


ENV = BuildEnv(config=object(), output_dir=Path("out"), options={"factor": 3})


def test_register_and_create() -> None:
    reg: Registry[_Echo] = Registry("things")
    reg.register("echo")(_Echo)
    assert reg.names() == ["echo"]
    assert "echo" in reg
    built = reg.create("echo", ENV)
    assert isinstance(built, _Echo)
    assert built.factor == 3


def test_duplicate_name_rejected() -> None:
    reg: Registry[_Echo] = Registry("things")
    reg.add("echo", _Echo.from_env)
    with pytest.raises(RegistryError, match="already registered"):
        reg.add("echo", _Echo.from_env)


def test_unknown_name_lists_options() -> None:
    reg: Registry[_Echo] = Registry("things")
    with pytest.raises(RegistryError, match=r"registered: \(none\)"):
        reg.create("nope", ENV)
    reg.add("a", _Echo.from_env)
    with pytest.raises(RegistryError, match="registered: a"):
        reg.create("nope", ENV)


def test_entry_points_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = EntryPoint(name="echo", value=f"{__name__}:_Echo", group="raw_pipeline.things")
    dup = EntryPoint(name="taken", value=f"{__name__}:_Echo", group="raw_pipeline.things")
    seen: list[str] = []

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        seen.append(group)
        return [ep, dup]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    reg: Registry[_Echo] = Registry("things")
    reg.add("taken", _Echo.from_env)
    assert reg.load_entry_points() == ["echo"]
    assert seen == ["raw_pipeline.things"]
    assert reg.entry_point_group == "raw_pipeline.things"
    assert reg.create("echo", ENV).factor == 3


def test_all_registries_have_distinct_kinds() -> None:
    kinds = [r.kind for r in ALL_REGISTRIES]
    assert len(kinds) == len(set(kinds))
    assert stage_registry.kind == "stages"
