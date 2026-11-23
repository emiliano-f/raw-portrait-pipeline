from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from raw_pipeline.config.models import AppConfig
from raw_pipeline.runtime.components import ComponentRegistries
from tests.unit.runtime.fakes import FakeServices, fake_config, make_registries, write_inputs

INPUTS = {"IMG_0003.CR2": b"three", "IMG_0001.CR2": b"one", "IMG_0002.CR2": b"two", "notes.txt": b"x"}


@pytest.fixture
def config() -> AppConfig:
    return fake_config()


@pytest.fixture
def config_path(tmp_path: Path, config: AppConfig) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return path


@pytest.fixture
def registries(config: AppConfig) -> ComponentRegistries:
    return make_registries(config)


@pytest.fixture
def services() -> FakeServices:
    return FakeServices()


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    write_inputs(folder, INPUTS)
    return folder
