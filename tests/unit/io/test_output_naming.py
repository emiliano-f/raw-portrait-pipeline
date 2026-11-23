from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.core.errors import ConfigError, OutputExistsError
from raw_pipeline.io.output_naming import OutputPlanner, ensure_writable


def test_ensure_writable(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    assert ensure_writable(path, overwrite=False) == path
    path.write_bytes(b"old")
    with pytest.raises(OutputExistsError, match="already exists"):
        ensure_writable(path, overwrite=False)
    assert ensure_writable(path, overwrite=True) == path
    assert path.read_bytes() == b"old"


def test_plan_keeps_relative_dirs_and_suffix(tmp_path: Path) -> None:
    inp, out = tmp_path / "in", tmp_path / "out"
    planner = OutputPlanner(inp, out, suffix="_final")
    plan = planner.plan([inp / "IMG_1.CR2", inp / "day2" / "IMG_2.CR2"])
    assert plan == {inp / "IMG_1.CR2": out / "IMG_1_final", inp / "day2" / "IMG_2.CR2": out / "day2" / "IMG_2_final"}


def test_plan_rejects_case_collisions(tmp_path: Path) -> None:
    planner = OutputPlanner(tmp_path / "in", tmp_path / "out")
    with pytest.raises(ConfigError, match="both be written"):
        planner.plan([tmp_path / "in" / "IMG_1.CR2", tmp_path / "in" / "IMG_1.cr2"])


@pytest.mark.parametrize("output", ["in", "in/out"])
def test_refuses_output_inside_input(tmp_path: Path, output: str) -> None:
    with pytest.raises(ConfigError, match="inside the input"):
        OutputPlanner(tmp_path / "in", tmp_path / output).validate()
    OutputPlanner(tmp_path / "in", tmp_path / output, allow_inside_input=True).validate()


def test_sibling_with_common_prefix_is_not_inside(tmp_path: Path) -> None:
    OutputPlanner(tmp_path / "in", tmp_path / "in-out").validate()


@pytest.mark.parametrize("suffix", ["a/b", "a\\b"])
def test_rejects_separators_in_suffix(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ConfigError, match="separators"):
        OutputPlanner(tmp_path / "in", tmp_path / "out", suffix=suffix).validate()


def test_check_writable_honours_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "out" / "IMG_1.jpg"
    existing.parent.mkdir()
    existing.write_bytes(b"x")
    with pytest.raises(OutputExistsError):
        OutputPlanner(tmp_path / "in", tmp_path / "out").check_writable(existing)
    assert OutputPlanner(tmp_path / "in", tmp_path / "out", overwrite=True).check_writable(existing) == existing
