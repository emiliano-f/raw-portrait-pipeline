from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from raw_pipeline.config import load_config
from raw_pipeline.config.defects import to_defect_map
from raw_pipeline.core.errors import ConfigError, OutputExistsError
from raw_pipeline.core.raw import ColumnDefect, DefectMap, PixelDefect, RowDefect
from raw_pipeline.io.defects_yaml import defect_map_yaml, merge_defect_map_into_config, write_defect_map

DEFECTS = DefectMap((ColumnDefect(2841), RowDefect(1203, start=0, end=3000), PixelDefect(512, 980)))


def test_standalone_yaml_is_a_loadable_partial_config(tmp_path: Path) -> None:
    path = write_defect_map(tmp_path / "defects.yaml", DEFECTS)
    text = path.read_text(encoding="utf-8")
    assert text == defect_map_yaml(DEFECTS)
    assert "- {type: column, index: 2841, width: 1}" in text
    assert to_defect_map(load_config(path).defects.map) == DEFECTS


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "defects.yaml"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        write_defect_map(path, DEFECTS)
    assert path.read_text(encoding="utf-8") == "keep"
    write_defect_map(path, DEFECTS, overwrite=True)
    assert "2841" in path.read_text(encoding="utf-8")


def test_empty_map(tmp_path: Path) -> None:
    path = write_defect_map(tmp_path / "defects.yaml", DefectMap(()))
    assert len(load_config(path).defects.map) == 0


def test_merge_preserves_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "tone:\n  highlight_rolloff_ev: 2.0\ndefects:\n  method: none\n  map:\n  - {type: pixel, x: 1, y: 1}\n"
        "  detection:\n    outlier_threshold_sigma: 7.0\n",
        encoding="utf-8",
    )
    merge_defect_map_into_config(path, DEFECTS, overwrite=True)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(data) == ["tone", "defects"]
    assert data["tone"] == {"highlight_rolloff_ev": 2.0}
    assert data["defects"]["method"] == "none"
    assert data["defects"]["detection"] == {"outlier_threshold_sigma": 7.0}
    assert to_defect_map(load_config(path).defects.map) == DEFECTS
    assert not list(tmp_path.glob(".*.tmp"))


def test_merge_creates_the_file_and_sets_method(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    merge_defect_map_into_config(path, DEFECTS, method="bayer_median")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["defects"]["method"] == "bayer_median"
    assert len(data["defects"]["map"]) == 3


@pytest.mark.parametrize("content", ["- a list\n", "defects: [1, 2]\n", "a: [unclosed\n"])
def test_merge_rejects_bad_files_without_touching_them(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        merge_defect_map_into_config(path, DEFECTS, overwrite=True)
    assert path.read_text(encoding="utf-8") == content


def test_merge_validates_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    content = "tone:\n  no_such_key: 1\n"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        merge_defect_map_into_config(path, DEFECTS, overwrite=True)
    assert path.read_text(encoding="utf-8") == content


def test_merge_never_rewrites_an_existing_config_silently(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    content = "# my notes\ntone:\n  highlight_rolloff_ev: 2.0\n"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(OutputExistsError):
        merge_defect_map_into_config(path, DEFECTS)
    assert path.read_text(encoding="utf-8") == content


def test_merge_to_a_new_path_leaves_the_source_untouched(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    content = "# my notes\ntone:\n  highlight_rolloff_ev: 2.0\n"
    path.write_text(content, encoding="utf-8")
    out = merge_defect_map_into_config(path, DEFECTS, output_path=tmp_path / "config.defects.yaml")
    assert out == tmp_path / "config.defects.yaml"
    assert path.read_text(encoding="utf-8") == content
    merged = load_config(out)
    assert merged.tone.highlight_rolloff_ev == 2.0
    assert to_defect_map(merged.defects.map) == DEFECTS
    with pytest.raises(OutputExistsError):
        merge_defect_map_into_config(path, DEFECTS, output_path=out)
