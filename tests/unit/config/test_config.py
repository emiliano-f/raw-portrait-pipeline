from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from raw_pipeline.config import AppConfig, app_config, load_config, parse_config, validate_components
from raw_pipeline.config.defects import entry_to_yaml_dict, to_defect_map, to_entry
from raw_pipeline.config.loader import _UniqueKeyLoader
from raw_pipeline.config.models import DefectEntry
from raw_pipeline.config.schema import config_json_schema
from raw_pipeline.config.validation import component_references
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.raw import ColumnDefect, PixelDefect, RowDefect
from raw_pipeline.core.registry import BuildEnv

REPO = Path(__file__).resolve().parents[3]

FULL_CATALOG = {
    "stages": {"decode", "meter", "tone", "retouch", "export", "report", "qa"},
    "raw_decoders": {"rawpy"},
    "defect_correctors": {"none", "neighbor_average", "directional"},
    "metering": {"face_weighted", "center_weighted"},
    "detectors": {"mediapipe", "opencv"},
    "retouchers": {"none", "frequency_separation"},
    "exporters": {"jpeg", "tiff"},
    "reporters": {"csv", "contact_sheet"},
    "qa": {"none", "openai_compatible"},
}


def test_defaults() -> None:
    cfg = AppConfig()
    assert cfg.pipeline.stages[0] == "decode"
    assert cfg.defects.method == "none"
    assert cfg.meter.target_face_luminance == 0.18
    assert cfg.qa.provider == "none"
    validate_components(cfg, FULL_CATALOG)


def test_load_none_and_empty_file(tmp_path: Path) -> None:
    assert load_config(None) == AppConfig()
    empty = tmp_path / "c.yaml"
    empty.write_text("")
    assert load_config(empty) == AppConfig()


def test_load_example_config() -> None:
    cfg = load_config(REPO / "config.example.yaml")
    validate_components(cfg, FULL_CATALOG)
    assert cfg.defects.method == "directional"
    assert len(cfg.defects.map) == 3


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1,")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_validation_messages_list_paths() -> None:
    with pytest.raises(ConfigError, match=r"meter\.target_face_luminance"):
        parse_config({"meter": {"target_face_luminance": -1}})
    with pytest.raises(ConfigError, match=r"tone\.nope"):
        parse_config({"tone": {"nope": 1}})
    with pytest.raises(ConfigError, match=r"qa\.preview_long_edge_px"):
        parse_config({"qa": {"preview_long_edge_px": 2048}})


def test_config_is_frozen() -> None:
    cfg = AppConfig()
    with pytest.raises(ValueError, match="frozen"):
        cfg.meter.strategy = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"type": "pixel", "x": 1}, "need x and y"),
        ({"type": "column"}, "need index"),
        ({"type": "row", "index": 3, "start": 5, "end": 5}, "greater than start"),
        ({"type": "column", "index": 3, "width": 0}, "width"),
        ({"type": "diagonal", "index": 3}, "type"),
    ],
)
def test_defect_entry_validation(entry: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config({"defects": {"map": [entry]}})


def test_defect_conversion_round_trip() -> None:
    cfg = parse_config(
        {
            "defects": {
                "method": "directional",
                "map": [
                    {"type": "column", "index": 2841, "width": 1},
                    {"type": "row", "index": 1203, "start": 0, "end": 3000, "width": 1},
                    {"type": "pixel", "x": 512, "y": 980},
                ],
            }
        }
    )
    dmap = to_defect_map(cfg.defects.map)
    assert dmap.defects == (
        ColumnDefect(2841, 1, None, None),
        RowDefect(1203, 1, 0, 3000),
        PixelDefect(512, 980),
    )
    assert tuple(to_entry(d) for d in dmap) == cfg.defects.map
    assert entry_to_yaml_dict(to_entry(PixelDefect(1, 2))) == {"type": "pixel", "x": 1, "y": 2}
    assert entry_to_yaml_dict(DefectEntry(type="column", index=4)) == {"type": "column", "index": 4, "width": 1}


def test_validate_components_reports_every_unknown_name() -> None:
    cfg = parse_config(
        {
            "pipeline": {"stages": ["decode", "vignette", "decode"]},
            "meter": {"detector": "dlib"},
            "export": {"formats": ["jpeg", "webp"]},
        }
    )
    with pytest.raises(ConfigError) as info:
        validate_components(cfg, FULL_CATALOG)
    text = str(info.value)
    assert "pipeline.stages[1]: unknown stages 'vignette'; registered: decode, export" in text
    assert "meter.detector: unknown detectors 'dlib'; registered: mediapipe, opencv" in text
    assert "export.formats[1]" in text
    assert "listed more than once" in text


def test_validate_components_empty_catalog() -> None:
    with pytest.raises(ConfigError, match=r"\(none registered\)"):
        validate_components(AppConfig(), {})


def test_component_references_skip_missing_fallback() -> None:
    refs = component_references(parse_config({"meter": {"fallback_detector": None}}))
    assert all(path != "meter.fallback_detector" for path, _, _ in refs)


def test_app_config_accessor() -> None:
    cfg = AppConfig()
    assert app_config(BuildEnv(config=cfg, output_dir=Path())) is cfg
    with pytest.raises(ConfigError, match="AppConfig"):
        app_config(BuildEnv(config={}, output_dir=Path()))


def test_schema_has_descriptions() -> None:
    schema = config_json_schema()
    meter = schema["$defs"]["MeterConfig"]["properties"]
    assert "description" in meter["target_face_luminance"]


def test_non_utf8_config_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"meter:\n  strategy: \xff\xfe\n")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("meter:\n  target_face_luminance: 0.2\nmeter:\n  strategy: center_weighted\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'meter'"):
        load_config(path)


@pytest.mark.parametrize("entry", [{"end": 0}, {"start": 5, "end": 5}, {"start": 6, "end": 2}])
def test_empty_defect_ranges_are_rejected(entry: dict[str, int]) -> None:
    with pytest.raises(ConfigError, match="end must be greater than start"):
        parse_config({"defects": {"map": [{"type": "row", "index": 3, **entry}]}})


def test_yaml_merge_keys_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "merge.yaml"
    path.write_text(
        "base: &base {target_face_luminance: 0.2}\nmeter:\n  <<: *base\n  strategy: center_weighted\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="base"):
        load_config(path)  # unknown top-level key, but the merge itself parses
    path.write_text("meter:\n  <<: {target_face_luminance: 0.2}\n  strategy: center_weighted\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.meter.target_face_luminance == 0.2
    assert cfg.meter.strategy == "center_weighted"


def test_unhashable_yaml_key_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "listkey.yaml"
    path.write_text("? [a, b]\n: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


@pytest.mark.parametrize(
    "text",
    [
        "meter:\n  <<: {strategy: center_weighted}\n  <<: {target_face_luminance: 0.2}\n",
        "meter:\n  <<: {target_face_luminance: 0.2, target_face_luminance: 0.3}\n",
    ],
)
def test_duplicate_keys_in_merges_are_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "dup-merge.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)


def test_aliased_merge_source_with_override_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "alias.yaml"
    path.write_text(
        "meter: &m\n  <<: {target_face_luminance: 0.2}\n  target_face_luminance: 0.25\n"
        "tone:\n  shadow_lift_ev: 0.1\n"
        "export: {<<: {jpeg_quality: 90}, jpeg_quality: 95}\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.meter.target_face_luminance == 0.25
    assert cfg.export.jpeg_quality == 95


def test_reused_merge_source_with_override_parses() -> None:
    text = "a: &a {<<: {k: 1}, k: 2}\nb: {<<: *a}\nc: *a\n"
    data = yaml.load(text, Loader=_UniqueKeyLoader)
    assert data == yaml.safe_load(text)


@pytest.mark.parametrize("text", ["a: {10: x, 0xA: y}\n", "a: {1: x, 1.0: y}\n", "a: {true: x, yes: y}\n"])
def test_equivalent_yaml_keys_are_duplicates(text: str) -> None:
    with pytest.raises(yaml.YAMLError, match="duplicate key"):
        yaml.load(text, Loader=_UniqueKeyLoader)


def test_bare_equals_key_parses() -> None:
    text = "a: {=: x, b: y}\n"
    assert yaml.load(text, Loader=_UniqueKeyLoader) == yaml.safe_load(text)
