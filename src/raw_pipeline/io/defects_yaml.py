"""Write detected defect maps as YAML: standalone files or merged into a config file."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from raw_pipeline.config import parse_config
from raw_pipeline.config.defects import entry_to_yaml_dict, to_entry
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.raw import DefectMap
from raw_pipeline.io.output_naming import ensure_writable

STANDALONE_HEADER: Final = (
    "# Sensor defects found by `raw-pipeline detect-defects`.\n"
    "# Raw sensor coordinates (visible area, before rotation). Merge into your config\n"
    "# under `defects.map` and choose `defects.method`.\n"
)


class _FlowEntry(dict[str, Any]):
    """A defect entry, dumped on one line: ``{type: column, index: 2841, width: 1}``."""


class _DefectsDumper(yaml.SafeDumper):
    """Safe dumper that writes defect entries in flow style and keeps key order."""


def _represent_flow_entry(dumper: yaml.SafeDumper, data: _FlowEntry) -> yaml.MappingNode:
    return dumper.represent_mapping("tag:yaml.org,2002:map", list(data.items()), flow_style=True)


_DefectsDumper.add_representer(_FlowEntry, _represent_flow_entry)


def _map_entries(defect_map: DefectMap) -> list[_FlowEntry]:
    return [_FlowEntry(entry_to_yaml_dict(to_entry(d))) for d in defect_map]


def _dump(data: Mapping[str, Any]) -> str:
    return yaml.dump(data, Dumper=_DefectsDumper, sort_keys=False, default_flow_style=False, allow_unicode=True)


def defect_map_yaml(defect_map: DefectMap) -> str:
    """Return a YAML document ``defects: {map: [...]}`` for ``defect_map`` (deterministic)."""
    return STANDALONE_HEADER + _dump({"defects": {"map": _map_entries(defect_map)}})


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_defect_map(path: Path, defect_map: DefectMap, *, overwrite: bool = False) -> Path:
    """Write ``defect_map`` to a standalone YAML file (``detect-defects --output``).

    The file is itself a valid partial configuration.

    Raises:
        OutputExistsError: If ``path`` exists and ``overwrite`` is false.
    """
    ensure_writable(path, overwrite=overwrite)
    _atomic_write(path, defect_map_yaml(defect_map))
    return path


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        msg = f"cannot read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"config file {path} must contain a mapping at the top level"
        raise ConfigError(msg)
    return data


def merge_defect_map_into_config(config_path: Path, defect_map: DefectMap, *, method: str | None = None) -> Path:
    """Replace ``defects.map`` in ``config_path`` (created if missing), keeping every other key.

    Key order is preserved; YAML comments are not (PyYAML does not keep them).
    The merged document is validated before the file is replaced atomically,
    so an invalid result never reaches disk.

    Args:
        config_path: The YAML configuration file.
        defect_map: Defects to store.
        method: Also set ``defects.method`` when given.

    Raises:
        ConfigError: If the file cannot be parsed or the merged config is invalid.
    """
    data = _read_mapping(config_path)
    defects = data.get("defects") or {}
    if not isinstance(defects, dict):
        msg = f"{config_path}: 'defects' must be a mapping"
        raise ConfigError(msg)
    defects = dict(defects)
    if method is not None:
        defects["method"] = method
    defects["map"] = _map_entries(defect_map)
    data["defects"] = defects
    parse_config(data)
    _atomic_write(config_path, _dump(data))
    return config_path
