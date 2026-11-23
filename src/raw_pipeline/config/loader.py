"""Load and validate the YAML configuration file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            msg = f"duplicate key {key!r}"
            raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def parse_config(data: Mapping[str, Any] | None) -> AppConfig:
    """Validate an already-parsed mapping. ``None`` (empty file) gives all defaults.

    Raises:
        ConfigError: With every validation problem listed, one per line.
    """
    try:
        return AppConfig.model_validate(dict(data or {}))
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in exc.errors()
        )
        msg = f"invalid configuration:\n{problems}"
        raise ConfigError(msg) from exc


def load_config(path: Path | None) -> AppConfig:
    """Read ``path`` (YAML) and validate it. ``None`` returns the defaults.

    Raises:
        ConfigError: If the file is missing, is not YAML, or fails validation.
    """
    if path is None:
        return AppConfig()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"cannot read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        msg = f"config file {path} is not valid YAML: {exc}"
        raise ConfigError(msg) from exc
    if data is not None and not isinstance(data, Mapping):
        msg = f"config file {path} must contain a mapping at the top level"
        raise ConfigError(msg)
    return parse_config(data)
