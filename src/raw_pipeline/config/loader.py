"""Load and validate the YAML configuration file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError

_MERGE_TAG = "tag:yaml.org,2002:merge"
_MERGE_KEY = object()


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate or non-scalar mapping keys.

    Keys are checked on the composed node graph, before PyYAML flattens ``<<``
    merges, so every mapping written in the file (merge sources included) is
    checked exactly as written, and keys a merge overrides are not duplicates.
    """

    def construct_document(self, node: yaml.Node) -> object:
        _check_mapping_keys(self, node)
        return super().construct_document(node)


def _check_mapping_keys(loader: yaml.SafeLoader, root: yaml.Node) -> None:
    pending: list[yaml.Node] = [root]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            seen: set[object] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    msg = "mapping keys must be plain scalars"
                    raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
                key = _MERGE_KEY if key_node.tag == _MERGE_TAG else loader.construct_object(key_node)
                if key in seen:
                    msg = f"duplicate key {key_node.value!r}"
                    raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
                seen.add(key)
                pending.append(value_node)
        elif isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)


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
