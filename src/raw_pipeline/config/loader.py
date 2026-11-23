"""Load and validate the YAML configuration file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError


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
    except OSError as exc:
        msg = f"cannot read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"config file {path} is not valid YAML: {exc}"
        raise ConfigError(msg) from exc
    if data is not None and not isinstance(data, Mapping):
        msg = f"config file {path} must contain a mapping at the top level"
        raise ConfigError(msg)
    return parse_config(data)
