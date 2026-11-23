"""Configuration: pydantic models, YAML loading, validation and schema export."""

from __future__ import annotations

from raw_pipeline.config.loader import load_config, parse_config
from raw_pipeline.config.models import AppConfig
from raw_pipeline.config.validation import validate_components
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.registry import BuildEnv


def app_config(env: BuildEnv) -> AppConfig:
    """Return ``env.config`` typed as :class:`AppConfig`.

    Raises:
        ConfigError: If the environment was built with something else.
    """
    if not isinstance(env.config, AppConfig):
        msg = f"BuildEnv.config must be an AppConfig, got {type(env.config).__name__}"
        raise ConfigError(msg)
    return env.config


__all__ = ["AppConfig", "app_config", "load_config", "parse_config", "validate_components"]
