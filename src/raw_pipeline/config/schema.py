"""JSON schema export, used to generate the configuration reference."""

from __future__ import annotations

from typing import Any

from raw_pipeline.config.models import AppConfig


def config_json_schema() -> dict[str, Any]:
    """Return the JSON schema of :class:`AppConfig`."""
    return AppConfig.model_json_schema()
