"""Where a remote QA provider sends previews: always named by the user, never a built-in default."""

from __future__ import annotations

from raw_pipeline.config.models import QAConfig
from raw_pipeline.core.errors import ConfigError


def require_endpoint_and_model(cfg: QAConfig) -> tuple[str, str]:
    """Return ``(qa.endpoint, qa.model)``; both are mandatory once ``qa.provider`` is not ``none``.

    Raises:
        ConfigError: Naming every missing field, so portraits never go to a host the user did not choose.
    """
    missing = [f"qa.{name}" for name in ("endpoint", "model") if not getattr(cfg, name)]
    if missing or cfg.endpoint is None or cfg.model is None:
        msg = f"{' and '.join(missing)} must be set when qa.provider is {cfg.provider!r}"
        raise ConfigError(msg)
    return cfg.endpoint, cfg.model
