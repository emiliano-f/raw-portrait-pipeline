"""Build the pipeline and batch reporters from a validated configuration."""

from __future__ import annotations

from raw_pipeline.config import app_config
from raw_pipeline.core.protocols import BatchReporter, RawDecoder
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.pipeline import Pipeline


class PipelineFactory:
    """Turns configuration into components through the registries (Factory).

    Adding a component means registering it; this class never changes.

    Args:
        registries: Registries to build from.
    """

    def __init__(self, registries: ComponentRegistries) -> None:
        self._registries = registries

    def build_pipeline(self, env: BuildEnv) -> Pipeline:
        """Validate ``env.config`` and build the stages of ``pipeline.stages``, in order.

        Raises:
            ConfigError: If the config names an unregistered component.
        """
        config = app_config(env)
        self._registries.validate(config)
        return Pipeline([self._registries.stages.create(name, env) for name in config.pipeline.stages])

    def build_reporters(self, env: BuildEnv) -> list[BatchReporter]:
        """Build the batch reporters of ``report.reporters``, in order."""
        return [self._registries.reporters.create(name, env) for name in app_config(env).report.reporters]

    def build_raw_decoder(self, env: BuildEnv) -> RawDecoder:
        """Build the RAW decoder of ``decode.decoder``."""
        return self._registries.raw_decoders.create(app_config(env).decode.decoder, env)
