"""Check component names in a configuration against the registries."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError

ComponentCatalog = Mapping[str, Collection[str]]
"""Registered names per extension-point kind, e.g. ``{"stages": {"decode", ...}}``."""


def component_references(config: AppConfig) -> list[tuple[str, str, str]]:
    """List ``(config path, registry kind, name)`` for every component the config names."""
    refs: list[tuple[str, str, str]] = [
        *((f"pipeline.stages[{i}]", "stages", name) for i, name in enumerate(config.pipeline.stages)),
        ("decode.decoder", "raw_decoders", config.decode.decoder),
        ("defects.method", "defect_correctors", config.defects.method),
        ("meter.strategy", "metering", config.meter.strategy),
        ("meter.fallback_strategy", "metering", config.meter.fallback_strategy),
        ("meter.detector", "detectors", config.meter.detector),
        ("retouch.method", "retouchers", config.retouch.method),
        *((f"export.formats[{i}]", "exporters", name) for i, name in enumerate(config.export.formats)),
        *((f"report.reporters[{i}]", "reporters", name) for i, name in enumerate(config.report.reporters)),
        ("qa.provider", "qa", config.qa.provider),
        ("raw_export.source", "dng_sources", config.raw_export.source),
    ]
    if config.meter.fallback_detector is not None:
        refs.append(("meter.fallback_detector", "detectors", config.meter.fallback_detector))
    return refs


def validate_components(config: AppConfig, catalog: ComponentCatalog) -> None:
    """Fail if the config names a component that is not registered.

    Raises:
        ConfigError: Listing each unknown name with the registered options.
    """
    problems = []
    for path, kind, name in component_references(config):
        options = catalog.get(kind, ())
        if name not in options:
            listed = ", ".join(sorted(options)) or "(none registered)"
            problems.append(f"  - {path}: unknown {kind} '{name}'; registered: {listed}")
    if len(set(config.pipeline.stages)) != len(config.pipeline.stages):
        problems.append("  - pipeline.stages: a stage is listed more than once")
    if problems:
        msg = "invalid configuration:\n" + "\n".join(problems)
        raise ConfigError(msg)
