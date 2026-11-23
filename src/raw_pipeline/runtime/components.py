"""Component discovery: built-in modules, entry-point plugins and the catalog.

Built-in components register themselves when their module is imported. The
runtime imports every module of the built-in component packages (walking them,
so module names inside a package are free) and then loads entry-point plugins.
A module that cannot be imported is logged and skipped: its components are
simply not registered, and configuration validation reports them as unknown.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Final

from raw_pipeline.config.models import AppConfig
from raw_pipeline.config.validation import ComponentCatalog, validate_components
from raw_pipeline.core import registries as core_registries
from raw_pipeline.core.protocols import (
    BatchReporter,
    DefectCorrector,
    Exporter,
    FaceDetector,
    MeteringStrategy,
    QAProvider,
    RawDecoder,
    Retoucher,
    Stage,
)
from raw_pipeline.core.registry import Registry

logger = logging.getLogger(__name__)

BUILTIN_PACKAGES: Final[tuple[str, ...]] = (
    "raw_pipeline.io",
    "raw_pipeline.defects",
    "raw_pipeline.detection",
    "raw_pipeline.metering",
    "raw_pipeline.exporters",
    "raw_pipeline.qa",
    "raw_pipeline.stages",
)
"""Packages whose modules register the built-in components, imported in this order."""


@dataclass(frozen=True)
class ComponentRegistries:
    """The registries the runtime builds components from (injectable for tests).

    Attributes:
        stages: Per-image stages.
        raw_decoders: RAW decoders (``detect-defects`` opens frames with them).
        defect_correctors: Raw-level defect correctors.
        detectors: Face detectors.
        metering: Metering strategies.
        retouchers: Skin retouchers.
        exporters: Image exporters.
        qa: QA providers.
        reporters: Batch reporters.
    """

    stages: Registry[Stage]
    raw_decoders: Registry[RawDecoder]
    defect_correctors: Registry[DefectCorrector]
    detectors: Registry[FaceDetector]
    metering: Registry[MeteringStrategy]
    retouchers: Registry[Retoucher]
    exporters: Registry[Exporter]
    qa: Registry[QAProvider]
    reporters: Registry[BatchReporter]

    @classmethod
    def default(cls) -> ComponentRegistries:
        """The process-wide registries of :mod:`raw_pipeline.core.registries`."""
        return cls(
            stages=core_registries.stage_registry,
            raw_decoders=core_registries.raw_decoder_registry,
            defect_correctors=core_registries.defect_corrector_registry,
            detectors=core_registries.face_detector_registry,
            metering=core_registries.metering_registry,
            retouchers=core_registries.retoucher_registry,
            exporters=core_registries.exporter_registry,
            qa=core_registries.qa_registry,
            reporters=core_registries.reporter_registry,
        )

    @classmethod
    def empty(cls) -> ComponentRegistries:
        """Fresh, empty registries with the standard kinds (for tests and embedding)."""
        return cls(
            stages=Registry("stages"),
            raw_decoders=Registry("raw_decoders"),
            defect_correctors=Registry("defect_correctors"),
            detectors=Registry("detectors"),
            metering=Registry("metering"),
            retouchers=Registry("retouchers"),
            exporters=Registry("exporters"),
            qa=Registry("qa"),
            reporters=Registry("reporters"),
        )

    def all(self) -> tuple[Registry[Any], ...]:
        """Every registry, in a fixed order."""
        return (
            self.stages,
            self.raw_decoders,
            self.defect_correctors,
            self.detectors,
            self.metering,
            self.retouchers,
            self.exporters,
            self.qa,
            self.reporters,
        )

    def catalog(self) -> ComponentCatalog:
        """Registered names per registry kind, for :func:`validate_components`."""
        return {registry.kind: frozenset(registry.names()) for registry in self.all()}

    def validate(self, config: AppConfig) -> None:
        """Fail if ``config`` names a component that is not registered.

        Raises:
            ConfigError: Listing each unknown name with the registered options.
        """
        validate_components(config, self.catalog())

    def load_entry_points(self) -> list[str]:
        """Load third-party plugins into every registry; return ``kind:name`` loaded."""
        return [f"{registry.kind}:{name}" for registry in self.all() for name in registry.load_entry_points()]


def import_builtin_modules(packages: tuple[str, ...] = BUILTIN_PACKAGES) -> list[str]:
    """Import every module of ``packages`` so their components register themselves.

    Missing packages and modules that fail to import are logged and skipped.

    Returns:
        Names of the modules imported, in import order (sorted per package).
    """
    imported: list[str] = []
    for package_name in packages:
        package = _try_import(package_name)
        if package is None:
            continue
        imported.append(package_name)
        search_path = getattr(package, "__path__", None)
        if search_path is None:
            continue
        found = sorted(info.name for info in pkgutil.walk_packages(search_path, prefix=f"{package_name}."))
        imported.extend(name for name in found if _try_import(name) is not None)
    return imported


def load_components(registries: ComponentRegistries | None = None) -> ComponentRegistries:
    """Import the built-in components and load entry-point plugins.

    Args:
        registries: Registries to load plugins into; the process-wide ones by default.
            Built-in modules always register into the process-wide registries.

    Returns:
        The registries, ready for validation and building.
    """
    target = registries or ComponentRegistries.default()
    modules = import_builtin_modules()
    plugins = target.load_entry_points()
    logger.debug("imported %d built-in modules; plugins loaded: %s", len(modules), ", ".join(plugins) or "none")
    return target


def _try_import(name: str) -> Any | None:  # noqa: ANN401 - a module object
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        logger.warning("built-in component module %s not available: %s", name, exc)
        return None
