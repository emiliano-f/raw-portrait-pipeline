"""Collaborators the runtime uses from lower-layer packages, resolved lazily.

``io`` (input discovery, defect-map YAML) and ``defects`` (the statistical
detector) are separate packages; the runtime reaches them through the small
callable interfaces below so tests inject fakes and a missing package fails
with a clear message only when the feature is actually used.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.errors import PipelineError
from raw_pipeline.core.protocols import DefectDetector
from raw_pipeline.core.raw import DefectMap


class ComponentUnavailableError(PipelineError):
    """A built-in module the command needs is not installed or not importable."""


class InputDiscovery(Protocol):
    """Lists the RAW files of an input directory, sorted, never inside ``exclude_dirs``."""

    def __call__(self, input_dir: Path, *, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
        """Return the RAW files of ``input_dir``."""
        ...


class DefectMapWriter(Protocol):
    """Writes a defect map to a standalone YAML file (``defects.map`` section)."""

    def __call__(self, defects: DefectMap, path: Path, *, overwrite: bool) -> Path:
        """Write ``defects`` to ``path`` and return it; refuses to overwrite unless asked."""
        ...


class DefectMapMerger(Protocol):
    """Replaces ``defects.map`` in an existing config file, keeping every other key."""

    def __call__(self, defects: DefectMap, config_path: Path) -> None:
        """Merge ``defects`` into ``config_path``."""
        ...


DefectDetectorFactory = Callable[[DefectDetectionConfig], DefectDetector]
"""Builds a defect detector from the ``defects.detection`` section."""


@dataclass(frozen=True)
class RuntimeServices:
    """Lower-layer collaborators of the CLI commands (injectable for tests).

    Attributes:
        discover: Input discovery.
        write_defect_map: Standalone defect-map writer (``--output``).
        merge_defect_map: Config-file merger (no ``--output``).
        make_defect_detector: Defect detector factory.
    """

    discover: InputDiscovery
    write_defect_map: DefectMapWriter
    merge_defect_map: DefectMapMerger
    make_defect_detector: DefectDetectorFactory

    @classmethod
    def builtin(cls) -> RuntimeServices:
        """The built-in implementations from ``raw_pipeline.io`` and ``raw_pipeline.defects``."""
        return cls(
            discover=_discover,
            write_defect_map=_write_defect_map,
            merge_defect_map=_merge_defect_map,
            make_defect_detector=_make_defect_detector,
        )


def _discover(input_dir: Path, *, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    found = resolve("raw_pipeline.io.discovery", "discover_raw_files")(input_dir, exclude_dirs=tuple(exclude_dirs))
    return cast("list[Path]", found)


def _write_defect_map(defects: DefectMap, path: Path, *, overwrite: bool) -> Path:
    return cast("Path", resolve("raw_pipeline.io.defects_yaml", "write_defect_map")(path, defects, overwrite=overwrite))


def _merge_defect_map(defects: DefectMap, config_path: Path) -> None:
    resolve("raw_pipeline.io.defects_yaml", "merge_defect_map_into_config")(config_path, defects)


def _make_defect_detector(config: DefectDetectionConfig) -> DefectDetector:
    detector_cls = resolve("raw_pipeline.defects.detection", "StatisticalDefectDetector")
    return cast("DefectDetector", detector_cls.from_config(config))  # type: ignore[attr-defined]


def resolve(module: str, attribute: str) -> Callable[..., Any]:
    """Import ``module`` and return its callable ``attribute``.

    Raises:
        ComponentUnavailableError: If the module or attribute does not exist.
    """
    try:
        target = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        msg = f"{module}.{attribute} is not available: {exc}"
        raise ComponentUnavailableError(msg) from exc
    return cast("Callable[..., Any]", target)
