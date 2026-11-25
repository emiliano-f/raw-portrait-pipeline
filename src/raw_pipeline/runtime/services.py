"""Collaborators the runtime uses from lower-layer packages, resolved lazily.

``io`` (input discovery, defect-map YAML) and ``defects`` (the statistical
detector) are separate packages; the runtime reaches them through the small
callable interfaces below so tests inject fakes and a missing package fails
with a clear message only when the feature is actually used.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.dng import DngRawImage
from raw_pipeline.core.errors import PipelineError
from raw_pipeline.core.protocols import DefectDetector, DngRawWriter
from raw_pipeline.core.raw import CFAFrame, Defect, DefectMap


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

    def __call__(self, defects: DefectMap, config_path: Path, *, overwrite: bool) -> Path:
        """Merge ``defects`` into ``config_path``; refuses to rewrite an existing file unless asked."""
        ...


DefectDetectorFactory = Callable[[DefectDetectionConfig], DefectDetector]
"""Builds a defect detector from the ``defects.detection`` section."""


class FileDiscovery(Protocol):
    """Lists the files of ``input_dir`` with one of ``extensions`` (lower case), sorted."""

    def __call__(
        self, input_dir: Path, *, extensions: Collection[str], exclude_dirs: Iterable[Path] = ()
    ) -> list[Path]:
        """Return the matching files of ``input_dir`` (not recursive)."""
        ...


class ReportWriter(Protocol):
    """Writes CSV rows to ``path``; refuses to replace it unless ``overwrite``."""

    def __call__(self, rows: Sequence[Mapping[str, object]], path: Path, *, overwrite: bool) -> Path:
        """Write ``rows`` and return ``path``."""
        ...


class RawPlacement(Protocol):
    """Finds the reference CFA inside a DNG raw image (``raw_pipeline.defects.placement``)."""

    def locate(self, reference: CFAFrame, raw: DngRawImage) -> Any:  # noqa: ANN401 - defects.placement.Placement
        """Return the placement (``offset``, ``method``) or raise ``CoordinateMappingError``."""
        ...

    def visible_frame(self, reference: CFAFrame, raw: DngRawImage, offset: Any) -> CFAFrame:  # noqa: ANN401
        """The DNG photosites under the reference's visible area as a :class:`CFAFrame`."""
        ...


RemainingDefects = Callable[[CFAFrame, DefectMap, DefectDetectionConfig], list[Defect]]
"""Configured defects the detector still finds in a corrected frame (empty when they are gone)."""


class _Placement:
    def locate(self, reference: CFAFrame, raw: DngRawImage) -> Any:  # noqa: ANN401
        return resolve("raw_pipeline.defects.placement", "find_offset")(reference, raw)

    def visible_frame(self, reference: CFAFrame, raw: DngRawImage, offset: Any) -> CFAFrame:  # noqa: ANN401
        return cast("CFAFrame", resolve("raw_pipeline.defects.placement", "visible_frame")(reference, raw, offset))


def _discover_files(input_dir: Path, *, extensions: Collection[str], exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    found = resolve("raw_pipeline.io.discovery", "discover_raw_files")(
        input_dir, exclude_dirs=tuple(exclude_dirs), extensions=extensions
    )
    return cast("list[Path]", found)


def _make_dng_writer() -> DngRawWriter:
    return cast("DngRawWriter", resolve("raw_pipeline.io.dng.writer", "TiffDngRawWriter")())


def _write_report(rows: Sequence[Mapping[str, object]], path: Path, *, overwrite: bool) -> Path:
    return cast("Path", resolve("raw_pipeline.io.dng.report", "write_fix_raw_report")(rows, path, overwrite=overwrite))


def _remaining_defects(frame: CFAFrame, defects: DefectMap, config: DefectDetectionConfig) -> list[Defect]:
    detector_cls = resolve("raw_pipeline.defects.detection", "StatisticalDefectDetector")
    detector = detector_cls.from_config(config)  # type: ignore[attr-defined]
    found = resolve("raw_pipeline.defects.verification", "remaining_defects")(frame, defects, detector)
    return [defect for defect, _ in found]


@dataclass(frozen=True)
class RuntimeServices:
    """Lower-layer collaborators of the CLI commands (injectable for tests).

    Attributes:
        discover: Input discovery.
        write_defect_map: Standalone defect-map writer (``--output``).
        merge_defect_map: Config-file merger (no ``--output``).
        make_defect_detector: Defect detector factory.
        discover_files: Input discovery by suffix (``fix-raw``).
        make_dng_writer: Builds the DNG raw writer (``fix-raw``).
        placement: Reference-to-DNG coordinate mapping (``fix-raw``).
        remaining_defects: Post-correction defect scoring (``fix-raw`` verification).
        write_report: CSV writer of the ``fix-raw`` report.
    """

    discover: InputDiscovery
    write_defect_map: DefectMapWriter
    merge_defect_map: DefectMapMerger
    make_defect_detector: DefectDetectorFactory
    discover_files: FileDiscovery = _discover_files
    make_dng_writer: Callable[[], DngRawWriter] = _make_dng_writer
    placement: RawPlacement = field(default_factory=_Placement)
    remaining_defects: RemainingDefects = _remaining_defects
    write_report: ReportWriter = _write_report

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


def _merge_defect_map(defects: DefectMap, config_path: Path, *, overwrite: bool) -> Path:
    merge = resolve("raw_pipeline.io.defects_yaml", "merge_defect_map_into_config")
    return cast("Path", merge(config_path, defects, overwrite=overwrite))


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
