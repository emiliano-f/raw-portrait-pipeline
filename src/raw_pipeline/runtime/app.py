"""Use cases behind the CLI commands: ``process`` and ``detect-defects``."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from raw_pipeline.config import load_config
from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.errors import ConfigError, PipelineError
from raw_pipeline.core.protocols import PipelineObserver
from raw_pipeline.core.raw import DefectMap
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.batch import BatchResult, BatchRunner, default_workers
from raw_pipeline.runtime.components import ComponentRegistries, load_components
from raw_pipeline.runtime.defect_detection import DefectDetectionResult, DefectDetectionService
from raw_pipeline.runtime.factory import PipelineFactory
from raw_pipeline.runtime.report_files import duplicate_output_stems, report_name_collisions, stale_contact_sheet_pages
from raw_pipeline.runtime.services import RuntimeServices

logger = logging.getLogger(__name__)

DEFECTS_FILE_SUFFIXES: Final = frozenset({".yaml", ".yml"})
"""Allowed ``--output`` suffixes, so a RAW input can never be the destination."""


class InputError(PipelineError):
    """The command line names inputs or outputs that cannot be used; nothing was processed."""


@dataclass(frozen=True)
class ProcessRequest:
    """Arguments of ``raw-pipeline process``.

    Attributes:
        input_dir: Folder of RAW files (read-only).
        output_dir: Folder for outputs and reports (created if missing).
        config_path: YAML config; ``None`` uses the defaults.
        overwrite: Allow replacing existing output files.
        workers: Worker processes; ``None`` uses ``pipeline.workers`` or CPU count - 1.
    """

    input_dir: Path
    output_dir: Path
    config_path: Path | None = None
    overwrite: bool = False
    workers: int | None = None


@dataclass(frozen=True)
class DetectDefectsRequest:
    """Arguments of ``raw-pipeline detect-defects``.

    Attributes:
        input_dir: Folder of RAW files analysed as a batch (ignored with ``dark_frame``).
        config_path: YAML config; ``None`` uses the defaults.
        dark_frame: A single dark frame (lens cap on, same ISO) to analyse instead.
        output: Standalone defects file to write later; checked before any frame is read.
        overwrite: Allow replacing ``output``, or rewriting ``config_path`` when merging.
    """

    input_dir: Path
    config_path: Path | None = None
    dark_frame: Path | None = None
    output: Path | None = None
    overwrite: bool = False


def process_batch(
    request: ProcessRequest,
    *,
    registries: ComponentRegistries | None = None,
    services: RuntimeServices | None = None,
    observer: PipelineObserver | None = None,
) -> BatchResult:
    """Run the configured pipeline on every RAW file of ``request.input_dir``.

    Raises:
        ConfigError: Invalid config or unknown component names (before any image).
        InputError: Missing input folder, output folder equal to the input folder, or no RAW files.
    """
    config = load_config(request.config_path)
    regs = load_components(registries)
    regs.validate(config)
    _check_dirs(request.input_dir, request.output_dir)
    sources = (services or RuntimeServices.builtin()).discover(request.input_dir, exclude_dirs=(request.output_dir,))
    if not sources:
        msg = f"no RAW files found in {request.input_dir}"
        raise InputError(msg)
    duplicates = duplicate_output_stems(sources)
    if duplicates:
        msg = "several inputs share an output name (rename them): " + "; ".join(duplicates)
        raise InputError(msg)
    clashes = report_name_collisions(sources, config)
    if clashes:
        msg = "output names collide with report files (rename the inputs or change report.*_filename): " + "; ".join(
            clashes
        )
        raise InputError(msg)
    workers = request.workers or config.pipeline.workers or default_workers()
    env = BuildEnv(config=config, output_dir=request.output_dir, overwrite=request.overwrite)
    factory = PipelineFactory(regs)
    pipeline = factory.build_pipeline(env)
    reporters = factory.build_reporters(env)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("stages: %s; workers: %d", ", ".join(pipeline.stage_names), workers)
    result = BatchRunner(pipeline, workers, observer).run(sources, reporters, request.output_dir)
    if request.overwrite:
        _warn_stale_pages(request.output_dir, config, result.report_paths)
    return result


def _warn_stale_pages(output_dir: Path, config: AppConfig, written: Sequence[Path]) -> None:
    try:
        stale = stale_contact_sheet_pages(output_dir, config, written)
    except OSError as exc:
        logger.warning("could not check %s for stale contact sheet pages: %s", output_dir, exc)
        return
    for page in stale:
        logger.warning("%s is a contact sheet page from an earlier run; this run did not replace it", page)


def detect_defects(
    request: DetectDefectsRequest,
    *,
    registries: ComponentRegistries | None = None,
    services: RuntimeServices | None = None,
) -> DefectDetectionResult:
    """Detect sensor defects on a dark frame or across the batch. Writes nothing.

    Raises:
        ConfigError: Invalid config or unknown decoder name.
        InputError: Missing inputs, or no frame could be read.
    """
    config = load_config(request.config_path)
    regs = load_components(registries)
    _check_decoder(config.decode.decoder, regs)
    _check_destination(request.output, request.config_path, overwrite=request.overwrite)
    svc = services or RuntimeServices.builtin()
    if request.dark_frame is not None:
        if not request.dark_frame.is_file():
            msg = f"dark frame {request.dark_frame} does not exist"
            raise InputError(msg)
        sources = [request.dark_frame]
    else:
        _check_input_dir(request.input_dir)
        sources = svc.discover(request.input_dir)
    if not sources:
        msg = f"no RAW files found in {request.input_dir}"
        raise InputError(msg)
    env = BuildEnv(config=config, output_dir=Path.cwd())
    service = DefectDetectionService(
        PipelineFactory(regs).build_raw_decoder(env), svc.make_defect_detector(config.defects.detection)
    )
    result = service.detect(sources)
    if not result.frames_used:
        msg = "none of the frames could be read"
        raise InputError(msg)
    return result


def save_defect_map(
    defects: DefectMap,
    *,
    output: Path | None,
    config_path: Path | None,
    overwrite: bool,
    confirm: Callable[[str], bool],
    services: RuntimeServices | None = None,
) -> Path | None:
    """Write detected defects to ``output``, or merge them into the config after confirmation.

    Args:
        defects: Defects to save.
        output: Standalone YAML file; when ``None`` the config file is updated instead.
        config_path: Config file to update when ``output`` is ``None``.
        overwrite: Allow replacing an existing ``output`` file, or rewriting ``config_path``.
        confirm: Asked before changing the config file; returns ``True`` to proceed.
        services: Lower-layer writers (built-ins by default).

    Returns:
        The file written, or ``None`` when the user declined.

    Raises:
        InputError: No usable destination (see ``_check_destination``).
    """
    svc = services or RuntimeServices.builtin()
    _check_destination(output, config_path, overwrite=overwrite)
    if output is not None:
        return svc.write_defect_map(defects, output, overwrite=overwrite)
    assert config_path is not None
    if not confirm(f"Replace defects.map in {config_path} with {len(defects)} detected defect(s)?"):
        return None
    return svc.merge_defect_map(defects, config_path, overwrite=overwrite)


def _check_input_dir(input_dir: Path) -> None:
    if not input_dir.is_dir():
        msg = f"input folder {input_dir} does not exist or is not a folder"
        raise InputError(msg)


def _check_dirs(input_dir: Path, output_dir: Path) -> None:
    _check_input_dir(input_dir)
    if output_dir.resolve() == input_dir.resolve():
        msg = "the output folder must differ from the input folder (inputs are read-only)"
        raise InputError(msg)
    if output_dir.exists() and not output_dir.is_dir():
        msg = f"output path {output_dir} exists and is not a folder"
        raise InputError(msg)


def _check_decoder(name: str, registries: ComponentRegistries) -> None:
    if name not in registries.raw_decoders:
        known = ", ".join(registries.raw_decoders.names()) or "none"
        msg = f"invalid configuration:\n  - decode.decoder: unknown raw_decoders {name!r}; registered: {known}"
        raise ConfigError(msg)


def _check_destination(output: Path | None, config_path: Path | None, *, overwrite: bool) -> None:
    if output is not None:
        if output.suffix.lower() not in DEFECTS_FILE_SUFFIXES:
            msg = f"--output must be a .yaml or .yml file, got {output}"
            raise InputError(msg)
        return
    if config_path is None:
        msg = "pass --output, or --config to update the config file"
        raise InputError(msg)
    if config_path.exists() and not overwrite:
        msg = (
            f"merging rewrites {config_path} and drops its YAML comments; pass --overwrite to allow it, "
            "or --output defects.yaml to write a separate file"
        )
        raise InputError(msg)
