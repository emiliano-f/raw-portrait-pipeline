"""Command-line entry point. Argument parsing only; delegates to ``raw_pipeline.runtime``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TextIO

from raw_pipeline.config.defects import entry_to_yaml_dict, to_entry
from raw_pipeline.core.errors import ConfigError, PipelineError
from raw_pipeline.core.protocols import PipelineObserver
from raw_pipeline.core.raw import DetectedDefect
from raw_pipeline.runtime.app import (
    DetectDefectsRequest,
    InputError,
    ProcessRequest,
    detect_defects,
    process_batch,
    save_defect_map,
)
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.observers import CompositeObserver, LoggingObserver, ProgressObserver
from raw_pipeline.runtime.services import RuntimeServices

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
"""At least one image (or reporter, or detection) failed."""
EXIT_USAGE: Final = 2
"""Invalid arguments, config or inputs: nothing was processed."""

_LOG_FORMAT: Final = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``raw-pipeline``."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML configuration file (defaults if omitted).")
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="count", default=0, help="More logging (-vv for debug).")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Errors only, no progress output.")

    parser = argparse.ArgumentParser(prog="raw-pipeline", description="Batch-process Canon CR2 portraits.")
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", parents=[common], help="Process a folder of RAW files.")
    process.add_argument("input_dir", type=Path, help="Folder of .CR2 files (never modified).")
    process.add_argument("output_dir", type=Path, help="Folder for images and reports.")
    process.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    process.add_argument("--workers", type=_positive_int, default=None, help="Parallel worker processes.")

    detect = commands.add_parser("detect-defects", parents=[common], help="Find hot/stuck rows, columns and pixels.")
    detect.add_argument("input_dir", type=Path, help="Folder of .CR2 files analysed as a batch.")
    detect.add_argument("--dark-frame", type=Path, default=None, help="Analyse this dark frame instead of the batch.")
    detect.add_argument("--output", type=Path, default=None, help="Write the defects to this YAML file.")
    detect.add_argument("--overwrite", action="store_true", help="Replace an existing --output file.")
    detect.add_argument("--yes", action="store_true", help="Update the config file without asking.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registries: ComponentRegistries | None = None,
    services: RuntimeServices | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    ask: Callable[[str], str] = input,
) -> int:
    """Run the ``raw-pipeline`` command and return the process exit code.

    Keyword arguments exist for tests and embedding; the console script uses the defaults.
    """
    args = build_parser().parse_args(argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    _configure_logging(args.verbose, quiet=args.quiet, stream=err)
    try:
        if args.command == "process":
            return _process(args, registries, services, err)
        return _detect(args, registries, services, out, ask)
    except (ConfigError, InputError) as exc:
        err.write(f"error: {exc}\n")
        return EXIT_USAGE
    except PipelineError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_FAILED


def _process(
    args: argparse.Namespace, registries: ComponentRegistries | None, services: RuntimeServices | None, err: TextIO
) -> int:
    observers: list[PipelineObserver] = [LoggingObserver()]
    if not args.quiet:
        observers.append(ProgressObserver(err))
    request = ProcessRequest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    result = process_batch(request, registries=registries, services=services, observer=CompositeObserver(observers))
    for path in result.report_paths:
        err.write(f"report: {path}\n")
    return EXIT_OK if result.ok else EXIT_FAILED


def _detect(
    args: argparse.Namespace,
    registries: ComponentRegistries | None,
    services: RuntimeServices | None,
    out: TextIO,
    ask: Callable[[str], str],
) -> int:
    request = DetectDefectsRequest(input_dir=args.input_dir, config_path=args.config, dark_frame=args.dark_frame)
    result = detect_defects(request, registries=registries, services=services)
    out.write(f"analysed {len(result.frames_used)} frame(s), {len(result.frames_failed)} unreadable\n")
    if not result.detected:
        out.write("no defects found; nothing written\n")
        return EXIT_OK
    out.write(f"found {len(result.detected)} defect(s) (raw sensor coordinates):\n")
    for found in result.detected:
        out.write(f"  {_describe(found)}\n")

    def confirm(question: str) -> bool:
        if args.yes:
            return True
        try:
            return ask(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
        except EOFError:
            return False

    written = save_defect_map(
        result.defect_map,
        output=args.output,
        config_path=args.config,
        overwrite=args.overwrite,
        confirm=confirm,
        services=services,
    )
    out.write(f"wrote defects.map to {written}\n" if written else "not written\n")
    return EXIT_OK


def _describe(found: DetectedDefect) -> str:
    fields = ", ".join(f"{k}: {v}" for k, v in entry_to_yaml_dict(to_entry(found.defect)).items())
    return f"{{{fields}}}  score {found.score:.1f}, in {found.frames_flagged}/{found.frames_total} frame(s)"


def _configure_logging(verbose: int, *, quiet: bool, stream: TextIO) -> None:
    level = logging.ERROR if quiet else (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=stream, force=True)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        msg = f"must be >= 1, got {number}"
        raise argparse.ArgumentTypeError(msg)
    return number
