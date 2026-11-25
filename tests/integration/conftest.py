"""Fixture discovery and shared CLI runs for the slow integration tests.

Two kinds of fixtures:

- :data:`DARK_FRAME`, a committed lens-cap-on EOS 800D CR2 with the sensor's
  hot column (no personal data). Defect detection and ``fix-raw`` tests use it,
  in CI too.
- Real portraits, which are personal data and never committed: they are read
  from the folder named by ``$RAW_PIPELINE_FIXTURES_DIR``. Tests that need them
  skip when it is unset.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from raw_pipeline.cli import main
from tests.support.fixtures import DARK_FRAME, FIXTURES_ENV

__all__ = ["DARK_FRAME", "FIXTURES_ENV", "run_cli", "sample_files", "sha256"]


def samples_dir() -> Path | None:
    """Local folder of real portrait CR2s, or ``None`` when ``$RAW_PIPELINE_FIXTURES_DIR`` is unset."""
    configured = os.environ.get(FIXTURES_ENV)
    return Path(configured).expanduser().resolve() if configured else None


def sample_files() -> list[Path]:
    """Portrait CR2s (any case), sorted; empty when there are none."""
    directory = samples_dir()
    if directory is None or not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".cr2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CliRun:
    """One ``raw-pipeline process`` run over the sample directory."""

    output_dir: Path
    exit_code: int
    hashes_before: dict[Path, str]


def run_process(output_dir: Path, config: Path, *extra: str) -> int:
    """Run ``raw-pipeline process`` in-process on the sample directory, one worker.

    The CLI reconfigures the root logger; it is restored so later tests still capture logs.
    """
    return run_cli(
        ["process", str(samples_dir()), str(output_dir), "--config", str(config), "--workers", "1", "-q", *extra]
    )


def run_cli(argv: list[str]) -> int:
    """Run the CLI in-process, restoring the root logger it reconfigures."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        return main(argv)
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture(scope="session")
def samples() -> list[Path]:
    found = sample_files()
    if not found:
        pytest.skip(f"no portrait CR2s: set {FIXTURES_ENV} to a local folder of them")
    return found


@pytest.fixture(scope="session")
def config_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Defaults plus a 16-bit TIFF next to the JPEG, so both exporters are covered."""
    path = tmp_path_factory.mktemp("config") / "config.yaml"
    path.write_text("export:\n  formats: [jpeg, tiff]\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def first_run(samples: list[Path], config_file: Path, tmp_path_factory: pytest.TempPathFactory) -> CliRun:
    hashes = {path: sha256(path) for path in samples}
    output_dir = tmp_path_factory.mktemp("run1") / "out"
    return CliRun(output_dir, run_process(output_dir, config_file), hashes)


@pytest.fixture(scope="session")
def second_run(first_run: CliRun, config_file: Path, tmp_path_factory: pytest.TempPathFactory) -> CliRun:
    output_dir = tmp_path_factory.mktemp("run2") / "out"
    return CliRun(output_dir, run_process(output_dir, config_file), first_run.hashes_before)
