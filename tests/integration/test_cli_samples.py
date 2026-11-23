"""End-to-end ``raw-pipeline process`` on the sample CR2s (slow, skipped without samples)."""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

import numpy as np
import pytest
import rawpy
import tifffile
from PIL import Image, ImageCms

from raw_pipeline.cli import EXIT_FAILED, EXIT_OK
from tests.integration.conftest import CliRun, run_process, sha256

pytestmark = pytest.mark.slow

MAX_CLIPPED_FRACTION = 0.02
"""Highest acceptable fraction of output pixels with a channel at full scale."""

ICC_PROFILE_TAG = 34675
LIBRAW_TRANSPOSED_FLIPS = frozenset({5, 6})


def expected_size(raw_path: Path) -> tuple[int, int]:
    """(width, height) LibRaw develops ``raw_path`` to, after the camera orientation."""
    with rawpy.imread(str(raw_path)) as raw:
        sizes = raw.sizes
    if sizes.flip in LIBRAW_TRANSPOSED_FLIPS:
        return sizes.height, sizes.width
    return sizes.width, sizes.height


def icc_description(profile: bytes) -> str:
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip()


def report_rows(output_dir: Path) -> dict[str, dict[str, str]]:
    with (output_dir / "report.csv").open(encoding="utf-8", newline="") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def tree_bytes(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_every_sample_is_processed(first_run: CliRun, samples: list[Path]) -> None:
    assert first_run.exit_code == EXIT_OK
    rows = report_rows(first_run.output_dir)
    assert sorted(rows) == sorted(p.name for p in samples)
    assert {row["status"] for row in rows.values()} == {"ok"}


def test_outputs_have_expected_size_and_srgb_profile(first_run: CliRun, samples: list[Path]) -> None:
    for sample in samples:
        width, height = expected_size(sample)
        jpeg_path = first_run.output_dir / f"{sample.stem}.jpg"
        with Image.open(jpeg_path) as jpeg:
            assert jpeg.format == "JPEG"
            assert jpeg.mode == "RGB"
            assert jpeg.size == (width, height)
            assert "sRGB" in icc_description(jpeg.info["icc_profile"])
        with tifffile.TiffFile(first_run.output_dir / f"{sample.stem}.tif") as tiff:
            page = tiff.pages[0]
            assert page.shape == (height, width, 3)
            assert page.dtype == np.uint16
            assert "sRGB" in icc_description(bytes(page.tags[ICC_PROFILE_TAG].value))


def test_clipped_fraction_stays_below_threshold(first_run: CliRun, samples: list[Path]) -> None:
    rows = report_rows(first_run.output_dir)
    for sample in samples:
        assert float(rows[sample.name]["tone.clipped_fraction"]) < MAX_CLIPPED_FRACTION
        with Image.open(first_run.output_dir / f"{sample.stem}.jpg") as jpeg:
            pixels = np.asarray(jpeg)
        assert float((pixels == np.iinfo(np.uint8).max).any(axis=-1).mean()) < MAX_CLIPPED_FRACTION


def test_inputs_are_unchanged(first_run: CliRun) -> None:
    assert first_run.exit_code == EXIT_OK
    assert {path: sha256(path) for path in first_run.hashes_before} == first_run.hashes_before


def test_output_is_byte_identical_across_runs(first_run: CliRun, second_run: CliRun) -> None:
    assert second_run.exit_code == EXIT_OK
    first, second = tree_bytes(first_run.output_dir), tree_bytes(second_run.output_dir)
    assert sorted(first) == sorted(second)
    assert [name for name in first if first[name] != second[name]] == []


def test_rerun_without_overwrite_refuses_and_keeps_outputs(
    first_run: CliRun, config_file: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    shutil.copytree(first_run.output_dir, output_dir)
    before = tree_bytes(output_dir)

    assert run_process(output_dir, config_file) == EXIT_FAILED
    assert tree_bytes(output_dir) == before

    assert run_process(output_dir, config_file, "--overwrite") == EXIT_OK
    assert tree_bytes(output_dir) == before
