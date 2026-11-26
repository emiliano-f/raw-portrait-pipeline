"""``fix-raw`` end to end with the real converters on the committed dark frame and local portraits (slow)."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import rawpy

from raw_pipeline.io.dng.tiff import read_raw
from tests.integration.conftest import DARK_FRAME, run_cli, sample_files, sha256

pytestmark = pytest.mark.slow

DEFECT_X = 1509
DEFECT_WIDTH = 2
ADOBE_ENV = "RAW_PIPELINE_ADOBE_DNG_CONVERTER"
LINE_FOUND_DN = 1000
"""Median step (raw DN) between the hot column and its same-colour neighbour in the dark frame."""
LINE_GONE_DN = 8
"""Largest median step left after correction (dark-frame noise is a few DN)."""


def _config(tmp_path: Path, source: str, executable: str | None = None) -> Path:
    path = tmp_path / f"{source}.yaml"
    exe = f"    executable: {executable!r}\n" if executable else ""
    path.write_text(
        "defects:\n  method: directional\n"
        f"  map:\n    - {{type: column, index: {DEFECT_X}, width: {DEFECT_WIDTH}}}\n"
        f"raw_export:\n  source: {source}\n  converter:\n{exe}    timeout_s: 300\n",
        encoding="utf-8",
    )
    return path


def _run_and_check(samples: list[Path], tmp_path: Path, config: Path) -> list[dict[str, str]]:
    hashes = {p: sha256(p) for p in samples}
    inputs = tmp_path / "in"
    inputs.mkdir()
    for sample in samples:
        (inputs / sample.name).symlink_to(sample)
    out = tmp_path / "out"
    assert run_cli(["fix-raw", str(inputs), str(out), "--config", str(config), "-q"]) == 0
    with (out / "fix_raw_report.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(samples)
    for row in rows:
        assert row["status"] == "ok", row["error"]
        assert row["verification"] == "passed"
        assert int(row["corrected_pixels"]) > 0
        output = Path(row["output_file"])
        with rawpy.imread(str(output)) as raw:
            assert raw.raw_image_visible.size > 0
        with rawpy.imread(row["reference_raw"]) as ref:
            visible = ref.raw_image_visible.copy()
        data = read_raw(output).data
        y, x = int(row["offset_y"]), int(row["offset_x"])
        window = data[y : y + visible.shape[0], x : x + visible.shape[1]]
        keep = np.ones(visible.shape, bool)
        keep[:, DEFECT_X : DEFECT_X + DEFECT_WIDTH] = False
        np.testing.assert_array_equal(window[keep], visible[keep])
    assert {p: sha256(p) for p in samples} == hashes
    return rows


def _dnglab() -> None:
    if shutil.which("dnglab") is None:
        pytest.skip("dnglab not on PATH")


def test_dark_frame_line_is_gone_with_dnglab(tmp_path: Path) -> None:
    _dnglab()
    (row,) = _run_and_check([DARK_FRAME], tmp_path, _config(tmp_path, "dnglab"))
    assert row["gps"] == "absent"
    with rawpy.imread(str(DARK_FRAME)) as ref:
        before = ref.raw_image_visible.astype(np.int32)
    data = read_raw(Path(row["output_file"])).data.astype(np.int32)
    y, x = int(row["offset_y"]), int(row["offset_x"])
    after = data[y : y + before.shape[0], x : x + before.shape[1]]

    def step(frame: np.ndarray, col: int) -> float:  # against the same CFA colour two columns away
        return float(np.median(np.abs(frame[:, col] - frame[:, col - 2])))

    assert step(before, DEFECT_X) > LINE_FOUND_DN
    for col in range(DEFECT_X, DEFECT_X + DEFECT_WIDTH):
        assert step(after, col) <= LINE_GONE_DN


def test_fix_raw_portraits_with_dnglab(samples: list[Path], tmp_path: Path) -> None:
    _dnglab()
    _run_and_check(samples, tmp_path, _config(tmp_path, "dnglab"))


def test_fix_raw_with_adobe_converter(tmp_path: Path) -> None:
    samples = sample_files()
    exe = os.environ.get(ADOBE_ENV)
    if not exe:
        pytest.skip(f"set {ADOBE_ENV} to the Adobe DNG Converter executable")
    _run_and_check([DARK_FRAME, *samples], tmp_path, _config(tmp_path, "adobe_converter", exe))
