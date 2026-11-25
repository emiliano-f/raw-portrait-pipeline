"""``fix-raw`` end to end on the sample CR2s with the real converters (slow, skipped without them)."""

from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import rawpy

from raw_pipeline.cli import main
from raw_pipeline.io.dng.tiff import read_raw
from tests.integration.conftest import sha256

pytestmark = pytest.mark.slow

DEFECT_X = 1509
DEFECT_WIDTH = 2
ADOBE_ENV = "RAW_PIPELINE_ADOBE_DNG_CONVERTER"


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


def _run_and_check(samples: list[Path], tmp_path: Path, config: Path) -> None:
    hashes = {p: sha256(p) for p in samples}
    inputs = tmp_path / "in"
    inputs.mkdir()
    for sample in samples:
        (inputs / sample.name).symlink_to(sample)
    out = tmp_path / "out"
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        assert main(["fix-raw", str(inputs), str(out), "--config", str(config), "-q"]) == 0
    finally:  # the CLI reconfigures the root logger; restore it for later caplog-based tests
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)
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


def test_fix_raw_with_dnglab(samples: list[Path], tmp_path: Path) -> None:
    exe = shutil.which("dnglab")
    if exe is None:
        pytest.skip("dnglab not on PATH")
    _run_and_check(samples, tmp_path, _config(tmp_path, "dnglab"))


def test_fix_raw_with_adobe_converter(samples: list[Path], tmp_path: Path) -> None:
    exe = os.environ.get(ADOBE_ENV)
    if not exe:
        pytest.skip(f"set {ADOBE_ENV} to the Adobe DNG Converter executable")
    _run_and_check(samples, tmp_path, _config(tmp_path, "adobe_converter", exe))
