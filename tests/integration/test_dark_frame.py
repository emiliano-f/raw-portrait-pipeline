"""Defect detection on the committed dark frame (slow)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.integration.conftest import DARK_FRAME, run_cli, sha256

pytestmark = pytest.mark.slow

HOT_COLUMNS = (1509, 1510)
"""Columns of the EOS 800D the dark frame was shot with: 1509 stuck high, 1510 weaker."""


def test_detect_defects_finds_the_hot_columns(tmp_path: Path) -> None:
    before = sha256(DARK_FRAME)
    output = tmp_path / "defects.yaml"
    code = run_cli(
        ["detect-defects", str(DARK_FRAME.parent), "--dark-frame", str(DARK_FRAME), "--output", str(output), "-q"]
    )
    assert code == 0
    found = yaml.safe_load(output.read_text(encoding="utf-8"))["defects"]["map"]
    columns = {d["index"] for d in found if d["type"] == "column"}
    assert set(HOT_COLUMNS) <= columns
    assert sha256(DARK_FRAME) == before
