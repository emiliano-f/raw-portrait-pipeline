"""Paths of the RAW fixtures shared by unit and integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Final

FIXTURES_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures"
DARK_FRAME: Final = FIXTURES_DIR / "dark" / "dark_iso400.CR2"
"""Committed lens-cap-on EOS 800D CR2 with the hot column at visible x = 1509; no personal data."""
FIXTURES_ENV: Final = "RAW_PIPELINE_FIXTURES_DIR"
"""Environment variable naming a local folder of real portrait CR2s (never committed)."""
