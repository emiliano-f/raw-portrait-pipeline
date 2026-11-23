"""Shared fixtures."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()
