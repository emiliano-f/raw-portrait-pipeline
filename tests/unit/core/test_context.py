from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.core.artifacts import EXPOSURE, FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.values import ExposureMeasurement
from tests.support.synthetic import gradient_image


def test_initial_context_is_not_decoded() -> None:
    ctx = ImageContext.initial(Path("a.CR2"))
    assert not ctx.is_decoded
    assert ctx.image.shape == (0, 0, 3)
    assert ctx.image.dtype == np.float32


def test_with_image_validates_shape_and_dtype() -> None:
    ctx = ImageContext.initial(Path("a.CR2"))
    assert ctx.with_image(gradient_image()).is_decoded
    with pytest.raises(ValueError, match="float32"):
        ctx.with_image(np.zeros((4, 4, 3), dtype=np.float64))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="float32"):
        ctx.with_image(np.zeros((4, 4), dtype=np.float32))


def test_with_helpers_do_not_mutate_original() -> None:
    ctx = ImageContext.initial(Path("a.CR2")).with_image(gradient_image())
    m = ExposureMeasurement(0.09, 0.18, 1.0, "fake")
    new = ctx.with_record("meter", {"ev": 1.0}).with_artifact(EXPOSURE, m)
    assert ctx.metadata.records == {}
    assert ctx.artifact(EXPOSURE) is None
    assert new.artifact(EXPOSURE) == m
    assert new.metadata.records["meter"]["ev"] == 1.0
    assert new.artifact(FACES) is None
    assert new.source_path == ctx.source_path


def test_artifacts_mapping_is_read_only() -> None:
    ctx = ImageContext.initial(Path("a.CR2")).with_artifact(FACES, ())
    with pytest.raises(TypeError):
        ctx.artifacts["x"] = 1  # type: ignore[index]
