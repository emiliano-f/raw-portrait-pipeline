"""Contract every :class:`~raw_pipeline.core.protocols.Stage` must satisfy.

Usage in a stage's test module::

    class TestToneStageContract(StageContract):
        def make_stage(self) -> Stage: return ToneStage(...)
"""

from __future__ import annotations

import pickle

import numpy as np

from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.protocols import Stage
from tests.support.synthetic import decoded_context


class StageContract:
    """Reusable suite. Override ``make_stage`` and, if needed, ``make_context``."""

    preserves_shape: bool = True

    def make_stage(self) -> Stage:
        raise NotImplementedError

    def make_context(self) -> ImageContext:
        return decoded_context()

    def test_has_name(self) -> None:
        assert isinstance(self.make_stage().name, str)
        assert self.make_stage().name

    def test_returns_float32_rgb(self) -> None:
        ctx = self.make_context()
        out = self.make_stage().process(ctx)
        assert out.image.dtype == np.float32
        assert out.image.ndim == 3
        assert out.image.shape[2] == 3
        if self.preserves_shape:
            assert out.image.shape == ctx.image.shape

    def test_does_not_mutate_input(self) -> None:
        ctx = self.make_context()
        before = ctx.image.copy()
        records = dict(ctx.metadata.records)
        artifacts = dict(ctx.artifacts)
        self.make_stage().process(ctx)
        np.testing.assert_array_equal(ctx.image, before)
        assert dict(ctx.metadata.records) == records
        assert dict(ctx.artifacts) == artifacts

    def test_records_parameters(self) -> None:
        stage = self.make_stage()
        out = stage.process(self.make_context())
        assert stage.name in out.metadata.records

    def test_no_nan_or_inf(self) -> None:
        out = self.make_stage().process(self.make_context())
        assert np.isfinite(out.image).all()

    def test_deterministic(self) -> None:
        stage = self.make_stage()
        a = stage.process(self.make_context())
        b = stage.process(self.make_context())
        np.testing.assert_array_equal(a.image, b.image)
        assert a.metadata == b.metadata

    def test_picklable(self) -> None:
        stage = self.make_stage()
        clone = pickle.loads(pickle.dumps(stage))
        np.testing.assert_array_equal(
            clone.process(self.make_context()).image, stage.process(self.make_context()).image
        )
