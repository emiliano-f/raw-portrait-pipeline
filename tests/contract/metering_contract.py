"""Contract every :class:`~raw_pipeline.core.protocols.MeteringStrategy` must satisfy."""

from __future__ import annotations

import math

from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.protocols import MeteringStrategy
from tests.support.synthetic import decoded_context, flat_image


class MeteringContract:
    """Reusable suite. Override ``make_strategy`` (target 0.18, max correction 2 EV)."""

    target: float = 0.18
    max_correction_ev: float = 2.0

    def make_strategy(self) -> MeteringStrategy:
        raise NotImplementedError

    def make_context(self, value: float) -> ImageContext:
        return decoded_context(flat_image(value))

    def test_target_needs_no_correction(self) -> None:
        m = self.make_strategy().measure(self.make_context(self.target))
        assert abs(m.correction_ev) < 0.05

    def test_one_stop_under(self) -> None:
        m = self.make_strategy().measure(self.make_context(self.target / 2))
        assert math.isclose(m.correction_ev, 1.0, abs_tol=0.05)

    def test_clamped(self) -> None:
        m = self.make_strategy().measure(self.make_context(self.target / 64))
        assert m.correction_ev <= self.max_correction_ev + 1e-6

    def test_finite_on_black(self) -> None:
        m = self.make_strategy().measure(self.make_context(0.0))
        assert math.isfinite(m.correction_ev)
