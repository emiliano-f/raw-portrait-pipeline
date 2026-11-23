"""Contract every :class:`~raw_pipeline.core.protocols.QAProvider` must satisfy (no network)."""

from __future__ import annotations

from raw_pipeline.core.protocols import QAProvider
from raw_pipeline.core.values import QAResult


class QAProviderContract:
    """Reusable suite. Override ``make_provider`` with a fake transport."""

    def make_provider(self) -> QAProvider:
        raise NotImplementedError

    def test_returns_result(self, tiny_jpeg: bytes) -> None:
        result = self.make_provider().review(tiny_jpeg)
        assert isinstance(result, QAResult)
        assert result.provider == self.make_provider().name
