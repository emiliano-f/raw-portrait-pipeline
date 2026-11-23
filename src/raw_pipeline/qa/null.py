"""The ``none`` QA provider (Null Object): QA disabled."""

from __future__ import annotations

from typing import ClassVar

from raw_pipeline.core.registries import qa_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import QAResult


@qa_registry.register("none")
class NullQAProvider:
    """Never reviews anything and never contacts a service; every result is ``skipped``."""

    name: ClassVar[str] = "none"

    @classmethod
    def from_env(cls, env: BuildEnv) -> NullQAProvider:
        """Build the provider; it has no settings."""
        del env
        return cls()

    def review(self, preview_jpeg: bytes) -> QAResult:
        """Return a skipped result without looking at ``preview_jpeg``."""
        del preview_jpeg
        return QAResult(provider=self.name, skipped=True, notes="QA disabled")
