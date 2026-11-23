from __future__ import annotations

from pathlib import Path

from raw_pipeline.config import AppConfig
from raw_pipeline.core.protocols import QAProvider
from raw_pipeline.core.registries import qa_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.qa.null import NullQAProvider
from tests.contract.qa_provider_contract import QAProviderContract


class TestNullQAProviderContract(QAProviderContract):
    def make_provider(self) -> QAProvider:
        return NullQAProvider()


def test_null_provider_is_skipped(tiny_jpeg: bytes) -> None:
    result = NullQAProvider().review(tiny_jpeg)
    assert result.skipped
    assert result.flags == frozenset()
    assert result.provider == "none"


def test_registered_as_none() -> None:
    env = BuildEnv(config=AppConfig(), output_dir=Path("out"))
    assert isinstance(qa_registry.create("none", env), NullQAProvider)
