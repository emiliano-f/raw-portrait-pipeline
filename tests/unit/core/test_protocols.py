from __future__ import annotations

from pathlib import Path

from raw_pipeline.core.protocols import ImageOutcome, Stage
from raw_pipeline.core.values import ImageMetadata
from tests.contract.stage_contract import StageContract
from tests.unit.core.test_registry import _Echo


def test_image_outcome_ok_flag() -> None:
    assert ImageOutcome(Path("a.CR2")).ok
    failed = ImageOutcome(Path("a.CR2"), error="boom", error_type="DecodeError")
    assert not failed.ok
    assert failed.metadata == ImageMetadata()


def test_stage_protocol_runtime_check() -> None:
    assert isinstance(_Echo(1), Stage)


class _RecordingIdentity:
    name = "identity"

    def process(self, ctx):  # type: ignore[no-untyped-def]
        return ctx.with_image(ctx.image.copy()).with_record(self.name, {"ok": True})


class TestStageContractSelfCheck(StageContract):
    """The stage contract suite itself passes for a well-behaved stage."""

    def make_stage(self) -> Stage:
        return _RecordingIdentity()
