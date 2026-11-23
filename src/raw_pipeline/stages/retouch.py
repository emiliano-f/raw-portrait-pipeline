"""Retouch stage: delegates to the configured :class:`~raw_pipeline.core.protocols.Retoucher`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.protocols import Retoucher
from raw_pipeline.core.registries import retoucher_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv


@stage_registry.register("retouch")
@dataclass(frozen=True)
class RetouchStage:
    """Runs a retoucher on the faces found by the meter stage (``FACES``).

    With no faces the retoucher still runs and is expected to return the
    image unchanged. Records ``retouch.method`` plus the retoucher's parameters.

    Attributes:
        retoucher: Strategy that edits skin; ``none`` disables retouching.
    """

    name: ClassVar[str] = "retouch"
    retoucher: Retoucher

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build with the retoucher named by ``retouch.method``."""
        return cls(retoucher_registry.create(app_config(env).retouch.method, env))

    def process(self, ctx: ImageContext) -> ImageContext:
        """Return ``ctx`` with skin retouched and the parameters recorded."""
        faces = ctx.artifact(FACES)
        result = self.retoucher.retouch(ctx.image, faces if faces is not None else ())
        return ctx.with_image(result.image).with_record(self.name, {"method": self.retoucher.name, **result.params})
