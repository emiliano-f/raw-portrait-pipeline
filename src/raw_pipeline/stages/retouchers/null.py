"""``none`` retoucher: the Null Object used when retouching is disabled."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.registries import retoucher_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import FaceRegion, RetouchResult, freeze_record


@retoucher_registry.register("none")
@dataclass(frozen=True)
class NullRetoucher:
    """Returns the input image unchanged (same array object)."""

    name: ClassVar[str] = "none"

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:  # noqa: ARG003 - registry factory signature
        """Build the null retoucher; it has no settings."""
        return cls()

    def retouch(self, image: NDArray[np.float32], faces: Sequence[FaceRegion]) -> RetouchResult:  # noqa: ARG002
        """Return ``image`` untouched with zero strength and coverage recorded."""
        return RetouchResult(
            image=image,
            params=freeze_record({"strength": 0.0, "mask_coverage": 0.0, "faces_retouched": 0, "mask_source": "none"}),
        )
