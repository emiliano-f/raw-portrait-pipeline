"""``none``: the Null Object defect corrector."""

from __future__ import annotations

from typing import ClassVar, Self

from raw_pipeline.core.raw import CFAFrame, DefectCorrectionResult, DefectMap
from raw_pipeline.core.registries import defect_corrector_registry
from raw_pipeline.core.registry import BuildEnv


@defect_corrector_registry.register("none")
class NullDefectCorrector:
    """Passthrough: returns the input frame object itself and reports 0 corrected photosites.

    The defect map is ignored (not even validated), so ``method: none`` is a
    fully supported way to keep a map in the config while disabling correction.
    """

    name: ClassVar[str] = "none"

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:  # noqa: ARG003 - no settings to read
        """Build the corrector; it has no configuration."""
        return cls()

    def correct(self, frame: CFAFrame, defects: DefectMap) -> DefectCorrectionResult:  # noqa: ARG002 - Null Object
        """Return ``frame`` unchanged."""
        return DefectCorrectionResult(frame=frame, corrected_pixel_count=0, method=self.name)
