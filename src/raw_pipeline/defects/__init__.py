"""Raw-level (pre-demosaic) sensor defect correction and detection (ADR 0002).

Importing this package registers the built-in correctors ``none``,
``neighbor_average`` and ``directional`` in
:data:`~raw_pipeline.core.registries.defect_corrector_registry`.
"""

from __future__ import annotations

from raw_pipeline.defects.detection import StatisticalDefectDetector
from raw_pipeline.defects.directional import DirectionalCorrector
from raw_pipeline.defects.neighbor_average import NeighborAverageCorrector
from raw_pipeline.defects.null import NullDefectCorrector

__all__ = ["DirectionalCorrector", "NeighborAverageCorrector", "NullDefectCorrector", "StatisticalDefectDetector"]
