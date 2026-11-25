"""The registries of every extension point.

Built-in components register themselves with the decorators below when
their module is imported; ``raw_pipeline.runtime`` imports the built-in
modules and loads entry-point plugins before validating the configuration.
"""

from __future__ import annotations

from typing import Final

from raw_pipeline.core.protocols import (
    BatchReporter,
    DefectCorrector,
    DngSource,
    Exporter,
    FaceDetector,
    MeteringStrategy,
    QAProvider,
    RawDecoder,
    Retoucher,
    Stage,
)
from raw_pipeline.core.registry import Registry

stage_registry: Final = Registry[Stage]("stages")
raw_decoder_registry: Final = Registry[RawDecoder]("raw_decoders")
defect_corrector_registry: Final = Registry[DefectCorrector]("defect_correctors")
dng_source_registry: Final = Registry[DngSource]("dng_sources")
face_detector_registry: Final = Registry[FaceDetector]("detectors")
metering_registry: Final = Registry[MeteringStrategy]("metering")
retoucher_registry: Final = Registry[Retoucher]("retouchers")
exporter_registry: Final = Registry[Exporter]("exporters")
qa_registry: Final = Registry[QAProvider]("qa")
reporter_registry: Final = Registry[BatchReporter]("reporters")

ALL_REGISTRIES: Final = (
    stage_registry,
    raw_decoder_registry,
    defect_corrector_registry,
    dng_source_registry,
    face_detector_registry,
    metering_registry,
    retoucher_registry,
    exporter_registry,
    qa_registry,
    reporter_registry,
)
