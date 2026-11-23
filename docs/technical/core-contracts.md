# Core contracts

The stable API lives in `raw_pipeline.core` (see [ADR 0001](../adr/0001-architecture-and-core-contracts.md)
and [ADR 0002](../adr/0002-raw-level-defect-correction.md)). Changing anything here needs a dedicated PR with an ADR.

| Protocol | Registry kind | Built-in names | Guarantees (contract suite) |
|---|---|---|---|
| `Stage` | `stages` | decode, meter, tone, retouch, export, report, qa | new context, input untouched, float32 `(H, W, 3)`, records params, deterministic, picklable (`stage_contract.py`) |
| `RawDecoder` / `RawHandle` | `raw_decoders` | rawpy | read-only, visible-area CFA, scene-linear develop, sensor→image mapping |
| `DefectCorrector` | `defect_correctors` | none, neighbor_average, directional | input untouched, only masked photosites change, same layout, `none` is passthrough (`defect_corrector_contract.py`) |
| `DefectDetector` | (injected) | — | streams frames, returns `DetectedDefect`s in sensor coordinates |
| `FaceDetector` | `detectors` | mediapipe, opencv | preview coordinates, empty list when no face (`face_detector_contract.py`) |
| `MeteringStrategy` | `metering` | face_weighted, center_weighted | correction in EV, clamped, finite (`metering_contract.py`) |
| `Retoucher` | `retouchers` | none, frequency_separation | skin mask only, no geometry change, no generative models |
| `Exporter` | `exporters` | jpeg, tiff | applies transfer function, ICC embedded, no silent overwrite, byte-identical (`exporter_contract.py`) |
| `QAProvider` | `qa` | none, openai_compatible | never raises, preview ≤ 1024 px only (`qa_provider_contract.py`) |
| `BatchReporter` | `reporters` | csv, contact_sheet | consumes `ImageOutcome`s, writes into the output dir |
| `PipelineObserver` | (injected) | — | progress and per-stage timing |

## Shared artifacts

| Key | Type | Producer | Consumers |
|---|---|---|---|
| `FACES` | `tuple[FaceRegion, ...]` (full-res coords) | meter | metering, retouch |
| `EXPOSURE` | `ExposureMeasurement` | meter | tone |
| `BEFORE_IMAGE` | downscaled scene-linear float32 | decode | report |
| `DEFECT_CROPS` | `DefectCrops` | decode | report |
| `OUTPUT_PATHS` | `tuple[Path, ...]` | export | runtime (outcome) |
| `REPORT_THUMBNAILS` | `Mapping[str, uint8 array]` | report | runtime → reporters |
| `QA_RESULT` | `QAResult` | qa | report CSV (via metadata) |

Luminance (Rec.709 weights) and the sRGB transfer function are defined once in
`raw_pipeline.core.color`; use them instead of re-implementing.

Preview coordinates are mapped to full resolution only through
`raw_pipeline.core.geometry.preview_to_full`.
