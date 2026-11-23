# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Core contracts: `ImageContext`, value objects, protocols for every extension point
  (including the raw-level `DefectCorrector`, ADR 0002), registries with entry-point
  plugin discovery, error hierarchy, shared colour helpers (`core.color`).
- Configuration models, YAML loading and component-name validation; `config.example.yaml`.
- Contract test suites for stages, defect correctors, exporters, face detectors,
  metering strategies and QA providers.
- Decode and IO: rawpy/LibRaw decoder with in-memory raw-level defect correction, EXIF
  reader, RAW discovery, output path planning, defect-map YAML, and the `decode` stage.
- Raw-level defect correction (`defects`): `none`, `neighbor_average` and `directional`
  correctors on the CFA mosaic, and `StatisticalDefectDetector` for hot/stuck columns,
  rows and pixels from dark frames or a batch.
- Detection and metering: `meter` stage with MediaPipe Tasks face detection + 478-point
  mesh (vendored Apache-2.0 models, offline), OpenCV Haar fallback, `face_weighted` and
  `center_weighted` metering with EV clamp.
- `tone` stage (EV exposure, C1 highlight shoulder and shadow lift on luminance) and
  `retouch` stage with `none` and `frequency_separation` retouchers (face-mesh skin mask).
- Export and report: `jpeg` and 16-bit `tiff` exporters (sRGB ICC, descriptive EXIF,
  GPS stripped by default, no silent overwrite, byte-identical reruns), `export` and
  `report` stages, `csv` and `contact_sheet` batch reporters.
- Optional vision-LLM QA: `qa` stage (EXIF-free JPEG preview, long edge <= 1024 px),
  `none` and `openai_compatible` providers (standard-library HTTP, HTTPS only except
  localhost; `qa.endpoint` and `qa.model` required, no vendor defaults); missing key or
  API errors skip QA with a warning; flags in `qa.*` CSV columns.
- Runtime and CLI: component discovery, pipeline factory, serial/parallel batch runner,
  observers, defect detection service, `raw-pipeline process` and `raw-pipeline detect-defects`.
