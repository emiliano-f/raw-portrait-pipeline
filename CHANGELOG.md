# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-11-23

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
- Documentation: functional guide (getting started, usage, configuration, stages, QA and
  privacy, troubleshooting), technical guide (architecture, extending, colour pipeline,
  testing) and API reference for every package. The configuration reference, component
  list and import rules are generated from the code at build time, and the strict docs
  build fails when a config field lacks a description or a component is undocumented.
- Slow integration tests (`tests/integration/`) running the CLI end to end on local CR2
  samples (`tests/fixtures/` or `RAW_PIPELINE_SAMPLES`): output size and sRGB ICC,
  clipping, untouched inputs, overwrite refusal, byte-identical reruns, face target.
- CI: lint, format, mypy, import contracts, tests with coverage and the strict docs build.

### Security

- QA over plain HTTP (localhost only) bypasses configured proxies, and an API key echoed by the service is redacted from notes, flags and warnings.
- `--overwrite` temporary files are created exclusively, so a planted symlink is never followed.
- `report.csv_filename` and `report.contact_sheet_filename` must be plain file names, so a report can never replace an input.

### Fixed

- Inputs whose names differ only in case (`IMG_1.CR2`, `img_1.cr2`) are rejected before processing instead of one export replacing the other.
- `export.keep_gps: true` no longer breaks JPEG export for RAW files with GPS coordinates.

[Unreleased]: https://github.com/emiliano-f/raw-portrait-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/emiliano-f/raw-portrait-pipeline/releases/tag/v0.1.0
