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
- Export and report: `jpeg` and 16-bit `tiff` exporters (sRGB ICC, descriptive EXIF,
  GPS stripped by default, no silent overwrite, byte-identical reruns), `export` and
  `report` stages, `csv` and `contact_sheet` batch reporters.
