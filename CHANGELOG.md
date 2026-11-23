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
- Optional vision-LLM QA: `qa` stage (EXIF-free JPEG preview, long edge <= 1024 px),
  `none` and `openai_compatible` providers (standard-library HTTP, HTTPS only except
  localhost); missing key or API errors skip QA with a warning; flags in `qa.*` CSV columns.
