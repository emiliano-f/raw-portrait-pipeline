# ADR 0001: Architecture and core contracts

- Status: accepted
- Date: 2026-11-23

## Context

The pipeline must be extensible without editing existing code, run offline,
be deterministic and process images in parallel without holding the batch in
memory (AGENTS.md). Several agents build the packages in parallel, so the
contracts must exist before any implementation.

## Decision

- `raw_pipeline.core` holds the contracts: `ImageContext` (frozen; `with_*`
  helpers return copies), value objects (`ImageMetadata`, `ExifInfo`,
  `FaceRegion`, `ExposureMeasurement`, `QAResult`, ...), protocols for every
  extension point, typed artifact keys, the error hierarchy and one `Registry`
  per extension point.
- Components are classes registered with `@<kind>_registry.register(name)` and
  built through a `from_env(env: BuildEnv)` classmethod. Constructors take
  collaborators explicitly (tests inject fakes); `from_env` is the only place
  that reads configuration and resolves sub-components from registries.
- Third-party plugins use the entry-point group `raw_pipeline.<kind>`, where
  `<kind>` is the registry kind (`stages`, `raw_decoders`, `defect_correctors`,
  `detectors`, `metering`, `retouchers`, `exporters`, `qa`, `reporters`).
- Configuration (`raw_pipeline.config`) is a tree of frozen pydantic models.
  Component names are validated against the registries
  (`validate_components`), so unknown names fail before any image is touched,
  listing the registered options. Plugins read `plugins.<name>`.
- Layering, enforced by import-linter:
  `cli → runtime → stages → {metering | detection | exporters | qa | defects | io} → config → core`.
  Packages in the same layer are independent of each other. `core` may not
  import rawpy, OpenCV, MediaPipe, Pillow, tifffile, pydantic, YAML or network
  modules.
- Per-image vs. batch: stages see one image. Batch outputs (CSV, contact sheet)
  are `BatchReporter`s fed with `ImageOutcome`s (metadata + small thumbnails,
  never full-resolution pixels) in the main process.
- Working space: scene-linear float32 RGB with sRGB/Rec.709 primaries (D65);
  1.0 = sensor white after black subtraction and camera WB. Only exporters (and
  the report/QA preview encoders built on the exporters' encoding helper) apply
  the sRGB transfer function.

## Consequences

- New components are added by registering a class; the factory never changes.
- `BuildEnv.config` is typed `object` in `core`; components read it with
  `raw_pipeline.config.app_config(env)`, which checks the type.
- Contract suites in `tests/contract/` are the executable form of the
  guarantees in the protocol docstrings; every implementation subclasses them.
