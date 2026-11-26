# Plan: corrected DNG export (`fix-raw`)

Single session (focused feature); contracts first, then implementation.

| Package | Owns | Depends on | Acceptance |
| --- | --- | --- | --- |
| Contracts | `core/dng.py`, `DngSource`/`DngRawWriter` in `core/protocols.py`, errors, `dng_sources` registry, `raw_export` config, ADR 0003 | ADR 0002 | mypy/import-linter clean; config validates source names |
| DNG adapters | `io/dng/` (tiff, writer, digest, preview, sources, report) | contracts | strips/tiles, in-place patch, digest/preview policy, rejection of compressed/linear/non-16-bit; `DngSource` contract suite |
| Mapping + verification | `defects/placement.py`, `defects/verification.py` | contracts | exact offset with Bayer-phase check, no guessing; detector scoring |
| Service + CLI | `runtime/raw_export.py`, `runtime/app.py`, `cli.py` | all above | per-image failures, overwrite rule, CSV report, verification |
| Docs + CI | `docs/`, `config.example.yaml`, CI DNGLab job | all | strict docs build; slow DNGLab test green in CI |
