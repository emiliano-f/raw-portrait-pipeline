# ADR 0002: Raw-level (pre-demosaic) defect correction extension point

- Status: accepted
- Date: 2026-11-23

## Context

The camera has a line of hot/stuck photosites. Correcting it after
demosaicing is too late: the demosaic algorithm spreads each bad photosite
into its neighbours (and into the other colour channels), so the line becomes
a 2-3 px wide coloured streak that is hard to remove without blurring. The
original contract only exposed the developed `(H, W, 3)` image to stages.

## Decision

- Add a raw-level extension point that runs **inside decoding, between reading
  the CFA (Bayer) data and demosaicing**:
  - `CFAFrame` (core): visible-area uint16 Bayer data plus pattern, black and
    white levels.
  - `DefectMap` with `ColumnDefect`, `RowDefect` (full or partial via
    `start`/`end`, `width >= 1`) and `PixelDefect`.
  - `DefectCorrector` protocol + `defect_corrector_registry`:
    `correct(frame, defects) -> DefectCorrectionResult(frame, corrected_pixel_count, method)`.
    Built-ins: `none` (Null Object passthrough, fully supported),
    `neighbor_average`, `directional`.
  - `RawDecoder` / `RawHandle` protocols split decoding into
    `read_cfa()` → correction → `develop(cfa)`, so the decode stage owns the
    order and the LibRaw adapter stays a thin adapter.
  - `DefectDetector` protocol for `raw-pipeline detect-defects` (dark frame or
    batch statistics), returning `DetectedDefect`s with scores.
- **Coordinates** are raw sensor coordinates of the *visible* area as exposed by
  LibRaw (`raw_image_visible`): `x` = column, `y` = row, origin top-left,
  before rotation/flip/crop. `detect-defects` writes the same coordinates, so a
  map produced by the detector always matches the corrector.
- Same-colour neighbours on a Bayer sensor are 2 photosites apart
  (`CFA_TILE_SIZE`); correctors only read photosites of the same colour that
  are outside the mask.
- The decode stage records `defects.method` and `defects.corrected_pixel_count`
  per image, and produces `DEFECT_CROPS` (100 % before/after) for the contact
  sheet when correction is enabled.

## Consequences

- The defect never reaches the demosaic step, so the corrected line is
  invisible at 100 % when neighbours are reliable.
- When correction is on, the contact sheet's "before" crop needs a second,
  uncorrected development of the same file (cost ~2x decode time, only when
  `defects.method != none` and a report is requested).
- Raw decoders other than LibRaw must expose the same visible-area coordinate
  system to be compatible with existing defect maps.
