# Defects: raw-level sensor defect correction

Package `raw_pipeline.defects` repairs hot/stuck columns, rows and pixels on the
undemosaiced CFA mosaic (`CFAFrame`), before demosaicing spreads a defect into
its neighbours ([ADR 0002](../../adr/0002-raw-level-defect-correction.md)). It
also provides the detector behind `detect-defects`.

| Module | Contents |
| --- | --- |
| `null.py` | `NullDefectCorrector` (`none`) |
| `neighbor_average.py` | `NeighborAverageCorrector` (`neighbor_average`) |
| `directional.py` | `DirectionalCorrector` (`directional`) |
| `sampling.py` | `SameColourSampler`, mask and replacement helpers shared by the correctors |
| `detection.py` | `StatisticalDefectDetector` |
| `statistics.py` | robust (median/MAD) residuals and z-scores used by the detector |

Importing `raw_pipeline.defects` registers the three correctors in
`raw_pipeline.core.registries.defect_corrector_registry`; `defects.method` in the
config selects one.

## Correctors

All correctors satisfy the `DefectCorrector` contract
(`tests/contract/defect_corrector_contract.py`): the input frame is never
mutated, the output has the same shape, dtype and CFA layout, and only
photosites inside `DefectMap.mask(height, width)` change. Replacement values
are rounded and clipped to `[0, 65535]`.

**Same-colour sampling.** In a 2×2 Bayer tile the nearest photosite of the
same colour is `CFA_TILE_SIZE` (2) photosites away, so neighbours are searched
in steps of 2. A neighbour is *valid* if it is inside the frame and not
masked. Search goes up to `MAX_SEARCH_STEPS` (16) steps, so bands up to 31
photosites wide can be repaired; a photosite with no valid neighbour raises
`ValueError`. When a band touches the image border, only the inner side is
sampled (one-sided).

### `none`

Returns the input frame object itself and `corrected_pixel_count=0`.

### `neighbor_average`

Each masked photosite becomes the mean of the nearest valid same-colour
photosite on both sides, perpendicular to the defect:

| Defect | Directions sampled |
| --- | --- |
| `ColumnDefect` | left, right |
| `RowDefect` | up, down |
| `PixelDefect` | all 8 (horizontal, vertical, diagonals) |

Overlapping defects (a column crossing a row, a pixel on a line) are each
corrected once, by the first defect in the map that covers them.

### `directional`

For every masked photosite four candidate directions are evaluated:
horizontal, vertical and the two diagonals. Along each direction the nearest
valid same-colour photosites on both sides are found; the candidate estimate
is their distance-weighted linear interpolation. The direction's gradient is
the mean `|endpoint difference| / distance` measured on the line through the
photosite and on the two parallel same-colour lines next to it. The direction
with the lowest gradient wins (ties in the order listed), so an edge crossing
a hot column (hair, eyelashes) is continued along the edge instead of being
blurred across it; on the synthetic sharp-edge test its error is ≪ that of
`neighbor_average`. Photosites with no two-sided direction (image corners)
fall back to the mean of the nearest valid ring neighbours.

## Detection

`StatisticalDefectDetector.detect(frames)` consumes an iterable of `CFAFrame`s
one at a time (memory is a few full-frame float32 arrays regardless of the
number of frames) and returns `DetectedDefect`s: columns, then rows, then
pixels in row-major order. It is built from `defects.detection`
(`outlier_threshold_sigma`, `min_frame_fraction`, `max_pixel_defects`) with
`from_env(env)` or `from_config(config)`.

Per frame:

1. **High-pass residual.** Each photosite minus the median of its same-colour
   neighbours; neighbours outside the frame are skipped. Smooth scene content
   cancels out, so the same statistics work on dark frames and on portraits.
2. **Lines.** For columns (rows use the transposed frame) the residual is
   taken separately against 3 same-colour neighbours on the left and on the
   right, reduced to its median over segments of `SEGMENT_LENGTH_PX` (64)
   rows and turned into robust z-scores (`(x - median) / (1.4826 · MAD)`)
   across columns of the same CFA parity. A defect differs from *both*
   sides, so a segment's score is the smaller of the two z-scores when they
   agree in sign, else 0: a sharp scene edge that stays in place across a
   tripod batch differs from one side only and is not flagged. Segments
   with `|z| > outlier_threshold_sigma` are flagged, excluded as neighbours,
   and the frame is rescored until nothing new is flagged (at most 8
   passes); each pass uncovers the next column of a wide band or of a band
   at the sensor border. Each side's reference is the median of its 3
   nearest same-colour neighbours not yet flagged (searched up to 16
   photosites out). Bands up to 6 photosites wide are detected, up to 4 at
   the sensor border: a wider band there looks exactly like a scene edge
   parallel to the border. Wider bands go unnoticed (every column then has a
   defective majority on one side) and must be written into the map by
   hand.
3. **Pixels.** The 8-neighbour residual, ignoring photosites flagged as lines
   in this frame, is scored per CFA colour plane in the same way. A
   photosite with more than a third of its in-frame neighbours on detected
   lines is not scored (its remaining neighbours lie on one side), so a
   pixel defect in the same-colour column or row right next to a line is
   not reported.

After all frames, a candidate is kept if it was flagged in at least
`ceil(min_frame_fraction · frames)` frames. Lines need at least
`MIN_LINE_SEGMENTS` (2) consecutive kept segments; a run that does not span the
sensor becomes a partial line with `start`/`end` (multiples of 64, so up to 63
good photosites at each end may be included). Adjacent lines with the same
extent are merged into one defect with `width > 1`. Pixels inside detected
lines are dropped and at most `max_pixel_defects` pixels are reported,
strongest first (a warning is logged when the cap is hit).

**Inputs.** A single dark frame (lens cap on, same ISO and exposure as the
shoot) is the most reliable input: every defect is visible and there is no
scene. In batch mode on normal photos, scene edges are outliers in single
frames only and are rejected by `min_frame_fraction`; a defect hidden by
clipped highlights in some frames needs a lower fraction.

**Cost.** About 10 s per 24 MP frame on one core (dominated by medians), and
under 0.1 s for either corrector.

## API

::: raw_pipeline.defects.null
::: raw_pipeline.defects.neighbor_average
::: raw_pipeline.defects.directional
::: raw_pipeline.defects.sampling
::: raw_pipeline.defects.detection
::: raw_pipeline.defects.statistics
