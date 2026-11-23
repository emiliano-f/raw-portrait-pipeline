# Stages

`pipeline.stages` lists the stages each image goes through, in order. The
default is `decode → meter → tone → retouch → export → report → qa`. Each
stage reads the image and the artifacts of earlier stages, and records the
parameters it used as `<stage>.<param>` columns of `report.csv`. Pixels stay
scene-linear (see [Colour pipeline](../technical/color-pipeline.md)) until
`export` encodes them to sRGB.

Every component below is chosen by name in the [configuration](configuration.md);
an unknown name fails before any image is read and lists the registered ones.

## `decode`

Reads the `.CR2` read-only with the `rawpy` decoder (LibRaw), repairs sensor
defects on the raw mosaic (see [Defect correction](#defect-correction)),
demosaics with camera white balance and converts to linear sRGB primaries.
1.0 is the sensor white level in every image (no per-image auto-brightening).
Options: `decode.*` (`demosaic`, `highlight_mode`). Records `decode.*`: LibRaw
version, black/white level, white-balance multipliers, image size.

### Defect correction

Hot or stuck **columns, rows and pixels** listed in `defects.map` are replaced
from same-colour neighbours *before* demosaicing, so a bad photosite is not
smeared into its neighbours. Coordinates are raw sensor coordinates, as
printed by `detect-defects`. `defects.method` picks the corrector:

| `defects.method` | What it does |
| --- | --- |
| `none` | No correction (the default while `defects.map` is empty). |
| `neighbor_average` | Mean of the nearest valid same-colour photosites on both sides, perpendicular to the defect (all 8 directions for a pixel). Simple and fast. |
| `directional` | Tries horizontal, vertical and both diagonals and interpolates along the direction with the smallest gradient, so edges crossing the defect stay sharp. Best default. |

Records `defects.method`, `defects.count` and `defects.corrected_pixel_count`;
the contact sheet adds a before/after crop of the corrected area.

### Finding defects: `detect-defects`

`uv run raw-pipeline detect-defects` scores every column, row and pixel of a
dark frame (`--dark-frame`) or of a batch against its same-colour neighbours
and reports the outliers in `defects.map` form. See
[Usage](usage.md#find-sensor-defects-detect-defects) for the workflow; its
thresholds are `defects.detection.*`.

## `meter`

Finds faces on a small preview and measures how many EV the image needs so the
faces reach `meter.target_face_luminance` (0.18, mid grey; `meter.target_center_luminance` without a face). It never changes pixels.

| Component | Option | What it does |
| --- | --- | --- |
| `mediapipe` | `meter.detector` | MediaPipe face detector plus a 478-point face mesh (used by the skin mask). Primary, models bundled, offline. |
| `opencv` | `meter.fallback_detector` | OpenCV Haar cascade, boxes only. Runs when MediaPipe fails or finds no face. |
| `face_weighted` | `meter.strategy` | Trimmed mean luminance of the central skin area of the faces. |
| `center_weighted` | `meter.fallback_strategy` | Centre-weighted median of the whole frame, used when no face is found. |

The correction is limited to `meter.max_correction_ev`. Records `meter.*`:
detector used, faces found/used, measured luminance, `correction_ev`,
whether the fallback was used, and any detector error.

## `tone`

Multiplies the image by `2 ** meter.correction_ev`, then applies a smooth
luminance curve that keeps hue and saturation: a gentle shadow lift below
`tone.shadow_pivot_luminance` and a highlight shoulder from
`tone.highlight_rolloff_start_ev` that rolls highlights up to
`tone.highlight_rolloff_ev` above white into 1.0. Mid-tones (and so the
metered face) are untouched. Records `tone.*`, including `clipped_fraction`
(fraction of pixels that will clip at export).

## `retouch`

Subtle skin smoothing, only on detected faces.

| `retouch.method` | What it does |
| --- | --- |
| `none` | No retouch. |
| `frequency_separation` | Splits each face into texture and tone, softens only tone blotches inside a skin mask (eyes, brows, lips and nostrils excluded) by at most `retouch.strength`. Pores and fine texture are kept; no geometry changes. |

Records `retouch.*`: strength, mask coverage, faces retouched, mask source
(`landmarks` from MediaPipe, `ellipse` from OpenCV boxes).

## `export`

Optionally downsizes to `export.long_edge_px` (in linear light), clips, encodes
sRGB and writes every format in `export.formats` with an embedded sRGB ICC
profile and the camera EXIF (GPS dropped unless `export.keep_gps`).

| Exporter | File | Pixels |
| --- | --- | --- |
| `jpeg` | `<stem><suffix>.jpg` | 8-bit, quality `export.jpeg_quality`, 4:4:4 |
| `tiff` | `<stem><suffix>.tif` | 16-bit, zlib compressed |

An existing file is never replaced without `--overwrite`.

## `report`

Keeps a small before/after thumbnail and the defect crops for the batch
reports, which the reporters in `report.reporters` write after all images:

| Reporter | File | Content |
| --- | --- | --- |
| `csv` | `report.csv_filename` (`report.csv`) | One row per image with every recorded parameter. |
| `contact_sheet` | `report.contact_sheet_filename` (`contact_sheet.jpg`) | Before/after thumbnails with captions, 40 rows per page. |

## `qa`

Optional review by a vision LLM that only flags closed eyes, blur or odd skin
tone in the CSV; it never changes pixels.

| `qa.provider` | What it does |
| --- | --- |
| `none` | Default. No review, no network. |
| `openai_compatible` | Sends one downscaled JPEG preview per image to the chat-completions endpoint you configure. |

See [QA and privacy](qa-and-privacy.md).

## All registered components

Generated from the component registries at docs build time; the build fails
if a component is registered but not described on this page.

<!-- generated:components -->
