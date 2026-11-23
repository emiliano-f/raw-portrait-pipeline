# Export and report

Package `export-report`: the exporters, the `export` and `report` stages and the
batch reporters. Everything here consumes core abstractions only
(`ImageContext`, `ImageOutcome`, `core.color`, the registries).

| Registry | Name | Class | Module |
|---|---|---|---|
| `exporter_registry` | `jpeg` | `JpegExporter` | `raw_pipeline.exporters.jpeg` |
| `exporter_registry` | `tiff` | `TiffExporter` | `raw_pipeline.exporters.tiff` |
| `stage_registry` | `export` | `ExportStage` | `raw_pipeline.stages.export` |
| `stage_registry` | `report` | `ReportStage` | `raw_pipeline.stages.report` |
| `reporter_registry` | `csv` | `CsvReporter` | `raw_pipeline.exporters.reporters.csv_report` |
| `reporter_registry` | `contact_sheet` | `ContactSheetReporter` | `raw_pipeline.exporters.reporters.contact_sheet` |

## Encoding (shared by exporters and reports)

`raw_pipeline.exporters.encoding`:

1. **Resize in linear light.** When `export.long_edge_px` is set and smaller
   than the image, the scene-linear float image is downscaled with OpenCV
   `INTER_AREA` (box filter). Images are never upscaled. Resizing before the
   transfer function keeps edges and highlights photometrically correct.
2. **Clip and encode.** `core.color.srgb_encode` clips to `[0, 1]` and applies
   the sRGB curve; `core.color.to_uint8` / `to_uint16` quantise.
3. **ICC profile.** An sRGB profile is generated once with Pillow `ImageCms`
   (LittleCMS). LittleCMS stamps the creation time into the header (bytes
   24–35); the exporter overwrites it with a fixed date (2026-01-01) so the
   profile, and every file embedding it, is byte-identical across runs.

## Exporters

Both exporters take `ctx.image` and a destination path without suffix, append
their suffix, and write through `exporters.output.write_output`:

- parent directories are created;
- without `--overwrite` the file is opened with `O_EXCL` (`"xb"`); an existing
  file raises `OutputExistsError` and is left untouched;
- with `--overwrite` the data goes to a temporary sibling that atomically
  replaces the file once complete, so a failed rewrite keeps the previous file;
- on any write error the partial file is removed and `ExportError` is raised.

| | `jpeg` (`.jpg`) | `tiff` (`.tif`) |
|---|---|---|
| Pixels | 8-bit sRGB | 16-bit sRGB |
| Encoder | Pillow, quality `export.jpeg_quality`, 4:4:4 chroma, baseline, standard Huffman tables | tifffile, zlib + horizontal predictor, contiguous RGB |
| ICC | embedded (APP2) | tag 34675 |
| EXIF | APP1 block | tags in IFD0 (+ ExifIFD) |
| GPS | only when `export.keep_gps` | never (see below) |

**EXIF.** `exporters.exif` writes only descriptive fields from `ExifInfo`:
camera make/model, lens, capture date (`DateTime`, `DateTimeOriginal`), ISO,
exposure time, f-number and focal length (as rationals). `Orientation` is
written as `1` because decoded pixels are already upright. No software version,
processing time or other run-dependent value is written, so reruns are
byte-identical. GPS tags (`ExifInfo.gps`) are copied into the JPEG GPS IFD only
when `keep_gps` is true; tag names Pillow does not know are skipped with a debug
log.

**TIFF and GPS.** TIFF outputs never contain GPS tags, whatever `keep_gps`
says, because tifffile cannot write a GPS sub-IFD. `keep_gps: true` therefore
applies to JPEG outputs only; when it is set, the TIFF exporter logs a warning
saying so when it is built. This is an accepted limitation: it errs on the side
of stripping location data.

## `export` stage

Builds the exporters listed in `export.formats` (in order) from the exporter
registry and calls each with `<output_dir>/<input stem><export.filename_suffix>`.
Pixels are not changed. If an exporter refuses an existing file
(`OutputExistsError`), the files already written for that image in the same call
are removed, so an image is exported all-or-nothing. Sets `OUTPUT_PATHS` (tuple, exporter order) and records:

| Key | Meaning |
|---|---|
| `export.formats` | comma-separated exporter names |
| `export.files` | `;`-separated output file names |
| `export.output_width`, `export.output_height` | pixel size after the optional resize |
| `export.jpeg_quality`, `export.long_edge_px`, `export.color_space`, `export.keep_gps` | settings used |

## `report` stage

Builds `REPORT_THUMBNAILS`, a read-only mapping of sRGB `uint8` arrays, so the
batch reporters never hold full-resolution images:

| Key | Source | Size |
|---|---|---|
| `before` | `BEFORE_IMAGE` (decoded, camera white balance; omitted when absent) | long edge ≤ `report.thumbnail_long_edge_px` |
| `after` | `ctx.image` (final pixels) | long edge ≤ `report.thumbnail_long_edge_px` |
| `defect_before`, `defect_after` | `DEFECT_CROPS` (only when present) | 100 %, not resized |

The defect crops are scene-linear crops taken before exposure. The stage applies
the metered gain (`EXPOSURE.correction_ev`, 0 EV when absent) to both crops so
they are visible at the same brightness as the output; the gain is recorded as
`report.defect_crop_ev`. Other records: `report.thumbnails` (keys produced) and
`report.thumbnail_long_edge_px`.

## `csv` reporter

Writes `report.csv_filename` (default `report.csv`) in the output directory,
UTF-8, `\n` line endings, one row per image including failed ones.

- Leading columns: `file`, `status` (`ok` / `failed`), `error_type`, `error`,
  `outputs` (`;`-separated file names).
- Then every key of `ImageMetadata.flatten()` (`exif.*` first, then
  `stage.param`), grouped by stage in first-appearance order and sorted within a
  stage.
- Rows are sorted by file name, so worker order never changes the file.
- Cells: `None` → empty, booleans → `true`/`false`, floats → shortest
  round-tripping `repr`. A cell missing for an image (stage not run) is empty.
- Text cells (file names, errors, string parameters) starting with `=`, `+`,
  `-`, `@`, tab or carriage return get a leading `'`, so spreadsheets do not
  evaluate them as formulas. Numbers are never prefixed.

## `contact_sheet` reporter

Writes `report.contact_sheet_filename` (default `contact_sheet.jpg`): a header
row of column titles, then one row per image sorted by file name.

- Columns: *Before (camera WB)*, *After*, each a `thumbnail_long_edge_px` square
  cell with the thumbnail centred; when any image in the sheet has defect
  crops, two more columns show them at 100 % (cell size = largest crop).
- Missing thumbnails (e.g. failed images) are grey placeholders.
- Caption (up to 3 lines): file name plus the key parameters present among
  `meter.method`, `meter.correction_ev`, `meter.faces_used`, `retouch.method`,
  `retouch.strength`, `defects.method`, `defects.corrected_pixel_count`,
  `qa.flags`. Failures show `FAILED <error_type>: <error>` in red. Every other
  value is in the CSV.
- Batches are split into `contact_sheet.jpg`, `contact_sheet-2.jpg`, … at 40
  images per page, or fewer when large thumbnails would make a page taller than
  the JPEG limit of 65 535 px. Pages are rendered and written one at a time.
- Output is deterministic: Pillow's built-in bitmap font, fixed layout, sRGB
  ICC, quality 90, 4:4:4, no EXIF.

## API

Generated reference: [`raw_pipeline.exporters`](../api/exporters.md), [`raw_pipeline.stages`](../api/stages.md).
