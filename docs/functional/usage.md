# Usage

```text
raw-pipeline process INPUT_DIR OUTPUT_DIR [--config FILE] [--overwrite] [--workers N] [-v | -vv | -q]
raw-pipeline detect-defects INPUT_DIR [--config FILE] [--dark-frame FILE] [--output FILE] [--overwrite] [--yes] [-v | -vv | -q]
```

Run either through `uv run`. `--config` is optional; without it the defaults
in the [configuration reference](configuration.md) apply.

## Process a folder (`process`)

```bash
uv run raw-pipeline process shoot/ shoot-out/ --config config.yaml
```

| Option | Effect |
| --- | --- |
| `--config FILE` | YAML configuration. |
| `--overwrite` | Replace existing outputs and reports. Without it nothing existing is ever replaced. |
| `--workers N` | Parallel processes (≥ 1). Default: `pipeline.workers`, else CPU count − 1. |
| `-v`, `-vv` | Info / debug logging (debug includes per-stage timings). |
| `-q` | Errors only, no progress lines. |

What it does:

- Reads every `.CR2` (any case) directly inside `INPUT_DIR`, sorted by name.
  Hidden files are skipped. The folder and the RAW files are never modified.
- Runs the stages of `pipeline.stages` on each image (see [Stages](stages.md)).
  A failing image is logged, recorded in the CSV with its error and skipped; the
  batch continues.
- Writes `<input stem><export.filename_suffix>.jpg` (and `.tif` with
  `export.formats: [jpeg, tiff]`), `report.csv` and `contact_sheet.jpg`
  (`contact_sheet-2.jpg`, … for batches above 40 images).

Output rules:

- `OUTPUT_DIR` may not be `INPUT_DIR`. If it lies inside it, it is excluded
  from the input scan.
- An existing output file is **never replaced silently**: without
  `--overwrite` that image fails with "already exists" and the file is left
  untouched (see [Troubleshooting](troubleshooting.md#output-already-exists)).
- Results are deterministic: the same RAW, config and version give
  byte-identical files, with any worker count.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Every image and every report succeeded. |
| 1 | At least one image or report failed; the others were still written. |
| 2 | Bad arguments, invalid config, unknown component name, missing or empty input folder, output folder equal to the input folder, or an input whose name collides with a report file. Nothing was processed. |

### The CSV report

`report.csv` has one row per image, failed ones included, sorted by file name:
`file`, `status` (`ok`/`failed`), `error_type`, `error`, `outputs`, then
`exif.*` and one column per recorded parameter as `stage.param` (for example
`meter.correction_ev`, `tone.clipped_fraction`, `retouch.strength`,
`defects.corrected_pixel_count`, `qa.flags`). [Stages](stages.md) lists what
each stage records.

## Corrected RAW files (`fix-raw`)

`raw-pipeline fix-raw input/ output/ --config config.yaml` writes a DNG per CR2
with the defect map already corrected on the raw data. See
[Corrected RAW (fix-raw)](fix-raw.md).

## Find sensor defects (`detect-defects`)

Hot or stuck columns, rows and pixels are corrected on the raw mosaic before
demosaicing, from a map in `defects.map`. `detect-defects` builds that map.

### With a dark frame (recommended)

1. Shoot one frame with the lens cap on, at the ISO and shutter speed of the
   shoot.
2. Analyse it:

    ```bash
    uv run raw-pipeline detect-defects shoot/ --config config.yaml --dark-frame dark/IMG_0001.CR2 --output defects.yaml
    ```

3. Review the printed findings (raw sensor coordinates, score, frames flagged)
   and copy the `defects:` block of `defects.yaml` into your config, choosing a
   `defects.method` (`directional` is the best default).

With `--dark-frame`, `INPUT_DIR` is not read.

### From a batch of photos

```bash
uv run raw-pipeline detect-defects shoot/ --config config.yaml --output defects.yaml
```

A defect is reported only if it is an outlier in at least
`defects.detection.min_frame_fraction` of the frames (default 80 %), so scene
edges, which move between shots, are rejected. Use a batch of several
different photos: **with a single normal photo, sharp scene edges are reported
as defects** (every candidate is in 1/1 frames).

### Writing into the config

Without `--output`, the command asks `Update defects.map in config.yaml? [y/N]`
before replacing `defects.map` in the `--config` file; `--yes` answers yes.
Because rewriting the file drops its YAML comments, an existing config is only
rewritten together with `--overwrite`; otherwise the command stops (exit 2)
before reading any frame. `--output` must end in `.yaml` or `.yml` and is
refused if it exists, unless `--overwrite`. Nothing is written when no defect
is found. The RAW files are never modified.

Exit code: 0 on success, 1 if some frames could not be read, 2 on usage errors.
