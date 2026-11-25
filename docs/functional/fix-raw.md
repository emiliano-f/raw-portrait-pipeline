# Corrected RAW files (`fix-raw`)

`fix-raw` writes a **DNG** for every input with the sensor defect line already
removed, so you can keep editing it as a RAW file in Lightroom, darktable,
RawTherapee and other RAW editors. It uses the same `defects` section as
`process`: the same map, the same corrector, the same result on the mosaic.

```bash
uv run raw-pipeline fix-raw input/ output/ --config config.yaml [--overwrite]
```

- Output: `output/<name>.dng` plus `output/fix_raw_report.csv`.
- Inputs are never modified. Converter intermediates live in a temporary
  folder that is deleted afterwards.
- An existing output or report is never replaced without `--overwrite`.
- A failing image is recorded in the report and the batch continues; the exit
  code is `1` when any image failed, `2` for configuration or input-folder errors.

CR2 files cannot be written back, so the corrected file is a DNG made by a
converter from your CR2. Only the defect photosites change (see
[What changes in the DNG](#what-changes-in-the-dng)).

## Workflow

1. Build the defect map once with [`detect-defects`](usage.md#find-sensor-defects-detect-defects)
   (a dark frame works best) or write it by hand.
2. Check it on a few photos with `raw-pipeline process` and the 100% defect crops
   of the contact sheet.
3. Run `fix-raw` on the folder and open the DNGs in your editor.

```yaml
defects:
  method: directional
  map:
    - {type: column, index: 1509, width: 2}
raw_export:
  source: dnglab
  preview: regenerate
  verify: true
  # keep_gps: unset = same as export.keep_gps (GPS stripped by default)
```

## Converters

`raw_export.source` picks where the base DNG comes from. The required output
format (uncompressed, mosaic/CFA, no embedded original) is enforced by the code;
`raw_export.converter.args` only adds extra flags, and flags that would change
that format are rejected at startup.

### `dnglab` (default)

[DNGLab](https://github.com/dnglab/dnglab) is open source and runs on Linux,
Windows and macOS. Its DNG keeps every CR2 photosite value unchanged (checked on
the EOS 800D). Install it yourself; the application never downloads it:

- Debian/Ubuntu: download `dnglab_<version>_amd64.deb` from the
  [releases](https://github.com/dnglab/dnglab/releases) and run
  `sudo apt install ./dnglab_<version>_amd64.deb`.
- Other Linux: download `dnglab_linux_x64`, `chmod +x` it and put it on `PATH`.
- Windows/macOS: unzip the release archive and put `dnglab` on `PATH`, or set
  `raw_export.converter.executable` to its full path.

Command run: `dnglab convert [args] --compression uncompressed --embed-raw false <in.CR2> <tmp>/<name>.dng`.

### `adobe_converter`

Adobe DNG Converter is free but closed source; it is not bundled or downloaded.
Install it from [Adobe](https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html)
and point `raw_export.converter.executable` at it:

```yaml
raw_export:
  source: adobe_converter
  converter:
    executable: "C:/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"
    # macOS: "/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"
    args: []
    timeout_s: 120
```

Command run: `<exe> [args] -u -d <tmp> -o <name>.dng <in.CR2>` (`-u` =
uncompressed; the default is mosaic data without the original embedded).
`-c`, `-l`, `-e`, `-lossy`, `-d` and `-o` are rejected in `args`.

**Linux:** Adobe ships no Linux build. It can run under Wine; point
`executable` at a small wrapper script that converts the paths, for example:

```sh
#!/bin/sh
# adobe-dng-wine: run Adobe DNG Converter under Wine with Windows paths.
exe="$HOME/.wine/drive_c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"
for a do
  shift
  case "$a" in /*) a=$(winepath -w "$a") ;; esac
  set -- "$@" "$a"
done
exec wine "$exe" "$@"
```

The Wine route is not tested in CI; DNGLab is the recommended converter on Linux.

### `existing_dng`

Uses `.DNG` files you already converted (for example with Adobe DNG Converter
on another machine) as the base. Each `X.DNG` needs its original `X.CR2` next
to it: the defect map is in CR2 coordinates, and `fix-raw` only writes when it
can match the DNG photosites to the CR2 exactly. The DNG must be uncompressed,
mosaic (not linear) and 16-bit; otherwise it is rejected with the reason.

## Input rules

| `raw_export.source` | Reads | Ignores |
| --- | --- | --- |
| `dnglab`, `adobe_converter` | `.CR2` | `.DNG` |
| `existing_dng` | `.DNG`, each with its `.CR2` of the same name beside it | `.CR2` without a DNG of the same name |

Both rules fail with a message that says what to do:

- A folder with nothing the source reads stops before any work (exit code `2`),
  for example
  `no .cr2 files found in in/ (raw_export.source: dnglab reads only .cr2); 3 other RAW file(s) found: to patch existing DNGs set raw_export.source: existing_dng and keep each original X.CR2 next to X.DNG`.
  When some files are read and others are not, the run goes on and a warning
  names the skipped files.
- With `existing_dng`, a DNG whose CR2 is missing fails on its own (the rest of
  the batch continues; exit code `1`) with
  `IMG_0002.DNG: original IMG_0002.CR2 not found next to it; raw_export.source: existing_dng needs it to map the defect coordinates (copy the CR2 beside the DNG, or run a converter source on the CR2)`.

### Future work: DNG-only input

The CR2 is needed only to find where the CR2 visible area (the coordinates of
`defects.map`) sits inside the DNG raw image. That offset is the same for
every file of one camera model and converter (for the EOS 800D with DNGLab it
is `(36, 264)`), but the DNG tags alone do not give it (`ActiveArea` is 2 rows
off). A per-camera offset calibration, measured once from a CR2/DNG pair and
stored with the defect map, would let `existing_dng` work on DNGs without their
CR2.

## What changes in the DNG

| Part | After `fix-raw` |
| --- | --- |
| Raw photosites outside the defect map | Byte-identical to the converter's DNG |
| Raw photosites in the defect map | Corrector output (same as `process`) |
| EXIF, camera profiles, colour matrices, white balance, crop, orientation | Unchanged (the TIFF/DNG structure is not rebuilt) |
| GPS (EXIF GPS IFD and XMP `exif:GPS*` fields) | Stripped, unless `raw_export.keep_gps` |
| `RawImageDigest` / `NewRawImageDigest` | Recomputed, or removed when they cannot be reproduced (see the report) |
| Embedded previews/thumbnails | Per `raw_export.preview` |

`raw_export.keep_gps` (default unset, which means "same as
`export.keep_gps`", itself off by default): corrected DNGs drop GPS like the
JPG/TIFF exports do. The GPS directory is zeroed and unlinked from IFD0 and the
XMP GPS fields are blanked in place; the raw image data and its digests are not
touched. Set `raw_export.keep_gps: true` to keep the location in the DNG only.
A DNG that embeds its original raw file (Adobe DNG Converter's "Embed Original
Raw File") fails while GPS is being stripped, because the embedded CR2 keeps its
own GPS; convert without embedding, or set `raw_export.keep_gps: true`.

`raw_export.preview`:

- `regenerate` (default): previews are rendered again from the corrected DNG,
  so the line is gone from the file browser thumbnails too. A preview that
  cannot be replaced in place is kept, and the report says `keep`.
- `keep`: previews stay as the converter made them; they may still show the line
  (only the preview, never the RAW data your editor develops).
- `remove`: embedded previews are dropped and the IFD0 thumbnail is blanked
  to grey. Editors regenerate their own.

## Verification

With `raw_export.verify: true` (default) every written DNG is checked; a DNG
that fails is never published (an earlier output stays in place) and the image
is reported as failed. The checks run on the finished file before it is moved into place:

1. It opens with LibRaw (rawpy).
2. Every photosite outside the defect mask equals the base DNG.
3. Every photosite inside it equals the corrector output.
4. The defect detector (`defects.detection`) scores each configured defect in
   the base frame and in the corrected one. A defect it still finds fails the
   check if its score did not drop (the line is unchanged, or new). A defect
   that is still found but weaker is accepted with a warning, and the report
   says `passed_residual`: a thin scene stripe right next to the corrected
   column (a cable, a hair) can keep a weak detector score after a correct
   repair, so a fixed threshold would reject good DNGs.
5. The DNG still has its raw image where it was (same layout, same CFA).
6. Inputs (CR2 and, for `existing_dng`, the DNG) have the same SHA-256 as before.

## The report

`fix_raw_report.csv` (`raw_export.report_filename`) has one row per input:

| Column | Meaning |
| --- | --- |
| `input_file`, `output_file`, `reference_raw` | Input, written DNG, CR2 used for coordinates |
| `status`, `error` | `ok` or `failed`, and why |
| `source` | Converter (`dnglab`, `adobe_converter`, `existing_dng`) |
| `offset_y`, `offset_x` | Where the CR2 visible area sits in the DNG raw image |
| `mapping_method` | How it was found: `active_area`, `default_crop`, `origin` or `search` |
| `defect_method`, `defects_configured`, `corrected_pixels` | Corrector, map entries, photosites changed |
| `passthrough` | `True` when nothing changed (method `none` or empty map); the raw data and previews are then those of the base (`preview` is `keep`), and the DNG is byte-identical to it unless GPS was stripped |
| `digest` | `absent`, `unchanged`, `recomputed` or `removed` |
| `preview` | Preview handling actually applied |
| `gps` | `absent`, `kept` or `stripped` |
| `verification` | `passed`, `passed_residual` (see [Verification](#verification)) or `skipped` |
