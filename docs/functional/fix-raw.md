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
args=""
for a in "$@"; do
  case "$a" in /*) a=$(winepath -w "$a") ;; esac
  args="$args \"$a\""
done
eval exec wine "\"$exe\"" $args
```

The Wine route is not tested in CI; DNGLab is the recommended converter on Linux.

### `existing_dng`

Uses `.DNG` files you already converted (for example with Adobe DNG Converter
on another machine) as the base. Each `X.DNG` needs its original `X.CR2` next
to it: the defect map is in CR2 coordinates, and `fix-raw` only writes when it
can match the DNG photosites to the CR2 exactly. The DNG must be uncompressed,
mosaic (not linear) and 16-bit; otherwise it is rejected with the reason.

## What changes in the DNG

| Part | After `fix-raw` |
| --- | --- |
| Raw photosites outside the defect map | Byte-identical to the converter's DNG |
| Raw photosites in the defect map | Corrector output (same as `process`) |
| EXIF, camera profiles, colour matrices, white balance, crop, orientation | Unchanged (the TIFF/DNG structure is not rebuilt) |
| `RawImageDigest` / `NewRawImageDigest` | Recomputed, or removed when they cannot be reproduced (see the report) |
| Embedded previews/thumbnails | Per `raw_export.preview` |

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
that fails is deleted and the image is reported as failed:

1. It opens with LibRaw (rawpy).
2. Every photosite outside the defect mask equals the base DNG.
3. Every photosite inside it equals the corrector output.
4. The defect detector (`defects.detection`) no longer finds the configured
   defects in the corrected frame.
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
| `passthrough` | `True` when nothing changed (method `none` or empty map); the DNG is then byte-identical to the base |
| `digest` | `absent`, `unchanged`, `recomputed` or `removed` |
| `preview` | Preview handling actually applied |
| `verification` | `passed` or `skipped` |
