# Troubleshooting

## `ImportError: libGL.so.1` (Linux)

OpenCV needs the OpenGL runtime even without a display. On Debian/Ubuntu:

```bash
sudo apt-get install libgl1
```

## MediaPipe falls back to OpenCV (Linux)

MediaPipe loads `libEGL.so.1` and `libGLESv2.so.2` at run time. When they are
missing (minimal servers, containers), the `mediapipe` detector fails with
`MediaPipe native library could not be loaded (...)`, the image is **not**
failed, and the `opencv` fallback detects the faces instead. In `report.csv`
you see `meter.detector_used = opencv` and the error in
`meter.detection_errors`; `retouch.mask_source` becomes `ellipse` because
OpenCV gives no face landmarks. Install the libraries to get MediaPipe back:

```bash
sudo apt-get install libegl1 libgles2
```

On Intel macOS MediaPipe is not installed at all and OpenCV is always used.

## Output already exists

```text
[1/1] IMG_3378.CR2: FAILED: /tmp/out/IMG_3378.jpg already exists; pass --overwrite to replace it
ERROR raw_pipeline.runtime.batch: reporter csv failed: /tmp/out/report.csv already exists; pass --overwrite to replace it
0 ok, 1 failed
```

The run exits with code 1 and the existing files are left byte-for-byte untouched.

The pipeline never replaces a file silently. Choose one:

- write to a new output folder;
- pass `--overwrite` to replace the outputs and reports (each file is
  replaced atomically, so a failed rewrite keeps the old one);
- set `export.filename_suffix` (for example `_v2`) to write new names.

The same rule applies to `report.csv`, the contact sheet and
`detect-defects --output`.

`detect-defects` without `--output` refuses to rewrite an existing `--config`
unless `--overwrite` is given, because rewriting drops the YAML comments. Use
`--output defects.yaml` to keep your config untouched.

## "output names collide with report files" (exit 2)

An input named like a report (for example `report.CR2` or
`contact_sheet.CR2`) would produce an image with the same name as a report.
Rename the RAW file, or change `export.filename_suffix` or the
`report.*_filename` options.

## No face found

The `center_weighted` fallback meters the whole frame (`meter.fallback_used =
true`, `meter.faces_found = 0`) and no retouch is applied. Small or turned
faces may be missed: raise `meter.detection_long_edge_px` or lower
`meter.min_face_confidence`.

## `detect-defects` reports scene edges

With one normal photo every sharp edge looks like a defect. Use a dark frame
(`--dark-frame`, lens cap on) or a batch of several different photos.

## Invalid configuration (exit 2)

The message lists every problem with its option path, for example
`meter.max_correction_ev: Input should be less than or equal to ...`. Check the
[configuration reference](configuration.md) for types and limits.
