# Getting started

`raw-pipeline` turns a folder of Canon `.CR2` portraits (Rebel T7i / EOS 800D)
into finished JPG (and optionally 16-bit TIFF) images, plus a contact sheet and
a CSV with every parameter it applied. It runs fully offline and never touches
the RAW files.

## Requirements

- Python ≥ 3.12 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).
  uv installs the right Python by itself if it is missing.
- Linux: OpenCV needs `libGL.so.1` and MediaPipe needs `libEGL.so.1` and
  `libGLESv2.so.2`. On Debian/Ubuntu:

    ```bash
    sudo apt-get install libgl1 libegl1 libgles2
    ```

    Without the EGL/GLES libraries face detection falls back to OpenCV (see
    [Troubleshooting](troubleshooting.md#mediapipe-falls-back-to-opencv-linux)).
- Intel macOS: MediaPipe has no wheel there, so it is not installed and OpenCV
  detects faces.

## Install

```bash
git clone https://github.com/emiliano-f/raw-portrait-pipeline.git
cd raw-portrait-pipeline
uv sync
```

`uv sync` creates `.venv/` from the committed `uv.lock`. Always run the tool
through `uv run` so it uses that environment.

## First run

```bash
uv run raw-pipeline process input/ output/
```

- `input/` is the folder with your `.CR2` files. It is only read.
- `output/` is created if needed. It receives, per RAW file, `<name>.jpg`, and
  for the batch `report.csv` and `contact_sheet.jpg`.

Each image prints one progress line (`[1/1] IMG_3378.CR2: ok`) and the run ends
with `N ok, M failed` and the report paths. The exit code is 0 only if every
image and report succeeded.

Open `contact_sheet.jpg` first: it shows every image before (camera white
balance) and after, with the exposure correction and retouch strength.

## Next steps

- Copy [`config.example.yaml`](https://github.com/emiliano-f/raw-portrait-pipeline/blob/main/config.example.yaml)
  to `config.yaml`, adjust it and pass `--config config.yaml`. Every option is
  in the [configuration reference](configuration.md).
- If your sensor has hot or stuck lines, find them with
  [`detect-defects`](usage.md#find-sensor-defects-detect-defects).
- [Stages](stages.md) explains what each processing step does to the pixels.
