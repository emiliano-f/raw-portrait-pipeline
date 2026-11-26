# Testing

## Checks

The same commands run in CI (`.github/workflows/ci.yml`):

```bash
uv sync --locked --extra docs
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest
NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict
```

`pytest` uses `--strict-markers` and turns warnings into errors.

## Layout

| Folder | What |
| --- | --- |
| `tests/unit/<package>/` | Unit tests per package, on synthetic data. |
| `tests/contract/` | Reusable contract suites (`StageContract`, `ExporterContract`, `DefectCorrectorContract`, `FaceDetectorContract`, `MeteringContract`, `QAProviderContract`). Every implementation subclasses its suite. |
| `tests/support/synthetic.py` | Deterministic synthetic images, CFA frames and faces. |
| `tests/integration/` | End-to-end runs on real CR2 files, marked `slow`. |
| `tests/fixtures/dark/` | The committed dark frame (EOS 800D, lens cap on, no personal data); see `tests/fixtures/README.md`. |

Rules: no network (QA providers get fake transports), no real photos in the
repository (only the metadata-stripped dark frame), deterministic seeds, and anything that writes files uses
`tmp_path`.

## Integration tests

`tests/integration/` has two kinds of slow tests.

On the committed dark frame (always available, run in CI):

- `detect-defects --dark-frame` finds the hot columns 1509 and 1510;
- `fix-raw` with DNGLab (skipped when `dnglab` is not on `PATH`) writes a DNG
  whose photosites outside the map equal the CR2, whose corrected columns are
  within dark-frame noise of their neighbours, and with the input unchanged.

On real portraits, read from the folder named by `RAW_PIPELINE_FIXTURES_DIR`
(skipped when it is unset), the `process` command, the decode → meter → tone
stages and `fix-raw` are checked, for each portrait:

- the JPEG and TIFF exist, have the decoded size and embed an sRGB ICC profile;
- `tone.clipped_fraction` and the clipped pixels of the JPEG stay below 2 %;
- the input files' SHA-256 are unchanged;
- a second run gives byte-identical outputs and reports;
- a rerun without `--overwrite` fails (exit 1) and leaves every file
  unchanged, while `--overwrite` succeeds;
- when a face is found, the toned face luminance is within 0.1 EV of
  `meter.target_face_luminance` (unless the correction was clamped); without a
  face, the centre-weighted target holds instead.

```bash
uv run pytest -m slow                                          # dark frame only
RAW_PIPELINE_FIXTURES_DIR=~/Pictures/portraits uv run pytest -m slow  # plus your portraits
```

`*.CR2` is git-ignored (except the dark frame). Never commit portraits or
other personal photos, with or without Git LFS: the repository is public.

## Docs

`mkdocs build --strict` fails on broken links and on the generated sections
(`docs/_hooks/generated.py`): a config field without a description, or a
registered component missing from [Stages](../functional/stages.md).
