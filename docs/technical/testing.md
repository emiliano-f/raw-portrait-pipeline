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
| `tests/fixtures/` | Where local sample CR2s go. **Not committed.** |

Rules: no network (QA providers get fake transports), no real photos in the
repository, deterministic seeds, and anything that writes files uses
`tmp_path`.

## Integration tests

`tests/integration/` runs the real `raw-pipeline process` command and the
decode → meter → tone stages on every `.CR2` in `tests/fixtures/`, or in the
folder named by `RAW_PIPELINE_SAMPLES`. Without samples they are skipped. They
check, for each sample:

- the JPEG and TIFF exist, have the decoded size and embed an sRGB ICC profile;
- `tone.clipped_fraction` and the clipped pixels of the JPEG stay below 2 %;
- the input files' SHA-256 are unchanged;
- a second run gives byte-identical outputs and reports;
- a rerun without `--overwrite` fails (exit 1) and leaves every file
  unchanged, while `--overwrite` succeeds;
- when a face is found, the toned face luminance is within 0.1 EV of
  `meter.target_face_luminance` (unless the correction was clamped); without a
  face, the centre-weighted target holds instead.

Get a sample (public, from raw.pixls.us) and run them:

```bash
curl -sSL -f -o tests/fixtures/IMG_3378.CR2 "https://raw.pixls.us/data/Canon/EOS%20800D/IMG_3378.CR2"
uv run pytest -m slow
```

or point at your own folder:

```bash
RAW_PIPELINE_SAMPLES=~/Pictures/shoot uv run pytest -m slow
```

`*.CR2` is git-ignored. Never commit sample or personal photos: the
repository is public.

## Docs

`mkdocs build --strict` fails on broken links and on the generated sections
(`docs/_hooks/generated.py`): a config field without a description, or a
registered component missing from [Stages](../functional/stages.md).
