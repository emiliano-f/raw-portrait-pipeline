# raw-portrait-pipeline

Batch pipeline that turns Canon CR2 portraits (Rebel T7i / 800D) into finished
JPG/TIFF images: raw-level sensor defect correction, face-metered exposure,
highlight recovery, optional subtle skin retouch, contact sheet and CSV report.
It runs offline and never modifies the RAW files.

```bash
uv sync
uv run raw-pipeline process input/ output/ --config config.yaml
```

- **Using it**: [Getting started](functional/getting-started.md),
  [Usage](functional/usage.md), [Configuration](functional/configuration.md),
  [Stages](functional/stages.md), [QA and privacy](functional/qa-and-privacy.md),
  [Troubleshooting](functional/troubleshooting.md).
- **Changing it**: [Architecture](technical/architecture.md),
  [Extending](technical/extending.md), [Colour pipeline](technical/color-pipeline.md),
  [Testing](technical/testing.md), the component pages and the API reference.
