# raw-portrait-pipeline

Command-line tool that batch-processes Canon RAW portraits (`.CR2`, Rebel T7i / 800D)
into finished JPG/TIFF images. See `AGENTS.md` for architecture and conventions and
`docs/` for the user and contributor guides.

```bash
uv sync
uv run raw-pipeline process input/ output/ --config config.yaml
uv run raw-pipeline detect-defects input/ --config config.yaml [--dark-frame path.CR2] [--output defects.yaml]
```
