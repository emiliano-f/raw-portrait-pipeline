# QA and privacy

## Offline by default

Processing never uses the network. The face models are bundled with the
package, and the default `qa.provider: none` contacts nothing. Your RAW
files are only read; outputs go to the folder you name.

Exported images keep the camera EXIF (camera, lens, exposure, date) but **GPS
is removed** unless `export.keep_gps: true`.

## Optional vision-LLM review

The `qa` stage can ask a vision-capable LLM to flag closed eyes, blur and odd
skin tone. It only adds columns to `report.csv`; it never changes an image and
never fails one.

Enable it in the config:

```yaml
qa:
  provider: openai_compatible
  endpoint: https://api.example.com/v1   # any chat-completions API with image input
  model: your-vision-model
  api_key_env: QA_API_KEY                # name of the environment variable, not the key
```

and export the key before running:

```bash
export QA_API_KEY=...
uv run raw-pipeline process input/ output/ --config config.yaml
```

The key is read from the environment only. Never put it in the config file.

### What is sent

Only when `qa.provider` is not `none` **and** the key variable is set, one
HTTPS request per image goes to `{qa.endpoint}/chat/completions` with:

- a fixed text prompt;
- one JPEG preview, long edge at most `qa.preview_long_edge_px` and never above
  1024 px, with **no EXIF, GPS, ICC profile or comment**;
- the model name and generation settings.

Never sent: RAW files, full-resolution images, file names or paths, EXIF,
your configuration or other images. The key travels only in the
`Authorization` header and is never logged or written to the report.

Safeguards: there is no built-in endpoint (you must name the host); the
endpoint must be `https://` (plain `http://` only for `localhost`); redirects
are refused; oversized previews are refused before any request.

### Results

| Column | Meaning |
| --- | --- |
| `qa.closed_eyes`, `qa.blur`, `qa.skin_tone` | `true`/`false`; empty when not checked |
| `qa.flags` | all flags, `;`-separated |
| `qa.skipped` | `true` when no review happened |
| `qa.notes` | the model's one-line note, or why QA was skipped |

A missing key, network error or unexpected answer skips QA for that image
with a warning; the image is still exported. LLM answers may differ between
runs; pixel output does not.
