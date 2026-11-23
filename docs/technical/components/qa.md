# QA (vision-LLM review)

Optional review of each processed image by a vision-capable LLM. It only
**flags** problems in the CSV report (`qa.closed_eyes`, `qa.blur`,
`qa.skin_tone`); it never changes pixels. The pipeline runs fully offline
without it.

## Components

| Name | Kind | Class | Notes |
|---|---|---|---|
| `qa` | stage | `raw_pipeline.stages.qa.QAStage` | builds the preview, calls the provider, records `qa.*` |
| `none` | qa provider | `raw_pipeline.qa.null.NullQAProvider` | Null Object, default; never contacts anything |
| `openai_compatible` | qa provider | `raw_pipeline.qa.openai_compatible.OpenAICompatibleProvider` | any chat-completions API with image input |

Supporting modules: `qa/preview.py` (preview encoder and checks),
`qa/response.py` (prompt and answer parsing), `qa/transport.py` (HTTP adapter
over the standard library `urllib`; tests inject fakes).

## What is sent externally

Only when `qa.provider` is not `none` **and** the API key variable
(`qa.api_key_env`, default `QA_API_KEY`) is set, one HTTPS `POST` per image goes
to `{qa.endpoint}/chat/completions` (default endpoint
`https://api.openai.com/v1`, default model `gpt-4o-mini`) containing:

- a fixed text prompt (`SYSTEM_PROMPT`, `USER_PROMPT` in `qa/response.py`);
- one JPEG preview as a base64 `data:` URL: long edge at most
  `qa.preview_long_edge_px` and **never above 1024 px**, sRGB, quality
  `qa.preview_jpeg_quality`, with **no EXIF, GPS, ICC profile or comment**;
- the model name, `temperature: 0` and `max_tokens`.

Never sent: RAW files, full-resolution pixels, file names or paths, EXIF
(camera, date, GPS), configuration or other images. The API key is sent only
in the `Authorization: Bearer` header; it is never logged, written to the
report or included in `repr()`.

Safeguards:

- the stage encodes the preview itself (downscale in linear light, then sRGB);
- the provider refuses anything that is not a JPEG or whose long edge exceeds
  1024 px, before any request is made (`MAX_PREVIEW_LONG_EDGE_PX`);
- `qa.endpoint` must be `https://`; plain `http://` is accepted only for
  `localhost` / `127.0.0.1` / `::1` (self-hosted models). Anything else is a
  `ConfigError` when the pipeline is built;
- HTTP redirects are refused (an error, so QA is skipped): following one would
  resend the bearer token to a host, or a plain-HTTP URL, that was never
  validated;
- an API key with whitespace, control or non-ASCII characters is treated as
  missing, so it can never end up in an HTTP header error message;
- responses larger than 1 MiB are rejected.

## Failure handling

QA never fails an image. Each case below yields `QAResult(skipped=True)` with
the reason in `qa.notes`:

| Situation | Logged |
|---|---|
| `qa.provider: none` | nothing (debug) |
| API key variable unset, empty or with invalid characters | one warning when the provider is built |
| network error, timeout, HTTP error status, redirect | warning per image |
| answer is not the expected JSON | warning per image |
| provider plugin raises | warning per image (the stage catches it) |

## CSV columns (`qa` record)

| Column | Value |
|---|---|
| `qa.provider`, `qa.model` | provider name and model |
| `qa.skipped` | `true` when no review happened |
| `qa.closed_eyes`, `qa.blur`, `qa.skin_tone` | `true`/`false`; empty when skipped (not checked) |
| `qa.flags` | all flags joined by `;` |
| `qa.notes` | one-line explanation from the model (≤ 300 chars) or the skip reason |

The model is asked for `{"flags": [...], "notes": "..."}`; code fences or
surrounding prose are tolerated and unknown flags are dropped. LLM answers are
not guaranteed deterministic even at `temperature: 0`; pixel output is
unaffected.

## Adding a provider

Implement `QAProvider.review(preview_jpeg) -> QAResult` (never raise; return
`skipped=True` and warn), register it with `@qa_registry.register("<name>")`
or the `raw_pipeline.qa` entry-point group, read settings in `from_env`, and
subclass `tests.contract.qa_provider_contract.QAProviderContract` with a fake
transport. Use `raw_pipeline.qa.preview.inspect_preview` to enforce the size
limit.

## API

::: raw_pipeline.stages.qa
::: raw_pipeline.qa.null
::: raw_pipeline.qa.openai_compatible
::: raw_pipeline.qa.preview
::: raw_pipeline.qa.response
::: raw_pipeline.qa.transport
