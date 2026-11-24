# Configuration

`raw-pipeline` reads one YAML file passed with `--config`. Every option is
optional: an omitted option (or no `--config` at all) takes the default below.
Unknown keys, wrong types and out-of-range values fail before any image is
read, with a message naming the option. Unknown component names (a stage,
detector, exporter, ... that is not registered) fail the same way and list the
registered names.

A commented starting point is [`config.example.yaml`](https://github.com/emiliano-f/raw-portrait-pipeline/blob/main/config.example.yaml)
in the repository root.

!!! note "Generated reference"
    The tables on this page are generated at build time from the pydantic
    models in `raw_pipeline.config.models` (their JSON schema, the same one
    `raw_pipeline.config.schema.config_json_schema()` returns). To change a
    description, default or limit, edit the model field, not this page.

Units are part of the option names: `_ev` is EV stops (a factor of `2**ev`),
`_px` is pixels, `_s` is seconds, `_luminance` is scene-linear luminance
(Rec.709 weights, 1.0 = diffuse white). "Allowed" lists the numeric limits.

<!-- generated:config-reference -->

## Complete defaults

The configuration used when no file is given:

<!-- generated:config-defaults -->
