# Extending

Every interchangeable part is a class that satisfies a `Protocol` from
`raw_pipeline.core.protocols`, has a `from_env` classmethod and is registered
under a name. Users select it with that name in the config.

| Registry (`raw_pipeline.core.registries`) | Kind / entry-point group | Protocol | Selected by |
| --- | --- | --- | --- |
| `stage_registry` | `raw_pipeline.stages` | `Stage` | `pipeline.stages` |
| `raw_decoder_registry` | `raw_pipeline.raw_decoders` | `RawDecoder` | `decode.decoder` |
| `defect_corrector_registry` | `raw_pipeline.defect_correctors` | `DefectCorrector` | `defects.method` |
| `face_detector_registry` | `raw_pipeline.detectors` | `FaceDetector` | `meter.detector`, `meter.fallback_detector` |
| `metering_registry` | `raw_pipeline.metering` | `MeteringStrategy` | `meter.strategy`, `meter.fallback_strategy` |
| `retoucher_registry` | `raw_pipeline.retouchers` | `Retoucher` | `retouch.method` |
| `exporter_registry` | `raw_pipeline.exporters` | `Exporter` | `export.formats` |
| `qa_registry` | `raw_pipeline.qa` | `QAProvider` | `qa.provider` |
| `reporter_registry` | `raw_pipeline.reporters` | `BatchReporter` | `report.reporters` |

## Add a built-in component

Example: a new retoucher `clarity`.

1. **Read the protocol** in `raw_pipeline.core.protocols` (`Retoucher`) and the
   matching [component page](components/tone-retouch.md). Do not change `core`
   without an ADR.
2. **Add settings** to the config section the component reads
   (`raw_pipeline.config.models.RetouchConfig`), each with a `description`
   (the docs build fails without one) and limits (`ge`/`le`). Allowed names
   are validated against the registry, not with a `Literal`.
3. **Implement** the class in the owning package
   (`src/raw_pipeline/stages/retouchers/clarity.py`):

    ```python
    from dataclasses import dataclass
    from typing import ClassVar, Self

    from raw_pipeline.config import app_config
    from raw_pipeline.core.registries import retoucher_registry
    from raw_pipeline.core.registry import BuildEnv


    @retoucher_registry.register("clarity")
    @dataclass(frozen=True)
    class ClarityRetoucher:
        name: ClassVar[str] = "clarity"
        strength: float

        @classmethod
        def from_env(cls, env: BuildEnv) -> Self:
            return cls(app_config(env).retouch.strength)

        def retouch(self, image, faces):  # -> RetouchResult
            ...
    ```

    Keep it scene-linear float32, deterministic, picklable, and never mutate
    the input. Record every parameter you apply.
4. **Nothing else to wire**: `load_components` imports every module under the
   built-in packages, so the decorator runs. Keep imports inside the package's
   [layer](architecture.md#import-rules).
5. **Test** it: subclass the contract suite in `tests/contract/` if the
   protocol has one (here none; see `stage_contract.py` for stages), add unit
   tests next to the existing ones in `tests/unit/`, using
   `tests/support/synthetic.py` instead of real photos.
6. **Document** it on the component page and in
   [Stages](../functional/stages.md): the docs build fails if a registered
   name is missing there. Add a `CHANGELOG.md` line.
7. Run the [checks](testing.md#checks).

## Add a stage

A stage implements `process(ctx) -> ctx`, registers with
`@stage_registry.register("name")` and is enabled by adding its name to
`pipeline.stages`. Read the artifacts it needs with `ctx.artifact(KEY)`
(absent → `None`), publish new ones with `ctx.with_artifact(KEY, value)`, and
add new artifact keys to `raw_pipeline.core.artifacts` only through a core PR.
Subclass `tests.contract.stage_contract.StageContract`.

## Entry-point plugins

Third-party packages add components without touching this repository. Each
entry point's **name** is the registration name and its **value** is the
class (which must have `from_env`). The group is `raw_pipeline.<kind>` from
the table above. In the plugin's `pyproject.toml`:

```toml
[project.entry-points."raw_pipeline.retouchers"]
clarity = "my_plugin.clarity:ClarityRetoucher"
```

Install the plugin into the same environment (`uv add ./my-plugin` or
`uv pip install my-plugin` in the project venv) and select it in the config:

```yaml
retouch:
  method: clarity
plugins:
  clarity:            # free-form settings for the plugin
    radius_px: 4
```

`load_components` loads entry points after the built-ins, so a plugin cannot
replace a built-in name (it is skipped). Plugin settings live under
`plugins.<name>`; read them in `from_env` from `app_config(env).plugins` and
validate them yourself.
