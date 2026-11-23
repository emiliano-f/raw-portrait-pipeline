# AGENTS.md

Guidance for AI coding agents (and humans) working on this repository.

## Project overview

`raw-portrait-pipeline` is a command-line tool that batch-processes Canon RAW files (`.CR2`, Canon Rebel T7i / 800D) into finished JPG/TIFF images. It is optimized for portraits shot in bright but varying light conditions.

Default processing stages, in order:

1. **Decode**: RAW to scene-linear float image (rawpy / LibRaw), starting from the camera white balance.
2. **Meter**: detect faces and compute an exposure correction from face luminance. Fall back to center-weighted metering.
3. **Tone**: exposure correction, highlight recovery, gentle shadow lift (tone curve).
4. **Retouch** (optional): subtle skin smoothing limited to a skin mask.
5. **Export**: JPG and/or 16-bit TIFF.
6. **Report**: contact sheet and CSV with every applied parameter.
7. **QA** (optional): vision-LLM review of downscaled previews. Flags only, never edits.

Stages are pluggable: the list above is the default configuration, not hard-coded behavior.

## Non-negotiable invariants

- **Never modify, move, or delete input files.** Treat the input directory as read-only.
- **Never overwrite outputs silently.** Require `--overwrite` or write to a new filename.
- **The pipeline must run fully offline.** The LLM QA module is optional and must degrade gracefully (skip with a warning) when no API key is set or the API fails.
- **Never send RAW files to external services.** Send only downscaled JPG previews (long edge ≤ 1024 px), and only when QA is explicitly enabled.
- **Deterministic output.** The same input + config + version must produce byte-identical results. Seed any randomness.
- **No identity-altering edits.** Retouching may smooth skin tone and texture lightly. It must not reshape features or touch eyes, eyebrows, lips, or teeth geometry. No generative models may modify pixels.

## Package management (uv)

This project uses [uv](https://docs.astral.sh/uv/) exclusively. **Do not use `pip`, `pip-tools`, `poetry`, or `conda`**, and do not create virtualenvs by hand.

- `pyproject.toml` is the single source of truth for metadata, dependencies, and tool configuration (ruff, mypy, pytest).
- `uv.lock` is **committed** and must be updated in the same PR that changes dependencies.
- The Python version is pinned in `.python-version` (≥ 3.11).
- Runtime dependencies go in `[project.dependencies]`. Dev tools go in `[dependency-groups] dev`. Optional features are extras (e.g. `qa` for LLM provider SDKs, `docs` for the documentation toolchain).

```bash
uv sync                          # create/refresh .venv from uv.lock (incl. dev group)
uv sync --extra qa               # include optional QA provider SDKs
uv add <package>                 # add a runtime dependency (updates lock)
uv add --dev <package>           # add a dev dependency
uv remove <package>
uv lock --upgrade-package <pkg>  # upgrade one dependency deliberately

uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pytest -m "not slow"
uv run mkdocs build --strict     # documentation must build without warnings

uv run raw-pipeline process input/ output/ --config config.yaml
```

Always run tools through `uv run` so they use the locked environment. Justify every new dependency in the PR description. Prefer the standard library or an existing dependency when reasonable.

## Architecture

### Principles

- **SOLID**, with emphasis on single responsibility and dependency inversion: high-level orchestration depends on abstractions, never on concrete libraries.
- **Composition over inheritance.** Inheritance only from the abstract base classes / protocols defined in `core/`. No deep class hierarchies.
- **Explicit dependency injection.** Components receive their collaborators through constructors. No global state, no singletons, no module-level side effects on import.
- **Pure processing.** Stages transform data and return results. Only adapters in `io/` touch the filesystem or network.
- **Immutability where practical.** Configuration and per-image metadata are frozen dataclasses / pydantic models. Stages return new values instead of mutating shared state.

### Core abstractions (`src/raw_pipeline/core/`)

These contracts are the stable API of the project. Change them only through a dedicated PR with an ADR (see Documentation).

```python
@dataclass(frozen=True)
class ImageContext:
    source_path: Path
    image: NDArray[np.float32]  # (H, W, 3), scene-linear, working color space
    metadata: ImageMetadata  # EXIF + everything stages record
    artifacts: Mapping[str, Any]  # e.g. face boxes, masks; typed accessors preferred


class Stage(Protocol):
    name: ClassVar[str]

    def process(self, ctx: ImageContext) -> ImageContext: ...


class FaceDetector(Protocol):
    def detect(self, preview: NDArray[np.uint8]) -> list[FaceRegion]: ...


class MeteringStrategy(Protocol):
    def measure(self, ctx: ImageContext) -> ExposureMeasurement: ...


class Exporter(Protocol):
    def export(self, ctx: ImageContext, destination: Path) -> Path: ...


class QAProvider(Protocol):
    def review(self, preview_jpeg: bytes) -> QAResult: ...
```

### Patterns in use

- **Pipeline / Chain of Responsibility**: `Pipeline` holds an ordered list of `Stage` objects and runs each `ImageContext` through them. It knows nothing about what the stages do.
- **Strategy**: interchangeable algorithms behind one interface: `MeteringStrategy` (`FaceWeightedMetering`, `CenterWeightedMetering`), `FaceDetector` (`MediaPipeDetector`, `OpenCVDetector`), `Exporter` (`JpegExporter`, `TiffExporter`), `QAProvider` (one per vendor, plus `NullQAProvider`).
- **Registry + Factory**: each extension point has a registry mapping config names to classes. `PipelineFactory` builds the pipeline from validated config. Adding a component means registering it, not editing the factory.
- **Null Object**: disabled optional features (QA, retouch) use no-op implementations instead of `if enabled:` branches scattered through the code.
- **Adapter**: third-party libraries (rawpy, MediaPipe, OpenCV, LLM SDKs) are wrapped in adapters so they can be swapped or mocked.
- **Observer** (lightweight): progress and timing are reported through a `PipelineObserver` interface (CLI progress bar, logging, tests) instead of print statements inside stages.

### Repository layout

```
src/raw_pipeline/
  core/              # protocols, ImageContext, errors, registries. No third-party imaging deps.
  config/            # pydantic models, YAML loading, validation, schema export
  stages/            # one module per stage: decode, meter, tone, retouch, export, report, qa
  metering/          # MeteringStrategy implementations
  detection/         # FaceDetector adapters
  exporters/         # Exporter implementations
  qa/                # QAProvider adapters + NullQAProvider
  io/                # filesystem reading/writing, input discovery, output naming
  runtime/           # Pipeline, PipelineFactory, parallel BatchRunner, observers
  cli.py             # argument parsing only; delegates to runtime
tests/
  unit/              # mirrors src/ structure; synthetic data, fast
  integration/       # full pipeline on sample RAWs, @pytest.mark.slow
  contract/          # shared test suites every implementation of a protocol must pass
  fixtures/          # sample CR2s (small set, Git LFS)
docs/                # see Documentation
config.example.yaml
```

Dependency direction is strictly inward: `cli → runtime → stages / strategies → core`. `core/` imports nothing from the rest of the package. Enforce it with `import-linter` (configured in `pyproject.toml`) in CI.

## Extensibility

The software must be extensible **without modifying existing code** (open/closed principle). To add a component:

1. Implement the relevant protocol from `core/` in a new module.
2. Register it under a unique name with the registry decorator (e.g. `@stage_registry.register("vignette")`).
3. Add its config model (pydantic) with defaults and units documented in field descriptions.
4. Make it pass the matching **contract test suite** in `tests/contract/`, plus its own unit tests.
5. Document it (functional + technical docs).

Third-party plugins are discovered through Python entry points (`[project.entry-points."raw_pipeline.stages"]`, `...metering`, `...exporters`, `...qa`), so external packages can add components without forking.

Stage order and selection come from config:

```yaml
pipeline:
  stages: [decode, meter, tone, retouch, export, report, qa]
meter:
  strategy: face_weighted
  detector: mediapipe
  target_face_luminance: 0.18
```

Unknown component names fail at config validation time with a message listing the registered options.

## Image-processing conventions

- Internal representation: `numpy.float32`, shape `(H, W, 3)`, **scene-linear** RGB in a defined working space. Document which space in `docs/technical/color-pipeline.md`.
- Apply the transfer function (gamma) **only in exporters**. Never do exposure math on gamma-encoded data.
- Exposure adjustments are expressed in **EV stops** (multiply by `2**ev`), never as raw multipliers in config.
- Clip only at export. Preserve values > 1.0 through the pipeline for highlight recovery.
- Output color space is sRGB by default with an embedded ICC profile. Preserve useful EXIF (camera, lens, date, ISO, shutter, aperture) and strip GPS unless `keep_gps: true`.
- Face detection runs on a downscaled, gamma-encoded preview. Map coordinates back to full resolution explicitly, in one tested helper.
- Every stage records the parameters it applied in `ImageContext.metadata`. The report stage writes them to the CSV.

## Coding standards

- Full type hints, `mypy --strict` clean. Use `Protocol` for interfaces and `typing.Final` / frozen dataclasses for constants and value objects.
- Formatting and linting with `ruff`. Do not disable rules inline without a comment explaining why.
- Small classes with one responsibility and small methods. If a class needs "and" to describe it, split it.
- No magic numbers in processing code: name them as constants or config fields with units in the name (`target_face_luminance`, `highlight_rolloff_ev`).
- Use `logging` (not `print`) with per-image context. CLI verbosity via `-v/-q`.
- Custom exception hierarchy in `core/errors.py` (`PipelineError` → `DecodeError`, `ConfigError`, ...). Fail per image, not per batch: a corrupt file is logged, recorded in the report, and skipped. Exit code is non-zero if any image failed.
- Google-style docstrings on all public classes and methods: what it does, parameter units, and assumptions about input color space. These feed the API reference.

## Testing

- **Unit tests** use synthetic arrays (gradients, flat patches, synthetic "face" regions) to verify math: EV scaling, curve monotonicity, no NaNs/Infs, mask bounds. Collaborators are injected fakes, not patched globals.
- **Contract tests**: one reusable suite per protocol (e.g. every `Exporter` writes a valid file with the expected profile; every `Stage` preserves shape and dtype and does not mutate its input). New implementations must pass them.
- **Integration tests** run the full pipeline on the sample CR2s and assert:
  - outputs exist, have expected dimensions and color profile;
  - face luminance lands within tolerance of the configured target;
  - highlight clipping percentage stays below a threshold;
  - inputs are unchanged (hashes before/after).
- **Regression tests** compare outputs against stored reference images with a perceptual metric (e.g. SSIM ≥ threshold), not exact bytes. Update references only intentionally, and say so in the PR.
- No network calls in the test suite. QA providers are tested with fakes.
- Coverage target: ≥ 90% for `core/`, `config/`, `runtime/`. Bug fixes include a test that fails before the fix.

## Performance

- `BatchRunner` processes images in parallel with a configurable worker count (`--workers`, default = CPU count − 1). Stages must be picklable / stateless between images.
- Never hold the whole batch in memory. Stream one image per worker.
- Per-stage timing reported through the observer at debug level.

## Documentation

Documentation lives in `docs/` and is built with MkDocs (Material theme) + mkdocstrings. It must build with `uv run mkdocs build --strict` in CI.

```
docs/
  index.md
  functional/              # for users: what it does and how to use it
    getting-started.md     # install with uv, first run
    usage.md               # CLI commands and examples
    configuration.md       # every config option, defaults, units (generated from pydantic schema)
    stages.md              # what each stage does to the image, with before/after examples
    qa-and-privacy.md      # what the QA module sends, where, and how to disable it
    troubleshooting.md
  technical/               # for contributors: how it works
    architecture.md        # layers, dependency rules, diagrams (Mermaid)
    core-contracts.md      # protocols and their guarantees
    extending.md           # step-by-step guide to adding a stage/strategy/exporter/plugin
    color-pipeline.md      # color spaces, linear vs encoded, where conversions happen
    testing.md             # test layers, fixtures, updating references
    api/                   # API reference generated from docstrings
  adr/                     # Architecture Decision Records: NNNN-title.md
CHANGELOG.md               # Keep a Changelog format, Unreleased section on top
```

**Every PR updates the documentation it affects.** Specifically:

- Behavior or CLI change → `functional/` docs and `CHANGELOG.md`.
- New or changed config option → config model field description (the reference is generated from it) and `config.example.yaml`.
- New component → `functional/stages.md` (if user-visible) and `technical/extending.md` if the process changed.
- Change to a core contract, layering, or a significant technical choice → new ADR in `docs/adr/` (context, decision, consequences) and update `technical/architecture.md`.
- Pixel-affecting change → update before/after examples.

A PR with code changes and no doc changes must state in its description why no docs were affected.

## Multi-agent work (Devin)

When a task is large enough to split (e.g. the initial build, or a feature touching several modules), the main session acts as **coordinator** and distributes the work across parallel agent sessions, if they are available.

### Coordinator responsibilities

1. **Plan first.** Break the task into work packages aligned with module boundaries (see layout). Write the plan as `docs/plan/<task>.md`: packages, owners, dependencies between them, and acceptance criteria per package.
2. **Contracts before parallelism.** The coordinator implements and merges `core/` (protocols, `ImageContext`, errors, registries) and `config/` skeleton **before** dispatching parallel work. Workers build against these contracts and must not change them.
3. **Dispatch** each package to a worker agent with: the plan, the relevant AGENTS.md sections, the files/directories it owns, and its acceptance criteria.
4. **Integrate.** Merge packages in dependency order, run the full test suite after each merge, and resolve integration issues.
5. **Review.** Use a separate reviewer agent that did not write the code to review each PR against this file (invariants, architecture, tests, docs).

### Suggested work packages for the initial build

| Package | Owns | Depends on |
|---|---|---|
| Core contracts + config | `core/`, `config/` | none (done by coordinator first) |
| Decode + IO | `stages/decode.py`, `io/` | core |
| Detection + metering | `detection/`, `metering/`, `stages/meter.py` | core |
| Tone + retouch | `stages/tone.py`, `stages/retouch.py` | core |
| Export + report | `exporters/`, `stages/export.py`, `stages/report.py` | core |
| QA | `qa/`, `stages/qa.py` | core |
| Runtime + CLI | `runtime/`, `cli.py` | core, config |
| Docs + CI | `docs/`, `mkdocs.yml`, CI workflow | all (finalizes at the end) |

### Rules for worker agents

- Work on your own branch (`agent/<package>`) and open one PR per package.
- **Only edit files you own.** If you need a change elsewhere (especially in `core/`), stop and request it from the coordinator instead of editing.
- Add dependencies only with `uv add`, and mention them in the PR so the coordinator can resolve `uv.lock` conflicts.
- Your package is done only when it meets the Definition of done below, including its docs.
- Report back: what was built, test results, open questions, and any deviation from the plan.

## Security and privacy

- API keys come only from environment variables (e.g. `QA_API_KEY`). Never commit them, log them, or write them to reports.
- Portraits are personal data: no telemetry, no uploads outside the explicit QA step, and the docs must state what QA sends and where.
- Do not commit real client photos. Fixtures must be images the owner has approved for the repo.

## Git and PR workflow

- Small, focused commits with imperative messages (`Add face-weighted metering`). Conventional Commits prefixes are welcome (`feat:`, `fix:`, `docs:`, `refactor:`).
- One logical change per PR. The description includes: what changed, why, how it was tested, docs updated, and before/after contact sheets for any change that affects pixels.
- Do not refactor unrelated code in the same PR.
- CI runs on every PR: `uv sync --locked`, lint, format check, mypy, import-linter, tests, and `mkdocs build --strict`. All must pass to merge.

## Definition of done

- [ ] `uv sync --locked` works. `uv.lock` updated if dependencies changed.
- [ ] Lint, format, type check, import-linter, and all tests pass.
- [ ] New code follows the architecture: depends on `core/` abstractions, registered via registry, injected dependencies.
- [ ] New behavior covered by unit tests. New implementations pass their contract suite.
- [ ] Functional and technical docs updated, `CHANGELOG.md` updated, docs build with `--strict`.
- [ ] ADR added if a contract or architectural decision changed.
- [ ] Pixel-affecting changes include before/after.
- [ ] Invariants still hold (inputs untouched, offline-capable, deterministic).

## When unsure

Prefer the conservative choice: subtler edits, fewer dependencies, explicit failure over silent guessing. If a requirement is ambiguous and the choice affects image output or a core contract, stop and ask instead of deciding alone.