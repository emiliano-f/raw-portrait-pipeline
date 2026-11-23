"""Fake components for runtime tests (module-level so worker processes can unpickle them)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import numpy as np

from raw_pipeline.config.models import AppConfig, DefectDetectionConfig
from raw_pipeline.config.validation import component_references
from raw_pipeline.core.artifacts import OUTPUT_PATHS, REPORT_THUMBNAILS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import ConfigError, DecodeError, OutputExistsError
from raw_pipeline.core.protocols import ImageOutcome
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, DefectMap, DetectedDefect, PixelDefect
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.services import RuntimeServices
from tests.support.synthetic import cfa_frame

STAGES = ("fake_decode", "fake_brighten", "fake_export")


def fake_config(**sections: object) -> AppConfig:
    """Config naming the fake components; ``sections`` override top-level sections."""
    data: dict[str, object] = {
        "pipeline": {"stages": list(STAGES)},
        "report": {"reporters": ["fake_summary"]},
        "decode": {"decoder": "fake_raw"},
    }
    data.update(sections)
    return AppConfig.model_validate(data)


class FakeDecode:
    """Decodes the file bytes into a flat image; files starting with ``BAD`` fail."""

    name: ClassVar[str] = "fake_decode"

    @classmethod
    def from_env(cls, env: BuildEnv) -> FakeDecode:
        del env
        return cls()

    def process(self, ctx: ImageContext) -> ImageContext:
        data = ctx.source_path.read_bytes()
        if data.startswith(b"BAD"):
            msg = f"cannot decode {ctx.source_path.name}"
            raise DecodeError(msg)
        value = hashlib.sha256(data).digest()[0] / 255.0
        image = np.full((8, 12, 3), value, dtype=np.float32)
        return ctx.with_image(image).with_record(self.name, {"value": round(value, 6), "bytes": len(data)})


class FakeBrighten:
    """Multiplies the image by 2 (EV +1); files containing ``CRASH`` raise a non-pipeline error."""

    name: ClassVar[str] = "fake_brighten"

    @classmethod
    def from_env(cls, env: BuildEnv) -> FakeBrighten:
        del env
        return cls()

    def process(self, ctx: ImageContext) -> ImageContext:
        if b"CRASH" in ctx.source_path.read_bytes():
            msg = "boom"
            raise ValueError(msg)
        thumbs = {"after": np.zeros((2, 2, 3), dtype=np.uint8)}
        return (
            ctx.with_image(ctx.image * np.float32(2.0))
            .with_record(self.name, {"ev": 1.0})
            .with_artifact(REPORT_THUMBNAILS, thumbs)
        )


class FakeExport:
    """Writes ``<stem>.txt`` into the output dir; refuses to overwrite unless allowed."""

    name: ClassVar[str] = "fake_export"

    def __init__(self, output_dir: Path, *, overwrite: bool) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite

    @classmethod
    def from_env(cls, env: BuildEnv) -> FakeExport:
        return cls(env.output_dir, overwrite=env.overwrite)

    def process(self, ctx: ImageContext) -> ImageContext:
        path = self.output_dir / f"{ctx.source_path.stem}.txt"
        if path.exists() and not self.overwrite:
            msg = f"{path} exists"
            raise OutputExistsError(msg)
        path.write_text(f"{float(ctx.image.mean()):.6f}\n", encoding="utf-8")
        return ctx.with_artifact(OUTPUT_PATHS, (path,)).with_record(self.name, {"path": path.name})


class FakeSummary:
    """Batch reporter writing one line per outcome, in the order received."""

    name: ClassVar[str] = "fake_summary"

    def __init__(self, *, overwrite: bool) -> None:
        self.overwrite = overwrite
        self.lines: list[str] = []

    @classmethod
    def from_env(cls, env: BuildEnv) -> FakeSummary:
        return cls(overwrite=env.overwrite)

    def add(self, outcome: ImageOutcome) -> None:
        flat = outcome.metadata.flatten()
        cells = [outcome.source_path.name, str(outcome.ok), str(outcome.error_type), *map(str, sorted(flat.items()))]
        self.lines.append(";".join(cells))

    def finalize(self, output_dir: Path) -> list[Path]:
        path = output_dir / "summary.txt"
        if path.exists() and not self.overwrite:
            msg = f"{path} exists"
            raise OutputExistsError(msg)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return [path]


class Inert:
    """Placeholder for config references the test does not exercise."""

    name: ClassVar[str] = "inert"

    @classmethod
    def from_env(cls, env: BuildEnv) -> Inert:
        del env
        return cls()


class RejectsConfig(Inert):
    """Stage whose ``from_env`` rejects the config, like the QA stage without endpoint/model."""

    name: ClassVar[str] = "rejects_config"

    @classmethod
    def from_env(cls, env: BuildEnv) -> RejectsConfig:
        del env
        msg = "qa.endpoint and qa.model are required when qa.provider is not 'none'"
        raise ConfigError(msg)


class FakeHandle:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_cfa(self) -> CFAFrame:
        return cfa_frame(seed=len(self.path.name))

    def develop(self, cfa: CFAFrame) -> object:  # pragma: no cover - not used by the runtime
        raise NotImplementedError

    def sensor_to_image(self, x: int, y: int) -> tuple[int, int]:  # pragma: no cover
        return x, y


class FakeDecoder:
    """RAW decoder whose handles return synthetic CFA frames; ``BAD`` files fail to open."""

    name: ClassVar[str] = "fake_raw"

    @classmethod
    def from_env(cls, env: BuildEnv) -> FakeDecoder:
        del env
        return cls()

    @contextmanager
    def open(self, path: Path) -> Iterator[FakeHandle]:
        if path.read_bytes().startswith(b"BAD"):
            msg = f"cannot open {path.name}"
            raise DecodeError(msg)
        yield FakeHandle(path)


class FakeDetector:
    """Reports a fixed column and pixel defect, counting the frames it saw."""

    def __init__(self, config: DefectDetectionConfig) -> None:
        self.config = config
        self.frames = 0

    def detect(self, frames: Iterable[CFAFrame]) -> list[DetectedDefect]:
        self.frames = sum(1 for _ in frames)
        if self.frames == 0:
            return []
        return [
            DetectedDefect(ColumnDefect(index=21, width=2), 42.0, self.frames, self.frames),
            DetectedDefect(PixelDefect(x=7, y=9), 12.5, self.frames, self.frames),
        ]


class EmptyDetector(FakeDetector):
    def detect(self, frames: Iterable[CFAFrame]) -> list[DetectedDefect]:
        self.frames = sum(1 for _ in frames)
        return []


def make_registries(config: AppConfig) -> ComponentRegistries:
    """Fresh registries with the fakes plus an inert placeholder for every other name in ``config``."""
    regs = ComponentRegistries.empty()
    for cls in (FakeDecode, FakeBrighten, FakeExport):
        regs.stages.register(cls.name)(cls)
    regs.reporters.register(FakeSummary.name)(FakeSummary)
    regs.stages.register(RejectsConfig.name)(RejectsConfig)
    regs.raw_decoders.register(FakeDecoder.name)(FakeDecoder)
    if config.decode.decoder not in regs.raw_decoders:
        regs.raw_decoders.register(config.decode.decoder)(FakeDecoder)
    by_kind = {r.kind: r for r in regs.all()}
    for _, kind, name in component_references(config):
        if name not in by_kind[kind]:
            by_kind[kind].register(name)(Inert)
    return regs


class FakeServices:
    """Recording fakes of the io/defects collaborators."""

    def __init__(self, detector_cls: type[FakeDetector] = FakeDetector) -> None:
        self.detector_cls = detector_cls
        self.detectors: list[FakeDetector] = []
        self.written: list[tuple[DefectMap, Path, bool]] = []
        self.merged: list[tuple[DefectMap, Path]] = []
        self.excluded: list[tuple[Path, ...]] = []

    def discover(self, input_dir: Path, *, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
        """Recursive, like ``discover_raw_files(recursive=True)``, so exclusion matters."""
        self.excluded.append(tuple(exclude_dirs))
        skip = [d.resolve() for d in exclude_dirs]
        found = (p for p in input_dir.rglob("*") if p.suffix.lower() == ".cr2" and p.is_file())
        return sorted(p for p in found if not any(d in p.resolve().parents for d in skip))

    def write(self, defects: DefectMap, path: Path, *, overwrite: bool) -> Path:
        self.written.append((defects, path, overwrite))
        return path

    def merge(self, defects: DefectMap, config_path: Path) -> None:
        self.merged.append((defects, config_path))

    def make_detector(self, config: DefectDetectionConfig) -> FakeDetector:
        detector = self.detector_cls(config)
        self.detectors.append(detector)
        return detector

    def services(self) -> RuntimeServices:
        return RuntimeServices(
            discover=self.discover,
            write_defect_map=self.write,
            merge_defect_map=self.merge,
            make_defect_detector=self.make_detector,
        )


def write_inputs(folder: Path, contents: dict[str, bytes]) -> list[Path]:
    """Create fake RAW files; returns them sorted."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (folder / name).write_bytes(data)
    return sorted(folder / name for name in contents)
