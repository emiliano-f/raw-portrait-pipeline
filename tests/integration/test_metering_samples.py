"""Metered luminance lands on its target after the tone stage, on the sample CR2s (slow)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from raw_pipeline.config.models import AppConfig
from raw_pipeline.core.artifacts import EXPOSURE, FACES
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.runtime.components import load_components
from tests.integration.conftest import SAMPLES_ENV, sample_files

pytestmark = pytest.mark.slow

TOLERANCE_EV = 0.1
"""Allowed distance from the target, in EV, of the luminance re-measured after tone."""

UNTIL_TONE = ("decode", "meter", "tone")

SAMPLES = sample_files()
per_sample = pytest.mark.parametrize(
    "sample",
    [pytest.param(p, id=p.name) for p in SAMPLES]
    or [pytest.param(None, marks=pytest.mark.skip(reason=f"no sample CR2 (set {SAMPLES_ENV})"))],
)


def run_until_tone(sample: Path, env: BuildEnv) -> ImageContext:
    registries = load_components()
    ctx = ImageContext.initial(sample)
    for name in UNTIL_TONE:
        ctx = registries.stages.create(name, env).process(ctx)
    return ctx


def assert_lands_on_target(sample: Path, *, with_faces: bool, tmp_path: Path) -> None:
    config = AppConfig()
    env = BuildEnv(config=config, output_dir=tmp_path)
    ctx = run_until_tone(sample, env)
    faces = ctx.artifact(FACES) or ()
    if bool(faces) != with_faces:
        pytest.skip(f"{sample.name}: {'no face' if with_faces else 'a face'} found")
    exposure = ctx.artifact(EXPOSURE)
    assert exposure is not None
    if abs(exposure.correction_ev) >= config.meter.max_correction_ev:
        pytest.skip(f"{sample.name}: correction clamped at {exposure.correction_ev:+.2f} EV")

    strategy_name = config.meter.strategy if with_faces else config.meter.fallback_strategy
    assert exposure.method == strategy_name
    after = load_components().metering.create(strategy_name, env).measure(ctx)
    full_ev = math.log2(exposure.target_luminance / exposure.measured_luminance)
    if exposure.correction_ev < full_ev - TOLERANCE_EV:
        # Lowered by meter.face_highlight_ceiling: the face stays below target, never above.
        assert after.measured_luminance < exposure.target_luminance
        return
    assert abs(math.log2(after.measured_luminance / exposure.target_luminance)) < TOLERANCE_EV


@per_sample
def test_face_luminance_lands_on_target(sample: Path, tmp_path: Path) -> None:
    assert_lands_on_target(sample, with_faces=True, tmp_path=tmp_path)


@per_sample
def test_center_luminance_lands_on_target_without_faces(sample: Path, tmp_path: Path) -> None:
    assert_lands_on_target(sample, with_faces=False, tmp_path=tmp_path)
