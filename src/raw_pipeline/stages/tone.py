"""Tone stage: exposure correction, highlight shoulder and shadow lift (scene-linear)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.artifacts import EXPOSURE
from raw_pipeline.core.color import luminance
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.registries import stage_registry
from raw_pipeline.core.registry import BuildEnv

DIFFUSE_WHITE: Final = 1.0
"""Scene-linear value of diffuse white in the working space."""

MAX_SHADOW_LIFT_EV: Final = 2.0
"""Largest ``shadow_lift_ev`` (inclusive, EV); same bound as ``ToneConfig``, under the
strict-monotonicity limit ``2 / ln 2``."""

_MIN_LUMINANCE: Final = 1e-12
"""Below this luminance the per-pixel gain is taken as its limit at zero."""


@dataclass(frozen=True)
class ToneCurve:
    """Monotonic, C1-continuous tone curve applied to scene-linear luminance.

    The curve is ``shoulder(lift(Y))``:

    * ``lift``: ``Y * 2**(shadow_lift_ev * (1 - Y / pivot)**2)`` for ``Y < pivot``,
      identity above. Deep shadows gain ``shadow_lift_ev``, the gain fades to 0 EV
      with zero slope at the pivot.
    * ``shoulder``: identity up to ``s = 2**highlight_rolloff_start_ev``; above it a
      rational shoulder ``s + d / (1 + d / k)`` (``d = Y - s``) with slope 1 at ``s``,
      chosen so that ``2**highlight_rolloff_ev`` (scene headroom above diffuse
      white) lands exactly on 1.0. Values beyond keep rising towards ``s + k``, so
      nothing is clipped before export.

    Between ``pivot`` and ``s`` the curve is the identity, so exposure scaling is
    exact in the mid-tones.

    Attributes:
        highlight_rolloff_start_ev: Shoulder start in EV relative to diffuse white; must be < 0.
        highlight_rolloff_ev: Scene headroom above diffuse white mapped to 1.0, in EV (> 0).
        shadow_lift_ev: Gain of the deepest shadows in EV, ``0 <= lift <= MAX_SHADOW_LIFT_EV``.
        shadow_pivot_luminance: Scene-linear luminance where the lift reaches 0 EV (> 0).
    """

    highlight_rolloff_start_ev: float
    highlight_rolloff_ev: float
    shadow_lift_ev: float
    shadow_pivot_luminance: float

    def __post_init__(self) -> None:
        if self.highlight_rolloff_start_ev >= 0:
            msg = f"tone.highlight_rolloff_start_ev must be < 0 (below white), got {self.highlight_rolloff_start_ev}"
            raise ConfigError(msg)
        if self.highlight_rolloff_ev <= 0:
            msg = f"tone.highlight_rolloff_ev must be > 0, got {self.highlight_rolloff_ev}"
            raise ConfigError(msg)
        if not 0 <= self.shadow_lift_ev <= MAX_SHADOW_LIFT_EV:
            msg = f"tone.shadow_lift_ev must be in [0, {MAX_SHADOW_LIFT_EV:g}] EV, got {self.shadow_lift_ev}"
            raise ConfigError(msg)
        if self.shadow_pivot_luminance <= 0:
            msg = f"tone.shadow_pivot_luminance must be > 0, got {self.shadow_pivot_luminance}"
            raise ConfigError(msg)

    @property
    def shoulder_start(self) -> float:
        """Scene-linear luminance where the highlight shoulder starts."""
        return DIFFUSE_WHITE * math.pow(2.0, self.highlight_rolloff_start_ev)

    @property
    def scene_white(self) -> float:
        """Scene-linear luminance mapped to diffuse white (1.0)."""
        return DIFFUSE_WHITE * math.pow(2.0, self.highlight_rolloff_ev)

    @property
    def shoulder_scale(self) -> float:
        """Shoulder constant ``k``; the curve approaches ``shoulder_start + k`` asymptotically."""
        s, w = self.shoulder_start, self.scene_white
        return (w - s) * (DIFFUSE_WHITE - s) / (w - DIFFUSE_WHITE)

    def apply(self, lum: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map scene-linear luminance through the curve (same shape, float32).

        The two knees are evaluated in float64 on the affected pixels only, so
        float32 rounding cannot break monotonicity near the shoulder asymptote.
        """
        out = np.array(lum, dtype=np.float32, copy=True)
        shadows = out < self.shadow_pivot_luminance
        y = out[shadows].astype(np.float64)
        t = np.clip(y / self.shadow_pivot_luminance, 0.0, 1.0)
        out[shadows] = y * np.exp2(self.shadow_lift_ev * np.square(1.0 - t))
        s = self.shoulder_start
        highlights = out > s
        d = out[highlights].astype(np.float64) - s
        out[highlights] = s + d / (1.0 + d / self.shoulder_scale)
        return out

    def gain(self, lum: NDArray[np.float32]) -> NDArray[np.float32]:
        """Per-pixel multiplicative gain ``apply(Y) / Y``; its limit ``2**shadow_lift_ev`` at ``Y <= 0``."""
        mapped = self.apply(lum)
        zero_gain = np.float32(math.pow(2.0, self.shadow_lift_ev))
        safe = lum > _MIN_LUMINANCE
        out = np.full(lum.shape, zero_gain, dtype=np.float32)
        np.divide(mapped, lum, out=out, where=safe)
        return out


@stage_registry.register("tone")
@dataclass(frozen=True)
class ToneStage:
    """Exposure correction from ``EXPOSURE`` followed by the :class:`ToneCurve`.

    The image is multiplied by ``2**correction_ev`` (0 EV when no measurement is
    present), then every pixel is scaled by ``curve(Y) / Y`` where ``Y`` is its
    Rec.709 luminance, which keeps RGB ratios (hue and saturation) intact.
    Input and output are scene-linear working-space RGB; nothing is clipped.

    Attributes:
        curve: Tone curve applied after the exposure correction.
    """

    name: ClassVar[str] = "tone"
    curve: ToneCurve

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``tone`` section of the configuration."""
        cfg = app_config(env).tone
        return cls(
            ToneCurve(
                highlight_rolloff_start_ev=cfg.highlight_rolloff_start_ev,
                highlight_rolloff_ev=cfg.highlight_rolloff_ev,
                shadow_lift_ev=cfg.shadow_lift_ev,
                shadow_pivot_luminance=cfg.shadow_pivot_luminance,
            )
        )

    def process(self, ctx: ImageContext) -> ImageContext:
        """Return ``ctx`` with exposure and tone curve applied and parameters recorded."""
        exposure = ctx.artifact(EXPOSURE)
        ev = exposure.correction_ev if exposure is not None else 0.0
        exposed = ctx.image * np.float32(math.pow(2.0, ev))
        toned = (exposed * self.curve.gain(luminance(exposed))[..., None]).astype(np.float32)
        clipped = float(np.mean(np.max(toned, axis=2) > DIFFUSE_WHITE)) if toned.size else 0.0
        return ctx.with_image(toned).with_record(
            self.name,
            {
                "exposure_ev": float(ev),
                "exposure_method": exposure.method if exposure is not None else "none",
                "highlight_rolloff_start_ev": self.curve.highlight_rolloff_start_ev,
                "highlight_rolloff_ev": self.curve.highlight_rolloff_ev,
                "shadow_lift_ev": self.curve.shadow_lift_ev,
                "shadow_pivot_luminance": self.curve.shadow_pivot_luminance,
                "clipped_fraction": clipped,
            },
        )
