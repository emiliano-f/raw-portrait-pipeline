"""Center-weighted metering: the fallback when no face is found."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.registries import metering_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import ExposureMeasurement
from raw_pipeline.metering.stats import correction_ev, subsampled_luminance, weighted_median


@metering_registry.register("center_weighted")
class CenterWeightedMetering:
    """Gaussian-weighted median luminance around the image centre.

    The median (rather than the mean) keeps small specular highlights and
    deep shadows from pulling the exposure.

    Args:
        target_luminance: Scene-linear luminance the weighted median should reach.
        max_correction_ev: Absolute correction limit in EV stops.
        sigma_fraction: Gaussian sigma as a fraction of the image long edge.
    """

    name: ClassVar[str] = "center_weighted"

    def __init__(self, target_luminance: float, max_correction_ev: float, sigma_fraction: float) -> None:
        self.target_luminance = target_luminance
        self.max_correction_ev = max_correction_ev
        self.sigma_fraction = sigma_fraction

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from ``meter.target_center_luminance``, ``max_correction_ev`` and ``center_weight_sigma_fraction``."""
        cfg = app_config(env).meter
        return cls(cfg.target_center_luminance, cfg.max_correction_ev, cfg.center_weight_sigma_fraction)

    def measure(self, ctx: ImageContext) -> ExposureMeasurement:
        """Meter ``ctx.image`` (scene-linear); faces are ignored.

        Raises:
            MeteringError: The image is empty or has no finite pixel.
        """
        lum = subsampled_luminance(ctx.image)
        measured = weighted_median(lum, center_weights(lum.shape[0], lum.shape[1], self.sigma_fraction))
        return ExposureMeasurement(
            measured_luminance=measured,
            target_luminance=self.target_luminance,
            correction_ev=correction_ev(measured, self.target_luminance, self.max_correction_ev),
            method=self.name,
            faces_used=0,
        )


def center_weights(height: int, width: int, sigma_fraction: float) -> NDArray[np.float32]:
    """Isotropic Gaussian weights ``(height, width)`` centred on the image, peak 1."""
    sigma = sigma_fraction * max(height, width)
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / sigma
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / sigma
    return np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2)).astype(np.float32)
