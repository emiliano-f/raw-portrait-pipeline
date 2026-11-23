"""``frequency_separation`` retoucher: subtle skin smoothing inside a skin mask."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.registries import retoucher_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import FaceRegion, ParamRecord, RetouchResult, freeze_record
from raw_pipeline.stages.retouchers.skin_mask import SkinMaskBuilder

SMOOTHING_SIGMA_FACTOR: Final = 3.0
"""Sigma of the low-frequency smoothing relative to the frequency-split sigma."""

MIN_SPLIT_SIGMA_PX: Final = 0.5
"""Lower bound of the frequency-split Gaussian sigma, in pixels."""

GAUSSIAN_SUPPORT_SIGMAS: Final = 3.0
"""Region padding around a face, in smoothing sigmas, so blurs see real neighbours."""

_MIN_WEIGHT: Final = 1e-6


@retoucher_registry.register("frequency_separation")
@dataclass(frozen=True)
class FrequencySeparationRetoucher:
    """Frequency-separation-style smoothing of skin tone, keeping skin texture.

    For each face, the image is split into a low-frequency layer ``L`` (Gaussian,
    sigma = ``blur_radius_face_fraction`` x face box width) and the texture
    ``image - L``, which is kept untouched. Only ``L`` is smoothed further
    (sigma x :data:`SMOOTHING_SIGMA_FACTOR`), and the result is blended back as
    ``image + strength * mask * (smooth(L) - L)``. Blurs are normalised by the
    skin mask, so hair, eyes or background never bleed into the skin. Pixels
    where the mask is 0 (outside the face, eyes, brows, lips, nostrils) are
    returned bit-for-bit. Works on scene-linear working-space RGB; no clipping.

    Attributes:
        strength: Blend of the smoothed low-frequency layer, ``[0, 1]``.
        blur_radius_face_fraction: Frequency-split sigma as a fraction of the face box width.
        mask_builder: Builds the per-face skin mask (carries ``mask_feather_px``).
    """

    name: ClassVar[str] = "frequency_separation"
    strength: float
    blur_radius_face_fraction: float
    mask_builder: SkinMaskBuilder = field(default_factory=lambda: SkinMaskBuilder(feather_px=8))

    @classmethod
    def from_env(cls, env: BuildEnv) -> Self:
        """Build from the ``retouch`` section of the configuration."""
        cfg = app_config(env).retouch
        return cls(
            strength=cfg.strength,
            blur_radius_face_fraction=cfg.blur_radius_face_fraction,
            mask_builder=SkinMaskBuilder(feather_px=cfg.mask_feather_px),
        )

    def retouch(self, image: NDArray[np.float32], faces: Sequence[FaceRegion]) -> RetouchResult:
        """Smooth skin of every face in ``image`` (full-resolution coordinates).

        Returns the input array itself when there is nothing to do (no faces or
        zero strength); otherwise a new array.
        """
        if self.strength == 0 or not faces or image.size == 0:
            return RetouchResult(image, self._params(0.0, 0, "none", 0.0))
        out = image.copy()
        covered = np.zeros(image.shape[:2], dtype=np.bool_)
        sources: set[str] = set()
        max_sigma = 0.0
        for face in faces:
            sigma = max(MIN_SPLIT_SIGMA_PX, self.blur_radius_face_fraction * face.box.width)
            region = _face_region(face, sigma * SMOOTHING_SIGMA_FACTOR * GAUSSIAN_SUPPORT_SIGMAS, image.shape[:2])
            if region is None:
                continue
            rows, cols = region
            origin, size = (cols.start, rows.start), (rows.stop - rows.start, cols.stop - cols.start)
            mask = self.mask_builder.build(face, origin, size)
            if not mask.hard.any():
                continue
            out[rows, cols] = self._smooth(out[rows, cols], mask.hard, mask.soft, sigma)
            covered[rows, cols] |= mask.soft > 0
            sources.add(mask.source)
            max_sigma = max(max_sigma, sigma)
        if not sources:
            return RetouchResult(image, self._params(0.0, 0, "none", 0.0))
        source = sources.pop() if len(sources) == 1 else "mixed"
        return RetouchResult(out, self._params(float(covered.mean()), len(faces), source, max_sigma))

    def _smooth(
        self, crop: NDArray[np.float32], hard: NDArray[np.bool_], soft: NDArray[np.float32], sigma: float
    ) -> NDArray[np.float32]:
        weight = hard.astype(np.float32)
        low = _masked_blur(crop, weight, sigma)
        smooth = _masked_blur(low, weight, sigma * SMOOTHING_SIGMA_FACTOR)
        blend = (np.float32(self.strength) * soft)[..., None]
        retouched = crop + blend * (smooth - low)
        return np.where(soft[..., None] > 0, retouched, crop).astype(np.float32)

    def _params(self, coverage: float, faces: int, source: str, sigma: float) -> ParamRecord:
        applied = self.strength if faces else 0.0
        return freeze_record(
            {
                "strength": applied,
                "mask_coverage": coverage,
                "faces_retouched": faces,
                "mask_source": source,
                "blur_radius_px": round(sigma, 3),
                "mask_feather_px": self.mask_builder.feather_px,
            }
        )


def _face_region(face: FaceRegion, pad_px: float, shape: tuple[int, ...]) -> tuple[slice, slice] | None:
    height, width = shape[0], shape[1]
    x0, y0, x1, y1 = face.box.x0, face.box.y0, face.box.x1, face.box.y1
    if face.landmarks is not None and face.landmarks.size and np.isfinite(face.landmarks).all():
        x0, y0 = min(x0, float(face.landmarks[:, 0].min())), min(y0, float(face.landmarks[:, 1].min()))
        x1, y1 = max(x1, float(face.landmarks[:, 0].max())), max(y1, float(face.landmarks[:, 1].max()))
    c0, r0 = max(0, math.floor(x0 - pad_px)), max(0, math.floor(y0 - pad_px))
    c1, r1 = min(width, math.ceil(x1 + pad_px)), min(height, math.ceil(y1 + pad_px))
    if c1 <= c0 or r1 <= r0:
        return None
    return slice(r0, r1), slice(c0, c1)


def _masked_blur(image: NDArray[np.float32], weight: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Gaussian blur of ``image`` using only pixels where ``weight`` > 0 (normalised convolution)."""
    num = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(weight, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)[..., None]
    out = np.zeros_like(image)
    np.divide(num, den, out=out, where=den > _MIN_WEIGHT)
    return out
