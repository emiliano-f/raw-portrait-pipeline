"""LibRaw adapter (rawpy): read the visible CFA and develop a (corrected) CFA in memory.

The file is opened read-only. :meth:`RawpyHandle.develop` copies the given CFA
into LibRaw's *in-memory* raw buffer (``raw_image_visible``) before
postprocessing; nothing is ever written back to disk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import rawpy
from numpy.typing import NDArray

from raw_pipeline.config import app_config
from raw_pipeline.core.errors import DecodeError
from raw_pipeline.core.raw import CFAFrame, DevelopedImage
from raw_pipeline.core.registries import raw_decoder_registry
from raw_pipeline.core.registry import BuildEnv

DEMOSAIC_ALGORITHMS: Final[Mapping[str, rawpy.DemosaicAlgorithm]] = {
    "ahd": rawpy.DemosaicAlgorithm.AHD,
    "aahd": rawpy.DemosaicAlgorithm.AAHD,
    "dcb": rawpy.DemosaicAlgorithm.DCB,
    "dht": rawpy.DemosaicAlgorithm.DHT,
    "linear": rawpy.DemosaicAlgorithm.LINEAR,
    "vng": rawpy.DemosaicAlgorithm.VNG,
    "ppg": rawpy.DemosaicAlgorithm.PPG,
}
"""Config names (``decode.demosaic``) to LibRaw demosaic algorithms."""

HIGHLIGHT_MODES: Final[Mapping[str, rawpy.HighlightMode]] = {
    "clip": rawpy.HighlightMode.Clip,
    "ignore": rawpy.HighlightMode.Ignore,
    "blend": rawpy.HighlightMode.Blend,
    "reconstruct": rawpy.HighlightMode.ReconstructDefault,
}
"""Config names (``decode.highlight_mode``) to LibRaw highlight modes."""

OUTPUT_BITS: Final = 16
OUTPUT_MAX: Final = 65535.0
LINEAR_GAMMA: Final = (1.0, 1.0)
NO_MAXIMUM_ADJUSTMENT: Final = 0.0
"""Disables LibRaw's per-image lowering of the white level, so 1.0 means the same in every image."""

_WB_CHANNELS: Final = 4
_FLIP_TRANSPOSE: Final = 4
_FLIP_VERTICAL: Final = 2
_FLIP_HORIZONTAL: Final = 1


class LibRawImage(Protocol):
    """The subset of :class:`rawpy.RawPy` used by the adapter (lets tests inject fakes)."""

    @property
    def raw_image_visible(self) -> NDArray[np.uint16]:
        """Writable view of the visible CFA area in LibRaw's memory."""
        ...

    @property
    def raw_colors_visible(self) -> NDArray[np.uint8]:
        """Colour index of every visible photosite."""
        ...

    @property
    def color_desc(self) -> bytes:
        """Channel letters, e.g. ``b"RGBG"``."""
        ...

    @property
    def black_level_per_channel(self) -> Sequence[int]:
        """Black level per colour index."""
        ...

    @property
    def white_level(self) -> int:
        """Saturation level in raw units."""
        ...

    @property
    def camera_whitebalance(self) -> Sequence[float]:
        """As-shot white balance multipliers (R, G, B, G2)."""
        ...

    @property
    def sizes(self) -> Any:  # noqa: ANN401 - rawpy.ImageSizes named tuple; only ``flip`` is read
        """Image sizes; ``flip`` is LibRaw's orientation code."""
        ...

    def postprocess(self, **kwargs: Any) -> NDArray[Any]:  # noqa: ANN401 - rawpy keyword API
        """Demosaic the in-memory raw buffer and return ``(H, W, 3)`` RGB."""
        ...

    def close(self) -> None:
        """Release LibRaw resources."""
        ...


Opener = Callable[[str], LibRawImage]
"""Opens a RAW file read-only; :func:`rawpy.imread` by default."""


@dataclass(frozen=True)
class DevelopSettings:
    """LibRaw postprocessing options derived from the ``decode`` config section.

    Attributes:
        demosaic: Key of :data:`DEMOSAIC_ALGORITHMS`.
        highlight_mode: Key of :data:`HIGHLIGHT_MODES`.
    """

    demosaic: str = "ahd"
    highlight_mode: str = "blend"

    def __post_init__(self) -> None:
        if self.demosaic not in DEMOSAIC_ALGORITHMS:
            msg = f"unknown demosaic '{self.demosaic}'; options: {', '.join(sorted(DEMOSAIC_ALGORITHMS))}"
            raise ValueError(msg)
        if self.highlight_mode not in HIGHLIGHT_MODES:
            msg = f"unknown highlight_mode '{self.highlight_mode}'; options: {', '.join(sorted(HIGHLIGHT_MODES))}"
            raise ValueError(msg)

    def postprocess_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`rawpy.RawPy.postprocess`.

        Camera white balance, 16-bit linear output (gamma 1), no auto-brightening,
        no per-image maximum adjustment, sRGB primaries, camera orientation.
        """
        return {
            "demosaic_algorithm": DEMOSAIC_ALGORITHMS[self.demosaic],
            "highlight_mode": HIGHLIGHT_MODES[self.highlight_mode],
            "use_camera_wb": True,
            "use_auto_wb": False,
            "output_color": rawpy.ColorSpace.sRGB,
            "output_bps": OUTPUT_BITS,
            "gamma": LINEAR_GAMMA,
            "no_auto_bright": True,
            "adjust_maximum_thr": NO_MAXIMUM_ADJUSTMENT,
        }

    def white_scale(self, camera_wb: Sequence[float]) -> float:
        """Factor that maps LibRaw's output so 1.0 is the white level of the least-amplified channel.

        With ``clip`` LibRaw normalises the WB multipliers by their minimum, so
        white already lands on 65535. With the other modes it normalises by
        their maximum to keep headroom, which darkens the image by
        ``max / min``; this factor undoes that, keeping highlights above 1.0.
        """
        if self.highlight_mode == "clip":
            return 1.0
        multipliers = normalized_wb(camera_wb)
        if multipliers is None:
            return 1.0
        return max(multipliers) / min(multipliers)


def normalized_wb(camera_wb: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Return the 4 WB multipliers (G2 defaults to G), or ``None`` when they are unusable."""
    values = [float(v) for v in camera_wb][:_WB_CHANNELS]
    if len(values) < _WB_CHANNELS - 1:
        return None
    if len(values) == _WB_CHANNELS - 1 or values[3] <= 0:
        values = [*values[:3], values[1]]
    if min(values) <= 0 or not all(np.isfinite(values)):
        return None
    return values[0], values[1], values[2], values[3]


def sensor_to_image(x: int, y: int, *, width: int, height: int, flip: int) -> tuple[int, int]:
    """Map visible-area sensor coordinates to developed-image coordinates.

    Implements LibRaw's (dcraw's) ``flip`` code: bit 4 transposes, bit 2 flips
    vertically and bit 1 flips horizontally, applied to the source as in
    ``flip_index``.

    Args:
        x: Sensor column.
        y: Sensor row.
        width: Visible sensor width in photosites.
        height: Visible sensor height in photosites.
        flip: LibRaw ``sizes.flip`` (0, 3, 5 or 6 for camera orientations).

    Returns:
        ``(x, y)`` in the developed image.

    Raises:
        ValueError: If ``(x, y)`` lies outside the sensor.
    """
    if not (0 <= x < width and 0 <= y < height):
        msg = f"sensor point ({x}, {y}) outside the {width}x{height} visible area"
        raise ValueError(msg)
    col = width - 1 - x if flip & _FLIP_HORIZONTAL else x
    row = height - 1 - y if flip & _FLIP_VERTICAL else y
    if flip & _FLIP_TRANSPOSE:
        return row, col
    return col, row


class RawpyHandle:
    """An opened RAW file (see :class:`~raw_pipeline.core.protocols.RawHandle`).

    Args:
        raw: The opened LibRaw image.
        settings: Postprocessing options.
    """

    def __init__(self, raw: LibRawImage, settings: DevelopSettings) -> None:
        self._raw = raw
        self._settings = settings

    def read_cfa(self) -> CFAFrame:
        """Return a copy of the visible Bayer data with its layout and levels."""
        colors = np.asarray(self._raw.raw_colors_visible)
        pattern = ((int(colors[0, 0]), int(colors[0, 1])), (int(colors[1, 0]), int(colors[1, 1])))
        return CFAFrame(
            data=np.array(self._raw.raw_image_visible, dtype=np.uint16, copy=True),
            pattern=pattern,
            color_desc=self._raw.color_desc.decode("ascii"),
            black_level=tuple(int(v) for v in self._raw.black_level_per_channel),
            white_level=int(self._raw.white_level),
        )

    def develop(self, cfa: CFAFrame) -> DevelopedImage:
        """Write ``cfa`` into LibRaw's in-memory buffer and demosaic it.

        Returns:
            Scene-linear float32 sRGB-primaries image; 1.0 = white level of the
            least-amplified channel after black subtraction and camera WB
            (see :meth:`DevelopSettings.white_scale`). Values above 1.0 are kept
            up to LibRaw's 16-bit output range.

        Raises:
            DecodeError: If ``cfa`` does not match the sensor or LibRaw fails.
        """
        buffer = self._raw.raw_image_visible
        if cfa.data.shape != buffer.shape:
            msg = f"CFA shape {cfa.data.shape} does not match the sensor {buffer.shape}"
            raise DecodeError(msg)
        buffer[...] = cfa.data
        try:
            rgb = self._raw.postprocess(**self._settings.postprocess_kwargs())
        except rawpy.LibRawError as exc:
            msg = f"LibRaw could not develop the image: {exc}"
            raise DecodeError(msg) from exc
        scale = self._settings.white_scale(self._raw.camera_whitebalance)
        image = (np.asarray(rgb, dtype=np.float32) * np.float32(scale / OUTPUT_MAX)).astype(np.float32)
        return DevelopedImage(image=image, params=self._params(cfa, scale, image))

    def sensor_to_image(self, x: int, y: int) -> tuple[int, int]:
        """Map raw sensor coordinates to developed-image coordinates (LibRaw flip)."""
        height, width = self._raw.raw_image_visible.shape
        return sensor_to_image(x, y, width=int(width), height=int(height), flip=self._flip())

    def _flip(self) -> int:
        return int(self._raw.sizes.flip)

    def _params(self, cfa: CFAFrame, scale: float, image: NDArray[np.float32]) -> dict[str, Any]:
        wb = normalized_wb(self._raw.camera_whitebalance)
        r, g, b, g2 = wb if wb is not None else (None, None, None, None)
        return {
            "decoder": "rawpy",
            "libraw_version": ".".join(str(v) for v in rawpy.libraw_version),
            "demosaic": self._settings.demosaic,
            "highlight_mode": self._settings.highlight_mode,
            "white_balance": "camera",
            "wb_r": r,
            "wb_g": g,
            "wb_b": b,
            "wb_g2": g2,
            "black_level": "/".join(str(v) for v in cfa.black_level),
            "white_level": cfa.white_level,
            "white_scale": scale,
            "flip": self._flip(),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }


def _imread(path: str) -> LibRawImage:
    raw: LibRawImage = rawpy.imread(path)
    return raw


@raw_decoder_registry.register("rawpy")
class RawpyDecoder:
    """RAW decoder backed by LibRaw through rawpy.

    Args:
        settings: Postprocessing options.
        opener: Opens a file read-only and returns a :class:`LibRawImage`
            (injected by tests; defaults to :func:`rawpy.imread`).
    """

    def __init__(self, settings: DevelopSettings | None = None, opener: Opener = _imread) -> None:
        self._settings = settings or DevelopSettings()
        self._opener = opener

    @classmethod
    def from_env(cls, env: BuildEnv) -> RawpyDecoder:
        """Build from the ``decode`` section of the configuration."""
        cfg = app_config(env).decode
        return cls(DevelopSettings(demosaic=cfg.demosaic, highlight_mode=cfg.highlight_mode))

    @contextmanager
    def open(self, path: Path) -> Iterator[RawpyHandle]:
        """Open ``path`` read-only and close it when the block exits.

        Raises:
            DecodeError: If the file is missing or LibRaw cannot read it.
        """
        if not path.is_file():
            msg = f"{path}: not a file"
            raise DecodeError(msg)
        try:
            raw = self._opener(str(path))
        except (rawpy.LibRawError, OSError) as exc:
            msg = f"{path}: LibRaw could not open the file: {exc}"
            raise DecodeError(msg) from exc
        try:
            yield RawpyHandle(raw, self._settings)
        finally:
            raw.close()
