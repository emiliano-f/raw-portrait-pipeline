from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rawpy
from numpy.typing import NDArray

from raw_pipeline.config import parse_config
from raw_pipeline.core.errors import DecodeError
from raw_pipeline.core.registries import raw_decoder_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.io.rawpy_decoder import (
    OUTPUT_MAX,
    DevelopSettings,
    RawpyDecoder,
    RawpyHandle,
    normalized_wb,
    sensor_to_image,
)
from tests.support.synthetic import BLACK_LEVEL, RGGB, WHITE_LEVEL, cfa_frame

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SAMPLES = sorted(FIXTURES.glob("*.CR2"))
WB = (2.0, 1.0, 1.5, 1.0)


class FakeLibRaw:
    """In-memory stand-in for rawpy.RawPy; postprocess echoes the buffer into RGB."""

    def __init__(self, height: int = 8, width: int = 12, flip: int = 0, wb: Sequence[float] = WB) -> None:
        self.raw_image_visible = np.arange(height * width, dtype=np.uint16).reshape(height, width) + 1000
        self.raw_colors_visible = np.tile(np.asarray(RGGB, dtype=np.uint8), (height // 2, width // 2))
        self.color_desc = b"RGBG"
        self.black_level_per_channel = [BLACK_LEVEL] * 4
        self.white_level = WHITE_LEVEL
        self.camera_whitebalance = list(wb)
        self.sizes = SimpleNamespace(flip=flip)
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.fail: Exception | None = None

    def postprocess(self, **kwargs: Any) -> NDArray[np.uint16]:
        if self.fail is not None:
            raise self.fail
        self.calls.append(kwargs)
        return np.repeat(self.raw_image_visible[:, :, None], 3, axis=2).copy()

    def close(self) -> None:
        self.closed = True


def handle(raw: FakeLibRaw, highlight_mode: str = "blend") -> RawpyHandle:
    return RawpyHandle(raw, DevelopSettings(highlight_mode=highlight_mode))


def test_read_cfa_returns_a_copy_with_layout_and_levels() -> None:
    raw = FakeLibRaw()
    cfa = handle(raw).read_cfa()
    assert cfa.pattern == RGGB
    assert cfa.color_desc == "RGBG"
    assert cfa.black_level == (BLACK_LEVEL,) * 4
    assert cfa.white_level == WHITE_LEVEL
    np.testing.assert_array_equal(cfa.data, raw.raw_image_visible)
    assert not np.shares_memory(cfa.data, raw.raw_image_visible)


def test_develop_writes_the_cfa_into_the_buffer_and_uses_linear_camera_wb() -> None:
    raw = FakeLibRaw()
    h = handle(raw, "clip")
    cfa = h.read_cfa()
    data = cfa.data.copy()
    data[:, 3] = 7
    out = h.develop(cfa.with_data(data))
    np.testing.assert_array_equal(raw.raw_image_visible, data)
    (kwargs,) = raw.calls
    assert kwargs["use_camera_wb"] is True
    assert kwargs["use_auto_wb"] is False
    assert kwargs["output_bps"] == 16
    assert kwargs["gamma"] == (1.0, 1.0)
    assert kwargs["no_auto_bright"] is True
    assert kwargs["output_color"] == rawpy.ColorSpace.sRGB
    assert kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.AHD
    assert kwargs["highlight_mode"] == rawpy.HighlightMode.Clip
    assert "user_flip" not in kwargs
    assert out.image.dtype == np.float32
    assert out.image.shape == (8, 12, 3)
    np.testing.assert_allclose(out.image[0, 3], 7 / OUTPUT_MAX, rtol=1e-6)
    assert out.params["highlight_mode"] == "clip"
    assert out.params["white_scale"] == 1.0


def test_develop_undoes_libraw_headroom_for_non_clip_modes() -> None:
    raw = FakeLibRaw()
    h = handle(raw, "blend")
    out = h.develop(h.read_cfa())
    expected = raw.raw_image_visible.astype(np.float32) * 2.0 / OUTPUT_MAX
    np.testing.assert_allclose(out.image[..., 0], expected, rtol=1e-6)
    assert out.params["white_scale"] == 2.0


def test_develop_rejects_a_mismatched_cfa() -> None:
    raw = FakeLibRaw()
    with pytest.raises(DecodeError, match="does not match"):
        handle(raw).develop(cfa_frame(4, 4))


def test_develop_wraps_libraw_errors() -> None:
    raw = FakeLibRaw()
    raw.fail = rawpy.LibRawDataError("boom")
    h = handle(raw)
    with pytest.raises(DecodeError, match="could not develop"):
        h.develop(h.read_cfa())


@pytest.mark.parametrize(
    ("flip", "expected"),
    [(0, (2, 1)), (3, (9, 6)), (5, (1, 9)), (6, (6, 2))],
)
def test_sensor_to_image(flip: int, expected: tuple[int, int]) -> None:
    assert sensor_to_image(2, 1, width=12, height=8, flip=flip) == expected


@pytest.mark.parametrize(("flip", "rot90_k"), [(0, 0), (3, 2), (5, 1), (6, -1)])
def test_sensor_to_image_matches_the_rotated_image(flip: int, rot90_k: int) -> None:
    sensor = np.arange(8 * 12).reshape(8, 12)
    image = np.rot90(sensor, rot90_k)
    for x, y in [(0, 0), (11, 0), (3, 5), (11, 7)]:
        ix, iy = sensor_to_image(x, y, width=12, height=8, flip=flip)
        assert image[iy, ix] == sensor[y, x]


def test_sensor_to_image_rejects_points_outside() -> None:
    with pytest.raises(ValueError, match="outside"):
        sensor_to_image(12, 0, width=12, height=8, flip=0)


def test_handle_maps_through_libraw_flip() -> None:
    assert handle(FakeLibRaw(flip=6)).sensor_to_image(2, 1) == (6, 2)


def test_normalized_wb() -> None:
    assert normalized_wb([2.0, 1.0, 1.5, 0.0]) == (2.0, 1.0, 1.5, 1.0)
    assert normalized_wb([2.0, 1.0, 1.5]) == (2.0, 1.0, 1.5, 1.0)
    assert normalized_wb([0.0, 0.0, 0.0, 0.0]) is None
    assert normalized_wb([1.0]) is None


def test_settings_reject_unknown_names() -> None:
    with pytest.raises(ValueError, match="demosaic"):
        DevelopSettings(demosaic="magic")
    with pytest.raises(ValueError, match="highlight_mode"):
        DevelopSettings(highlight_mode="magic")


def test_decoder_opens_and_closes(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0001.CR2"
    path.write_bytes(b"raw")
    raw = FakeLibRaw()
    opened: list[str] = []

    def opener(p: str) -> FakeLibRaw:
        opened.append(p)
        return raw

    with RawpyDecoder(opener=opener).open(path) as h:
        assert h.read_cfa().width == 12
        assert not raw.closed
    assert raw.closed
    assert opened == [str(path)]
    assert path.read_bytes() == b"raw"


def test_decoder_closes_on_error(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0001.CR2"
    path.write_bytes(b"raw")
    raw = FakeLibRaw()
    with pytest.raises(RuntimeError), RawpyDecoder(opener=lambda _: raw).open(path):
        raise RuntimeError
    assert raw.closed


def test_decoder_reports_missing_and_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(DecodeError, match="not a file"), RawpyDecoder().open(tmp_path / "missing.CR2"):
        pass
    junk = tmp_path / "junk.CR2"
    junk.write_bytes(b"not a raw file")
    with pytest.raises(DecodeError, match="could not open"), RawpyDecoder().open(junk):
        pass
    assert junk.read_bytes() == b"not a raw file"


def test_registered_and_built_from_config() -> None:
    assert "rawpy" in raw_decoder_registry
    cfg = parse_config({"decode": {"demosaic": "dcb", "highlight_mode": "clip"}})
    decoder = raw_decoder_registry.create("rawpy", BuildEnv(config=cfg, output_dir=Path("out")))
    assert isinstance(decoder, RawpyDecoder)


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLES, reason="no tests/fixtures/*.CR2")
def test_real_cr2_round_trip_leaves_the_file_untouched() -> None:
    path = SAMPLES[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with RawpyDecoder().open(path) as h:
        cfa = h.read_cfa()
        first = h.develop(cfa)
        data = cfa.data.copy()
        data[:, cfa.width // 2] = cfa.white_level
        h.develop(cfa.with_data(data))
        again = h.develop(cfa)
    assert first.image.dtype == np.float32
    assert first.image.shape[2] == 3
    assert np.isfinite(first.image).all()
    assert 0.0 < float(np.median(first.image)) < 1.0
    np.testing.assert_array_equal(first.image, again.image)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
