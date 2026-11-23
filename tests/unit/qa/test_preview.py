from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from raw_pipeline.qa.preview import MAX_PREVIEW_LONG_EDGE_PX, encode_preview, inspect_preview
from tests.support.synthetic import flat_image, gradient_image


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


@pytest.mark.parametrize(
    ("height", "width", "requested", "expected"),
    [
        (1500, 2400, 1024, (1024, 640)),
        (2400, 1500, 1024, (640, 1024)),
        (1500, 2400, 4096, (1024, 640)),
        (60, 90, 1024, (90, 60)),
        (1500, 2400, 300, (300, 188)),
    ],
)
def test_size_capped_and_never_upscaled(height: int, width: int, requested: int, expected: tuple[int, int]) -> None:
    data = encode_preview(flat_image(0.18, height, width), requested, 85)
    with _open(data) as img:
        assert img.size == expected
        assert max(img.size) <= MAX_PREVIEW_LONG_EDGE_PX


def test_jpeg_without_metadata() -> None:
    data = encode_preview(gradient_image(), 1024, 85)
    with _open(data) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert "exif" not in img.info
        assert "icc_profile" not in img.info
        assert "comment" not in img.info
    info = inspect_preview(data)
    assert not info.has_exif


def test_deterministic() -> None:
    img = gradient_image(300, 400)
    assert encode_preview(img, 128, 85) == encode_preview(img, 128, 85)


def test_srgb_encoded_and_clipped() -> None:
    img = np.concatenate([flat_image(0.18, 32, 32), flat_image(4.0, 32, 32)], axis=1)
    with _open(encode_preview(img, 1024, 100)) as out:
        px = np.asarray(out)
    # 0.18 linear is ~118 in sRGB; values above 1.0 clip to white.
    assert abs(int(px[16, 8, 1]) - 118) <= 2
    assert px[16, 56].min() >= 253


def test_input_not_mutated() -> None:
    img = gradient_image()
    before = img.copy()
    encode_preview(img, 32, 85)
    np.testing.assert_array_equal(img, before)


def test_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_preview(np.zeros((0, 0, 3), np.float32), 1024, 85)


def test_inspect_rejects_non_jpeg() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    with pytest.raises(ValueError, match="must be a JPEG"):
        inspect_preview(buf.getvalue())
    with pytest.raises(ValueError, match="must be a JPEG"):
        inspect_preview(b"II*\x00 definitely not a jpeg")
    with pytest.raises(ValueError, match="not a readable JPEG"):
        inspect_preview(b"\xff\xd8\xff truncated")


def test_inspect_reports_exif() -> None:
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG", exif=exif)
    assert inspect_preview(buf.getvalue()).has_exif
