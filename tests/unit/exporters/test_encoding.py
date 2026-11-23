from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import ImageCms

from raw_pipeline.exporters.encoding import (
    ICC_DATE_LENGTH,
    ICC_DATE_OFFSET,
    encode_srgb_uint8,
    encode_srgb_uint16,
    output_size,
    resize_long_edge,
    srgb_icc_profile,
    srgb_thumbnail,
)
from tests.support.synthetic import flat_image, gradient_image


@pytest.mark.parametrize(
    ("size", "long_edge", "expected"),
    [((96, 64), None, (96, 64)), ((96, 64), 48, (48, 32)), ((64, 96), 48, (32, 48)), ((96, 64), 500, (96, 64))],
)
def test_output_size_fits_long_edge_and_never_upscales(size, long_edge, expected) -> None:
    assert output_size(*size, long_edge) == expected


def test_resize_is_area_average_in_linear_light() -> None:
    img = flat_image(0.3, 64, 96)
    img[:, ::2] = 1.7  # values above 1.0 survive resizing
    out = resize_long_edge(img, 48)
    assert out.shape == (32, 48, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_resize_without_limit_returns_input() -> None:
    img = gradient_image()
    assert resize_long_edge(img, None) is img
    assert resize_long_edge(img, 10_000) is img


def test_encoding_clips_and_applies_transfer_function() -> None:
    img = np.asarray([[[-0.5, 0.18, 2.0]]], dtype=np.float32)
    assert encode_srgb_uint8(img).tolist() == [[[0, 118, 255]]]
    assert encode_srgb_uint16(img).tolist()[0][0][0::2] == [0, 65535]


def test_thumbnail_is_uint8_and_fitted() -> None:
    thumb = srgb_thumbnail(gradient_image(64, 96), 24)
    assert thumb.dtype == np.uint8
    assert thumb.shape == (16, 24, 3)


def test_icc_profile_is_srgb_and_has_fixed_date() -> None:
    profile = srgb_icc_profile()
    srgb_icc_profile.cache_clear()
    assert srgb_icc_profile() == profile
    assert profile[ICC_DATE_OFFSET : ICC_DATE_OFFSET + ICC_DATE_LENGTH] == bytes.fromhex("07ea00010001000000000000")
    parsed = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    assert "sRGB" in ImageCms.getProfileDescription(parsed)
