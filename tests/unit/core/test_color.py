import numpy as np

from raw_pipeline.core.color import luminance, srgb_decode, srgb_encode, to_uint8, to_uint16


def test_luminance_of_grey_is_grey() -> None:
    img = np.full((2, 3, 3), 0.18, dtype=np.float32)
    lum = luminance(img)
    assert lum.shape == (2, 3)
    assert lum.dtype == np.float32
    np.testing.assert_allclose(lum, 0.18, rtol=1e-6)


def test_luminance_weights_green_most() -> None:
    img = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    lum = luminance(img)[0]
    assert lum[1] > lum[0] > lum[2]


def test_srgb_roundtrip_and_known_values() -> None:
    x = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    np.testing.assert_allclose(srgb_decode(srgb_encode(x)), x, atol=1e-5)
    assert abs(float(srgb_encode(np.float32([0.18]))[0]) - 0.4614) < 1e-3


def test_srgb_encode_clips_and_is_monotonic() -> None:
    x = np.float32([-1.0, 0.0, 0.5, 1.0, 4.0])
    y = srgb_encode(x)
    assert y[0] == 0.0
    assert abs(float(y[-1]) - 1.0) < 1e-6
    assert np.all(np.diff(y) >= 0)


def test_quantization() -> None:
    x = np.float32([0.0, 0.5, 1.0, 2.0])
    assert to_uint8(x).tolist() == [0, 128, 255, 255]
    assert to_uint16(x).tolist() == [0, 32768, 65535, 65535]
