"""``RawImageDigest`` / ``NewRawImageDigest`` of 16-bit raw data (DNG specification, Adobe DNG SDK).

Both are MD5 digests of the stored samples in little-endian byte order:

- ``RawImageDigest`` (legacy): the whole image in row-scan order.
- ``NewRawImageDigest``: the MD5 of the MD5s of 256 x 256 tiles, row-major
  (edge tiles are smaller, never padded).

``fix-raw`` only rewrites a digest after checking that these functions
reproduce the value stored in the base DNG; otherwise the tag is removed
(ADR 0003).
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
from numpy.typing import NDArray

DIGEST_TILE_PX: Final = 256
"""Tile size of ``NewRawImageDigest`` (``dng_find_new_raw_image_digest_task::kTileSize``)."""
DIGEST_BYTES: Final = 16


def _le(data: NDArray[np.uint16]) -> NDArray[np.uint16]:
    return np.ascontiguousarray(data, dtype=np.dtype("<u2"))


def legacy_raw_digest(data: NDArray[np.uint16]) -> bytes:
    """``RawImageDigest`` of a single-plane 16-bit image."""
    return hashlib.md5(_le(data).tobytes(), usedforsecurity=False).digest()


def new_raw_digest(data: NDArray[np.uint16]) -> bytes:
    """``NewRawImageDigest`` of a single-plane 16-bit image without transparency mask."""
    height, width = data.shape
    tile_h, tile_w = min(DIGEST_TILE_PX, height), min(DIGEST_TILE_PX, width)
    combined = hashlib.md5(usedforsecurity=False)
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            tile = _le(data[y : y + tile_h, x : x + tile_w])
            combined.update(hashlib.md5(tile.tobytes(), usedforsecurity=False).digest())
    return combined.digest()
