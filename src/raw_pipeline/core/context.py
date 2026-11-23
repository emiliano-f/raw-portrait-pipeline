"""The per-image context that flows through the pipeline, plus typed artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.core.values import ImageMetadata, Scalar

T = TypeVar("T")


@dataclass(frozen=True)
class ArtifactKey(Generic[T]):
    """Typed key for :attr:`ImageContext.artifacts`.

    Attributes:
        name: Unique string key stored in the mapping.
    """

    name: str


def empty_image() -> NDArray[np.float32]:
    """Return the placeholder ``(0, 0, 3)`` image held before decoding."""
    return np.zeros((0, 0, 3), dtype=np.float32)


@dataclass(frozen=True)
class ImageContext:
    """Everything known about one image at one point in the pipeline.

    Stages never mutate a context or its arrays; they return a new context
    built with the ``with_*`` helpers.

    Attributes:
        source_path: Path of the input RAW file (read-only).
        image: ``(H, W, 3)`` float32 scene-linear RGB in the working space. Values
            above 1.0 are allowed until export. ``(0, 0, 3)`` before decoding.
        metadata: EXIF plus every parameter recorded by the stages so far.
        artifacts: Side results (faces, masks, previews), read through
            :meth:`artifact` with an :class:`ArtifactKey`.
    """

    source_path: Path
    image: NDArray[np.float32] = field(default_factory=empty_image)
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    artifacts: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def initial(cls, source_path: Path) -> ImageContext:
        """Context handed to the first stage: only the source path is known."""
        return cls(source_path=source_path)

    @property
    def is_decoded(self) -> bool:
        """``True`` once a decode stage has produced pixels."""
        return self.image.size > 0

    def with_image(self, image: NDArray[np.float32]) -> ImageContext:
        """Return a copy holding ``image`` (must be float32 ``(H, W, 3)``)."""
        if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:  # noqa: PLR2004
            msg = f"image must be float32 (H, W, 3), got {image.dtype} {image.shape}"
            raise ValueError(msg)
        return ImageContext(self.source_path, image, self.metadata, self.artifacts)

    def with_metadata(self, metadata: ImageMetadata) -> ImageContext:
        """Return a copy with ``metadata`` replaced."""
        return ImageContext(self.source_path, self.image, metadata, self.artifacts)

    def with_record(self, stage: str, params: Mapping[str, Scalar]) -> ImageContext:
        """Return a copy with ``params`` recorded for ``stage`` in the metadata."""
        return self.with_metadata(self.metadata.with_record(stage, params))

    def with_artifact(self, key: ArtifactKey[T], value: T) -> ImageContext:
        """Return a copy with artifact ``key`` set to ``value``."""
        artifacts = dict(self.artifacts)
        artifacts[key.name] = value
        return ImageContext(self.source_path, self.image, self.metadata, MappingProxyType(artifacts))

    def artifact(self, key: ArtifactKey[T]) -> T | None:
        """Return the artifact stored under ``key``, or ``None``."""
        return cast("T | None", self.artifacts.get(key.name))
