"""Batch pipeline that turns Canon CR2 portraits into finished JPG/TIFF images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raw-pipeline")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
