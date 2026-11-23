"""Discovery of input RAW files."""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Final

RAW_EXTENSIONS: Final = frozenset({".cr2"})
"""Lower-case file suffixes treated as RAW inputs."""


def _is_candidate(path: Path, extensions: Collection[str]) -> bool:
    return not path.name.startswith(".") and path.suffix.lower() in extensions and path.is_file()


def _sort_key(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root).as_posix()
    return rel.casefold(), rel


def _walk(root: Path, excluded: set[Path]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and (current / d).resolve() not in excluded)
        yield from (current / name for name in filenames)


def discover_raw_files(
    input_dir: Path,
    *,
    recursive: bool = False,
    exclude_dirs: Iterable[Path] = (),
    extensions: Collection[str] = RAW_EXTENSIONS,
) -> list[Path]:
    """List RAW files in ``input_dir``, sorted case-insensitively by relative path.

    Suffixes match case-insensitively (``.CR2`` and ``.cr2``). Hidden files and
    directories (leading ``.``, e.g. macOS ``._IMG_0001.CR2``) are skipped.
    Symlinked directories are not followed.

    Args:
        input_dir: Directory to scan. Never modified.
        recursive: Also scan subdirectories.
        exclude_dirs: Directories never descended into (e.g. the output
            directory when it lives inside the input directory).
        extensions: Lower-case suffixes to accept, including the dot.

    Returns:
        Paths of the RAW files found (possibly empty).

    Raises:
        NotADirectoryError: If ``input_dir`` is not an existing directory.
    """
    if not input_dir.is_dir():
        msg = f"input directory {input_dir} does not exist or is not a directory"
        raise NotADirectoryError(msg)
    suffixes = {e.lower() for e in extensions}
    excluded = {Path(d).resolve() for d in exclude_dirs}
    candidates = _walk(input_dir, excluded) if recursive else input_dir.iterdir()
    found = [p for p in candidates if _is_candidate(p, suffixes)]
    return sorted(found, key=lambda p: _sort_key(p, input_dir))
