from __future__ import annotations

from pathlib import Path

import pytest

from raw_pipeline.io.discovery import discover_raw_files


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def names(paths: list[Path], root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in paths]


def test_sorted_case_insensitive_non_recursive(tmp_path: Path) -> None:
    for name in ["img_0003.cr2", "IMG_0001.CR2", "IMG_0002.Cr2", "notes.txt", "IMG_0004.JPG", "._IMG_0005.CR2"]:
        touch(tmp_path / name)
    touch(tmp_path / "sub" / "IMG_0000.CR2")
    (tmp_path / "dir.CR2").mkdir()
    found = discover_raw_files(tmp_path)
    assert names(found, tmp_path) == ["IMG_0001.CR2", "IMG_0002.Cr2", "img_0003.cr2"]


def test_recursive_skips_excluded_hidden_and_symlinked_dirs(tmp_path: Path) -> None:
    touch(tmp_path / "b" / "IMG_2.CR2")
    touch(tmp_path / "a" / "IMG_1.CR2")
    touch(tmp_path / "out" / "IMG_9.CR2")
    touch(tmp_path / ".cache" / "IMG_8.CR2")
    elsewhere = touch(tmp_path.parent / f"{tmp_path.name}-elsewhere" / "IMG_7.CR2").parent
    (tmp_path / "link").symlink_to(elsewhere, target_is_directory=True)
    found = discover_raw_files(tmp_path, recursive=True, exclude_dirs=[tmp_path / "out"])
    assert names(found, tmp_path) == ["a/IMG_1.CR2", "b/IMG_2.CR2"]


def test_custom_extensions(tmp_path: Path) -> None:
    touch(tmp_path / "a.CR3")
    touch(tmp_path / "b.CR2")
    assert names(discover_raw_files(tmp_path, extensions={".CR3"}), tmp_path) == ["a.CR3"]


def test_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        discover_raw_files(tmp_path / "missing")


def test_empty_dir(tmp_path: Path) -> None:
    assert discover_raw_files(tmp_path) == []
