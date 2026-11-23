from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import BinaryIO

import pytest

from raw_pipeline.core.errors import ExportError, OutputExistsError
from raw_pipeline.exporters.output import write_output


def _write_ok(handle: BinaryIO) -> None:
    handle.write(b"data")


def _write_fail(handle: BinaryIO) -> None:
    handle.write(b"partial")
    raise ValueError("encoder failed")


def test_creates_parent_and_writes(tmp_path: Path) -> None:
    path = write_output(tmp_path / "sub" / "a.bin", _write_ok, overwrite=False)
    assert path.read_bytes() == b"data"


def test_refuses_existing_file_and_keeps_it(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"original")
    with pytest.raises(OutputExistsError, match="--overwrite"):
        write_output(target, _write_ok, overwrite=False)
    assert target.read_bytes() == b"original"
    write_output(target, _write_ok, overwrite=True)
    assert target.read_bytes() == b"data"


def test_partial_file_removed_on_failure(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="encoder failed"):
        write_output(tmp_path / "a.bin", _write_fail, overwrite=False)
    assert not (tmp_path / "a.bin").exists()


def test_unwritable_location_is_export_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    with pytest.raises(ExportError, match="cannot create"):
        write_output(blocker / "a.bin", _write_ok, overwrite=False)


def test_failed_overwrite_keeps_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"original")
    with pytest.raises(ExportError, match="encoder failed"):
        write_output(target, _write_fail, overwrite=True)
    assert target.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["a.bin"]


def test_overwrite_replaces_with_default_permissions(tmp_path: Path) -> None:
    fresh = write_output(tmp_path / "fresh.bin", _write_ok, overwrite=False)
    target = tmp_path / "a.bin"
    target.write_bytes(b"original")
    write_output(target, _write_ok, overwrite=True)
    assert target.read_bytes() == b"data"
    assert target.stat().st_mode == fresh.stat().st_mode
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.bin", "fresh.bin"]


def test_overwrite_never_follows_a_planted_temporary_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"keep")
    path = tmp_path / "out.jpg"
    monkeypatch.setattr(os, "getpid", lambda: 1)
    monkeypatch.setattr(threading, "get_ident", lambda: 2)
    (tmp_path / ".out.jpg.1-2.partial").symlink_to(victim)
    with pytest.raises(ExportError):
        write_output(path, lambda handle: handle.write(b"new"), overwrite=True)
    assert victim.read_bytes() == b"keep"
