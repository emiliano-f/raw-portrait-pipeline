"""Contract every :class:`~raw_pipeline.core.protocols.DngSource` must satisfy."""

from __future__ import annotations

import hashlib
from pathlib import Path

from raw_pipeline.core.protocols import DngSource
from raw_pipeline.io.dng.tiff import check_patchable, read_layout


class DngSourceContract:
    """Reusable suite. Override ``make_source`` and ``make_input`` (returns a valid input file)."""

    def make_source(self) -> DngSource:
        raise NotImplementedError

    def make_input(self, folder: Path) -> Path:
        raise NotImplementedError

    def test_name_and_suffixes(self) -> None:
        source = self.make_source()
        assert source.name
        assert source.input_suffixes
        assert all(s.startswith(".") and s == s.lower() for s in source.input_suffixes)

    def test_accepts_its_input_suffix(self, tmp_path: Path) -> None:
        assert self.make_input(tmp_path).suffix.lower() in self.make_source().input_suffixes

    def test_produces_patchable_dng_without_touching_input(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "work").mkdir()
        source_file = self.make_input(tmp_path / "in")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "in").iterdir()}
        base = self.make_source().prepare(source_file, tmp_path / "work")
        check_patchable(read_layout(base.path), base.path.name)
        assert base.source == self.make_source().name
        assert base.reference.is_file()
        assert base.path == source_file or base.path.parent == tmp_path / "work"
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "in").iterdir()}
        assert after == before
