from __future__ import annotations

import dataclasses
import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from raw_pipeline import cli
from raw_pipeline.core.errors import DecodeError
from raw_pipeline.core.raw import CFAFrame, ColumnDefect
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects.directional import DirectionalCorrector
from raw_pipeline.defects.null import NullDefectCorrector
from raw_pipeline.io.dng.sources import ExistingDngSource
from raw_pipeline.io.dng.tiff import read_raw
from raw_pipeline.runtime.components import ComponentRegistries
from raw_pipeline.runtime.services import RuntimeServices
from tests.support.dng_builder import DngSpec, raw_frame, write_dng
from tests.unit.runtime.fakes import FakeServices, fake_config, make_registries

HOT_X = 14  # stored column; visible column 10 at offset (2, 4)


class DngReferenceDecoder:
    """Opens ``X.CR2`` as the active area of ``X.dng`` (the 'CR2' bytes are ignored)."""

    name: ClassVar[str] = "dng_ref"
    opened: ClassVar[list[Path]] = []

    @classmethod
    def from_env(cls, env: BuildEnv) -> DngReferenceDecoder:
        del env
        return cls()

    @contextmanager
    def open(self, path: Path) -> Iterator[DngReferenceDecoder]:
        DngReferenceDecoder.opened.append(path)
        if path.read_bytes().startswith(b"BAD"):
            raise DecodeError(f"cannot open {path.name}")
        self._path = path
        yield self

    def read_cfa(self) -> CFAFrame:
        if self._path.read_bytes().startswith(b"NOMATCH"):
            return CFAFrame(raw_frame(40, 50, seed=9), ((0, 1), (3, 2)), "RGBG", (0,) * 4, 16383)
        dng = self._path.with_suffix(".dng") if self._path.suffix.lower() == ".cr2" else self._path
        data = read_raw(dng).data[2:, 4:].copy()
        return CFAFrame(data, ((0, 1), (3, 2)), "RGBG", (0,) * 4, 16383)


def _hot_spec() -> DngSpec:
    raw = raw_frame()
    raw[:, HOT_X] = 11900
    return DngSpec(raw=raw)


def _inputs(folder: Path, names: tuple[str, ...] = ("IMG_0001", "IMG_0002")) -> Path:
    folder.mkdir()
    for name in names:
        write_dng(folder / f"{name}.dng", _hot_spec())
        (folder / f"{name}.CR2").write_bytes(b"cr2")
    return folder


def _config(
    tmp_path: Path, method: str = "directional", defects: list[dict[str, object]] | None = None, **raw_export: object
) -> Path:
    import yaml  # noqa: PLC0415

    data = fake_config().model_dump(mode="json")
    data["decode"]["decoder"] = "dng_ref"
    data["defects"]["method"] = method
    data["defects"]["map"] = [{"type": "column", "index": 10}] if defects is None else defects
    data["raw_export"] = {"source": "existing_dng", "preview": "keep", **raw_export}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _registries() -> ComponentRegistries:
    regs = make_registries(fake_config())
    regs.raw_decoders.register("dng_ref")(DngReferenceDecoder)
    regs.defect_correctors.register("directional")(DirectionalCorrector)
    regs.defect_correctors.register("none_real")(NullDefectCorrector)
    regs.dng_sources.register("existing_dng")(ExistingDngSource)
    return regs


def _services(remaining: list[object] | None = None) -> RuntimeServices:
    return dataclasses.replace(FakeServices().services(), remaining_defects=lambda *_: list(remaining or []))


def _run(*argv: object, services: RuntimeServices | None = None) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(a) for a in argv], registries=_registries(), services=services or _services(), stdout=out, stderr=err
    )
    return code, out.getvalue() + err.getvalue()


def _hashes(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def _report(out: Path) -> list[dict[str, str]]:
    import csv  # noqa: PLC0415

    with (out / "fix_raw_report.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_fix_raw_corrects_only_the_defect(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in"), tmp_path / "out"
    before = _hashes(src)
    code, text = _run("fix-raw", src, out, "--config", _config(tmp_path))
    assert code == 0, text
    assert _hashes(src) == before
    assert sorted(p.name for p in out.iterdir()) == ["IMG_0001.dng", "IMG_0002.dng", "fix_raw_report.csv"]
    base, fixed = read_raw(src / "IMG_0001.dng").data, read_raw(out / "IMG_0001.dng").data
    mask = np.zeros(base.shape, bool)
    mask[2:, HOT_X] = True
    np.testing.assert_array_equal(fixed[~mask], base[~mask])
    assert fixed[2:, HOT_X].max() < 5000
    rows = _report(out)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    row = rows[0]
    assert (row["offset_y"], row["offset_x"], row["mapping_method"]) == ("2", "4", "active_area")
    assert row["source"] == "existing_dng"
    assert row["defect_method"] == "directional"
    assert row["corrected_pixels"] == "46"
    assert (row["digest"], row["preview"], row["verification"], row["passthrough"]) == (
        "absent",
        "keep",
        "passed",
        "False",
    )
    assert any(p.parent == out and p.name.startswith(".IMG_0001.dng.") for p in DngReferenceDecoder.opened)


@pytest.mark.parametrize(
    ("method", "defects"), [("none_real", None), ("directional", [])], ids=["method-none", "empty-map"]
)
def test_passthrough_is_byte_identical(tmp_path: Path, method: str, defects: list[dict[str, object]] | None) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path, method=method, defects=defects))
    assert code == 0
    assert (out / "IMG_0001.dng").read_bytes() == (src / "IMG_0001.dng").read_bytes()
    (row,) = _report(out)
    assert (row["passthrough"], row["corrected_pixels"], row["digest"]) == ("True", "0", "absent")


def test_failure_is_per_image(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in"), tmp_path / "out"
    (src / "IMG_0001.CR2").write_bytes(b"BAD")
    code, text = _run("fix-raw", src, out, "--config", _config(tmp_path))
    assert code == 1
    assert "FAILED" in text
    rows = _report(out)
    assert [r["status"] for r in rows] == ["failed", "ok"]
    assert "cannot open" in rows[0]["error"]
    assert not (out / "IMG_0001.dng").exists()


def test_existing_output_needs_overwrite(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    out.mkdir()
    (out / "IMG_0001.dng").write_bytes(b"old")
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path, report_filename="r1.csv"))
    assert code == 1
    assert (out / "IMG_0001.dng").read_bytes() == b"old"
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path, report_filename="r1.csv"))
    assert code == 2  # report exists now
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path, report_filename="r1.csv"), "--overwrite")
    assert code == 0
    assert read_raw(out / "IMG_0001.dng").data.shape == (48, 64)


def test_remaining_defect_fails_verification_and_removes_output(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path), services=_services([ColumnDefect(index=10)]))
    assert code == 1
    assert not (out / "IMG_0001.dng").exists()
    (row,) = _report(out)
    assert "still detected" in row["error"]


def test_verify_false_skips_checks(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    code, _ = _run(
        "fix-raw", src, out, "--config", _config(tmp_path, verify=False), services=_services([ColumnDefect(index=10)])
    )
    assert code == 0
    assert _report(out)[0]["verification"] == "skipped"


def test_no_inputs(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    code, text = _run("fix-raw", tmp_path / "in", tmp_path / "out", "--config", _config(tmp_path))
    assert code == 2
    assert ".dng" in text


def test_unknown_source_is_config_error(tmp_path: Path) -> None:
    code, text = _run(
        "fix-raw", _inputs(tmp_path / "in"), tmp_path / "out", "--config", _config(tmp_path, source="nope")
    )
    assert code == 2
    assert "nope" in text


def test_mapping_failure_writes_nothing(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    (src / "IMG_0001.CR2").write_bytes(b"NOMATCH")
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path))
    assert code == 1
    assert not (out / "IMG_0001.dng").exists()
    (row,) = _report(out)
    assert row["status"] == "failed"
    assert row["mapping_method"] == ""


def test_report_name_clashing_with_output_is_rejected(tmp_path: Path) -> None:
    src = _inputs(tmp_path / "in", ("IMG_0001",))
    code, text = _run("fix-raw", src, tmp_path / "out", "--config", _config(tmp_path, report_filename="img_0001.DNG"))
    assert code == 2
    assert "clashes" in text
    assert not (tmp_path / "out").exists()


def test_failed_verification_keeps_previous_output(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    out.mkdir()
    (out / "IMG_0001.dng").write_bytes(b"previous")
    services = _services([ColumnDefect(index=10)])
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path), "--overwrite", services=services)
    assert code == 1
    assert (out / "IMG_0001.dng").read_bytes() == b"previous"
    assert sorted(p.name for p in out.iterdir()) == ["IMG_0001.dng", "fix_raw_report.csv"]


def test_symlinked_input_is_never_replaced(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    out.mkdir()
    (out / "IMG_0001.dng").symlink_to(src / "IMG_0001.dng")
    before = _hashes(src)
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path), "--overwrite")
    assert code == 1
    assert _hashes(src) == before
    assert (out / "IMG_0001.dng").is_symlink()
    assert "inputs" in _report(out)[0]["error"]


def test_method_none_ignores_out_of_frame_map(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    config = _config(tmp_path, method="none_real", defects=[{"type": "column", "index": 5000}])
    code, text = _run("fix-raw", src, out, "--config", config)
    assert code == 0, text
    assert (out / "IMG_0001.dng").read_bytes() == (src / "IMG_0001.dng").read_bytes()


def test_passthrough_with_regenerate_is_byte_identical(tmp_path: Path) -> None:
    src, out = _inputs(tmp_path / "in", ("IMG_0001",)), tmp_path / "out"
    code, _ = _run("fix-raw", src, out, "--config", _config(tmp_path, defects=[], preview="regenerate"))
    assert code == 0
    assert (out / "IMG_0001.dng").read_bytes() == (src / "IMG_0001.dng").read_bytes()
    assert _report(out)[0]["preview"] == "keep"
