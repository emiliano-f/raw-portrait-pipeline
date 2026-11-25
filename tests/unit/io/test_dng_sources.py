from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from raw_pipeline.config.models import ConverterConfig
from raw_pipeline.core.errors import (
    ConfigError,
    ConverterError,
    ConverterNotFoundError,
    CoordinateMappingError,
    DngFormatError,
)
from raw_pipeline.core.protocols import DngSource
from raw_pipeline.core.registries import dng_source_registry
from raw_pipeline.io.dng.sources import AdobeConverterSource, DnglabSource, ExistingDngSource
from tests.contract.dng_source_contract import DngSourceContract
from tests.support.dng_builder import DngSpec, write_dng


class FakeConverter:
    """Records the command and writes a synthetic DNG where the converter would."""

    def __init__(
        self, *, returncode: int = 0, write: bool = True, timeout: bool = False, spec: DngSpec | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: dict[str, Any] = {}
        self.returncode, self.write, self.timeout, self.spec = returncode, write, timeout, spec

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        self.kwargs = kwargs
        if self.timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if self.write and self.returncode == 0:
            output = (
                Path(command[command.index("-d") + 1]) / command[command.index("-o") + 1]
                if "-d" in command
                else Path(command[-1])
            )
            write_dng(output, self.spec)
        return subprocess.CompletedProcess(command, self.returncode, "", "bad raw" if self.returncode else "")


def _found(name: str) -> str | None:
    return f"/usr/bin/{name}"


def _cr2(folder: Path) -> Path:
    path = folder / "IMG_0001.CR2"
    path.write_bytes(b"fake cr2")
    return path


class TestDnglabContract(DngSourceContract):
    def make_source(self) -> DngSource:
        return DnglabSource(runner=FakeConverter(), which=_found)

    def make_input(self, folder: Path) -> Path:
        return _cr2(folder)


class TestAdobeContract(DngSourceContract):
    def make_source(self) -> DngSource:
        return AdobeConverterSource(runner=FakeConverter(), which=_found)

    def make_input(self, folder: Path) -> Path:
        return _cr2(folder)


class TestExistingDngContract(DngSourceContract):
    def make_source(self) -> DngSource:
        return ExistingDngSource()

    def make_input(self, folder: Path) -> Path:
        _cr2(folder)
        return write_dng(folder / "IMG_0001.dng")


def test_registered() -> None:
    assert {"dnglab", "adobe_converter", "existing_dng"} <= set(dng_source_registry.names())


def test_dnglab_required_flags(tmp_path: Path) -> None:
    fake = FakeConverter()
    cfg = ConverterConfig(args=("--dng-thumbnail", "true"), timeout_s=7)
    DnglabSource(cfg, runner=fake, which=_found).prepare(_cr2(tmp_path), tmp_path)
    (command,) = fake.calls
    assert command[:2] == ["/usr/bin/dnglab", "convert"]
    assert command[2:4] == ["--dng-thumbnail", "true"]
    assert command[4:8] == ["--compression", "uncompressed", "--embed-raw", "false"]
    assert command[-2:] == [str(tmp_path / "IMG_0001.CR2"), str(tmp_path / "IMG_0001.dng")]
    assert fake.kwargs["timeout"] == 7
    assert fake.kwargs["stdin"] is subprocess.DEVNULL


def test_adobe_required_flags(tmp_path: Path) -> None:
    fake = FakeConverter()
    exe = tmp_path / "dngconv.sh"
    exe.write_text("#!/bin/sh\n")
    work = tmp_path / "work"
    work.mkdir()
    AdobeConverterSource(ConverterConfig(executable=str(exe), args=("-p1",)), runner=fake).prepare(_cr2(tmp_path), work)
    (command,) = fake.calls
    assert command == [str(exe), "-p1", "-u", "-d", str(work), "-o", "IMG_0001.dng", str(tmp_path / "IMG_0001.CR2")]


@pytest.mark.parametrize(
    ("cls", "arg"),
    [
        (DnglabSource, "--compression=lossless"),
        (DnglabSource, "-c"),
        (AdobeConverterSource, "-l"),
        (AdobeConverterSource, "-e"),
    ],
)
def test_conflicting_args_rejected(cls: type[DnglabSource] | type[AdobeConverterSource], arg: str) -> None:
    with pytest.raises(ConfigError, match="conflict"):
        cls(ConverterConfig(args=(arg,)))


@pytest.mark.parametrize(("cls", "hint"), [(DnglabSource, "DNGLab"), (AdobeConverterSource, "Adobe DNG Converter")])
def test_missing_executable_explains_install(
    tmp_path: Path, cls: type[DnglabSource] | type[AdobeConverterSource], hint: str
) -> None:
    with pytest.raises(ConverterNotFoundError, match=hint):
        cls(runner=FakeConverter(), which=lambda _: None).prepare(_cr2(tmp_path), tmp_path)
    with pytest.raises(ConverterNotFoundError, match="install"):
        cls(ConverterConfig(executable=str(tmp_path / "missing")), runner=FakeConverter()).prepare(
            _cr2(tmp_path), tmp_path
        )


def test_timeout(tmp_path: Path) -> None:
    with pytest.raises(ConverterError, match="timed out after 120 s"):
        DnglabSource(runner=FakeConverter(timeout=True), which=_found).prepare(_cr2(tmp_path), tmp_path)


def test_nonzero_exit_quotes_output(tmp_path: Path) -> None:
    with pytest.raises(ConverterError, match="status 3: bad raw"):
        DnglabSource(runner=FakeConverter(returncode=3), which=_found).prepare(_cr2(tmp_path), tmp_path)


def test_missing_output(tmp_path: Path) -> None:
    with pytest.raises(ConverterError, match="did not write"):
        DnglabSource(runner=FakeConverter(write=False), which=_found).prepare(_cr2(tmp_path), tmp_path)


def test_os_error(tmp_path: Path) -> None:
    def broken(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise PermissionError("denied")

    with pytest.raises(ConverterError, match="denied"):
        DnglabSource(runner=broken, which=_found).prepare(_cr2(tmp_path), tmp_path)


def test_existing_dng_needs_reference(tmp_path: Path) -> None:
    dng = write_dng(tmp_path / "IMG_0002.dng")
    with pytest.raises(CoordinateMappingError, match="IMG_0002.CR2 not found"):
        ExistingDngSource().prepare(dng, tmp_path)


def test_existing_dng_finds_lowercase_reference(tmp_path: Path) -> None:
    (tmp_path / "IMG_0003.cr2").write_bytes(b"x")
    base = ExistingDngSource().prepare(write_dng(tmp_path / "IMG_0003.dng"), tmp_path)
    assert base.reference == tmp_path / "IMG_0003.cr2"
    assert base.path == tmp_path / "IMG_0003.dng"


@pytest.mark.parametrize("spec", [DngSpec(compression=7), DngSpec(photometric=34892)], ids=["compressed", "linear"])
def test_existing_dng_rejects_unpatchable(tmp_path: Path, spec: DngSpec) -> None:
    _cr2(tmp_path)
    with pytest.raises(DngFormatError):
        ExistingDngSource().prepare(write_dng(tmp_path / "IMG_0001.dng", spec), tmp_path)
