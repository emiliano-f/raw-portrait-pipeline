"""Base-DNG sources of ``fix-raw``: DNGLab, Adobe DNG Converter and existing DNGs (ADR 0003).

Converters are external programs; nothing here downloads or bundles them.
They run with an argument list (never a shell), a timeout and their output
confined to the caller's temporary ``workdir``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from raw_pipeline.config import app_config
from raw_pipeline.config.models import ConverterConfig
from raw_pipeline.core.dng import BaseDng
from raw_pipeline.core.errors import ConfigError, ConverterError, ConverterNotFoundError, CoordinateMappingError
from raw_pipeline.core.registries import dng_source_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.io.dng.tiff import check_patchable, read_layout

CR2_SUFFIXES: Final = frozenset({".cr2"})
DNG_SUFFIXES: Final = frozenset({".dng"})
STDERR_TAIL_CHARS: Final = 600
"""Characters of converter output quoted in error messages."""

DNGLAB_INSTALL: Final = (
    "install DNGLab (https://github.com/dnglab/dnglab/releases: Linux .deb/binary, Windows and macOS builds) "
    "and put 'dnglab' on PATH or set raw_export.converter.executable to its full path"
)
ADOBE_INSTALL: Final = (
    "install Adobe DNG Converter (free, Windows and macOS: "
    "https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html; on Linux run it under Wine "
    "through a wrapper script) and set raw_export.converter.executable to its path"
)
DNGLAB_REQUIRED: Final = ("--compression", "uncompressed", "--embed-raw", "false")
DNGLAB_CONFLICTING: Final = frozenset({"-c", "--compression", "--embed-raw"})
ADOBE_REQUIRED: Final = ("-u",)
ADOBE_CONFLICTING: Final = frozenset({"-c", "-l", "-e", "-lossy", "-d", "-o"})

Runner = Callable[..., "subprocess.CompletedProcess[Any]"]
Which = Callable[[str], str | None]


def _check_args(args: Sequence[str], conflicting: frozenset[str], source: str) -> tuple[str, ...]:
    bad = sorted({a.split("=", 1)[0] for a in args} & conflicting)
    if bad:
        msg = (
            f"raw_export.converter.args: {', '.join(bad)} conflict with the flags {source} "
            "requires for an uncompressed CFA DNG"
        )
        raise ConfigError(msg)
    return tuple(args)


@dataclass(frozen=True)
class _Converter:
    """Runs one converter command and checks that it produced ``output``."""

    executable: str
    timeout_s: float
    install_hint: str
    runner: Runner
    which: Which

    def resolve(self) -> str:
        """Absolute path of the executable, or :class:`ConverterNotFoundError` with install instructions."""
        candidate = Path(self.executable).expanduser()
        if candidate.is_absolute() or len(candidate.parts) > 1:
            if candidate.is_file():
                return str(candidate)
        elif (found := self.which(self.executable)) is not None:
            return found
        msg = f"converter executable {self.executable!r} not found: {self.install_hint}"
        raise ConverterNotFoundError(msg)

    def run(self, command: list[str], output: Path) -> None:
        try:
            done = self.runner(
                command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"converter timed out after {self.timeout_s:g} s (raw_export.converter.timeout_s)"
            raise ConverterError(msg) from exc
        except OSError as exc:
            msg = f"cannot run converter {command[0]!r}: {exc}"
            raise ConverterError(msg) from exc
        if done.returncode != 0:
            detail = ((done.stderr or "") + (done.stdout or "")).strip()[-STDERR_TAIL_CHARS:]
            msg = f"converter exited with status {done.returncode}: {detail or '(no output)'}"
            raise ConverterError(msg)
        if not output.is_file():
            msg = f"converter finished but did not write {output.name}"
            raise ConverterError(msg)


@dng_source_registry.register("dnglab")
class DnglabSource:
    """Converts CR2 files with DNGLab (``dnglab convert``) into uncompressed CFA DNGs.

    Always adds ``--compression uncompressed --embed-raw false``. DNGLab
    keeps the photosites exactly as stored in the CR2 (verified on EOS 800D).

    Args:
        config: Converter section (``raw_export.converter``).
        runner: Runs the command (``subprocess.run`` by default; injected by tests).
        which: Looks up a command on PATH (``shutil.which``).
    """

    name: ClassVar[str] = "dnglab"
    input_suffixes: ClassVar[frozenset[str]] = CR2_SUFFIXES
    default_executable: ClassVar[str] = "dnglab"

    def __init__(
        self, config: ConverterConfig | None = None, runner: Runner = subprocess.run, which: Which = shutil.which
    ) -> None:
        cfg = config or ConverterConfig()
        self._args = _check_args(cfg.args, DNGLAB_CONFLICTING, self.name)
        self._converter = _Converter(
            cfg.executable or self.default_executable, cfg.timeout_s, DNGLAB_INSTALL, runner, which
        )

    @classmethod
    def from_env(cls, env: BuildEnv) -> DnglabSource:
        """Build from ``raw_export.converter``."""
        return cls(app_config(env).raw_export.converter)

    def command(self, executable: str, source: Path, output: Path) -> list[str]:
        """The argument list run for ``source``."""
        return [executable, "convert", *self._args, *DNGLAB_REQUIRED, str(source), str(output)]

    def prepare(self, source: Path, workdir: Path) -> BaseDng:
        """Convert ``source`` into ``workdir``; see :class:`~raw_pipeline.core.protocols.DngSource`."""
        output = workdir / f"{source.stem}.dng"
        self._converter.run(self.command(self._converter.resolve(), source, output), output)
        return BaseDng(path=output, reference=source, source=self.name)


@dng_source_registry.register("adobe_converter")
class AdobeConverterSource:
    """Converts CR2 files with Adobe DNG Converter's command line.

    Always adds ``-u`` (uncompressed); linear (``-l``), compressed (``-c``,
    ``-lossy``) and embedded-original (``-e``) output are refused. Runs
    natively on Windows and macOS; on Linux point ``executable`` at a Wine
    wrapper script (see the fix-raw guide).

    Args:
        config: Converter section (``raw_export.converter``).
        runner: Runs the command (``subprocess.run`` by default; injected by tests).
        which: Looks up a command on PATH (``shutil.which``).
    """

    name: ClassVar[str] = "adobe_converter"
    input_suffixes: ClassVar[frozenset[str]] = CR2_SUFFIXES
    default_executable: ClassVar[str] = "Adobe DNG Converter"

    def __init__(
        self, config: ConverterConfig | None = None, runner: Runner = subprocess.run, which: Which = shutil.which
    ) -> None:
        cfg = config or ConverterConfig()
        self._args = _check_args(cfg.args, ADOBE_CONFLICTING, self.name)
        self._converter = _Converter(
            cfg.executable or self.default_executable, cfg.timeout_s, ADOBE_INSTALL, runner, which
        )

    @classmethod
    def from_env(cls, env: BuildEnv) -> AdobeConverterSource:
        """Build from ``raw_export.converter``."""
        return cls(app_config(env).raw_export.converter)

    def command(self, executable: str, source: Path, output: Path) -> list[str]:
        """The argument list run for ``source``."""
        return [executable, *self._args, *ADOBE_REQUIRED, "-d", str(output.parent), "-o", output.name, str(source)]

    def prepare(self, source: Path, workdir: Path) -> BaseDng:
        """Convert ``source`` into ``workdir``; see :class:`~raw_pipeline.core.protocols.DngSource`."""
        output = workdir / f"{source.stem}.dng"
        self._converter.run(self.command(self._converter.resolve(), source, output), output)
        return BaseDng(path=output, reference=source, source=self.name)


@dng_source_registry.register("existing_dng")
class ExistingDngSource:
    """Uses an input DNG (e.g. made earlier with Adobe DNG Converter) as the base DNG.

    The DNG must be uncompressed, 16-bit and CFA (not linear), and its
    original CR2 must lie next to it with the same stem: the defect map is in
    the CR2's LibRaw coordinates and the CR2 confirms the offset.
    """

    name: ClassVar[str] = "existing_dng"
    input_suffixes: ClassVar[frozenset[str]] = DNG_SUFFIXES

    @classmethod
    def from_env(cls, env: BuildEnv) -> ExistingDngSource:  # noqa: ARG003 - no configuration
        """Build (no options)."""
        return cls()

    def prepare(self, source: Path, workdir: Path) -> BaseDng:  # noqa: ARG002 - nothing is converted
        """Validate ``source`` and find its reference CR2."""
        check_patchable(read_layout(source), source.name)
        return BaseDng(path=source, reference=reference_raw(source), source=self.name)


def reference_raw(dng: Path) -> Path:
    """The CR2 beside ``dng`` with the same stem (any suffix case)."""
    for candidate in sorted(dng.parent.iterdir()):
        if candidate.stem == dng.stem and candidate.suffix.lower() in CR2_SUFFIXES and candidate.is_file():
            return candidate
    msg = f"{dng.name}: original {dng.stem}.CR2 not found next to it (needed to map the defect coordinates)"
    raise CoordinateMappingError(msg)
