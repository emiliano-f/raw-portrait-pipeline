"""Output path planning: where each input's outputs go, without silent overwrites."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from raw_pipeline.core.errors import ConfigError, OutputExistsError


def ensure_writable(path: Path, *, overwrite: bool) -> Path:
    """Return ``path`` if it may be written.

    Raises:
        OutputExistsError: If ``path`` exists and ``overwrite`` is false.
    """
    if path.exists() and not overwrite:
        msg = f"{path} already exists; pass --overwrite to replace it"
        raise OutputExistsError(msg)
    return path


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass(frozen=True)
class OutputPlanner:
    """Maps input RAW files to output destinations (paths without suffix).

    Inputs in subdirectories of ``input_dir`` keep their relative directory
    under ``output_dir``. Exporters append their own suffix (``.jpg``, ``.tif``).

    Attributes:
        input_dir: The read-only input directory.
        output_dir: Directory receiving the outputs.
        suffix: Appended to the input stem (``export.filename_suffix``).
        overwrite: Whether existing outputs may be replaced.
        allow_inside_input: Allow ``output_dir`` to be (inside) ``input_dir``.
    """

    input_dir: Path
    output_dir: Path
    suffix: str = ""
    overwrite: bool = False
    allow_inside_input: bool = False

    def validate(self) -> None:
        """Check the planner before the batch starts.

        Raises:
            ConfigError: If ``suffix`` contains a path separator, or
                ``output_dir`` is inside ``input_dir`` without ``allow_inside_input``.
        """
        if "/" in self.suffix or "\\" in self.suffix:
            msg = f"export.filename_suffix must not contain path separators: {self.suffix!r}"
            raise ConfigError(msg)
        if not self.allow_inside_input and _is_within(self.output_dir.resolve(), self.input_dir.resolve()):
            msg = f"output directory {self.output_dir} is inside the input directory {self.input_dir}"
            raise ConfigError(msg)

    def destination(self, source: Path) -> Path:
        """Return the output path (without suffix) for ``source``."""
        try:
            parent = source.parent.relative_to(self.input_dir)
        except ValueError:
            parent = Path()
        return self.output_dir / parent / f"{source.stem}{self.suffix}"

    def plan(self, sources: Iterable[Path]) -> dict[Path, Path]:
        """Map every source to its destination, in input order.

        Raises:
            ConfigError: If two inputs map to the same destination (e.g.
                ``IMG_1.CR2`` and ``IMG_1.cr2``), which would overwrite each other.
        """
        self.validate()
        planned: dict[Path, Path] = {}
        seen: dict[str, Path] = {}
        for source in sources:
            dest = self.destination(source)
            key = dest.as_posix().casefold()
            if key in seen:
                msg = f"{source} and {seen[key]} would both be written to {dest}.*"
                raise ConfigError(msg)
            seen[key] = source
            planned[source] = dest
        return planned

    def check_writable(self, path: Path) -> Path:
        """Return ``path`` if it may be written (see :func:`ensure_writable`)."""
        return ensure_writable(path, overwrite=self.overwrite)
