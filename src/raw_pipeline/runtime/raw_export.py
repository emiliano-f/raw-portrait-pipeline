"""``fix-raw``: write a corrected DNG per input, independent of the image pipeline (ADR 0003).

Per image: source -> read CFA -> map coordinates -> correct -> write -> verify.
Nothing is demosaiced except by the preview renderer and LibRaw's open check.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.dng import BaseDng, CFAOffset, DngRawImage, PreviewMode, RawPatch
from raw_pipeline.core.errors import OutputExistsError, PipelineError, VerificationError
from raw_pipeline.core.protocols import DefectCorrector, DngRawWriter, DngSource, RawDecoder
from raw_pipeline.core.raw import CFAFrame, DefectMap, DefectResidual
from raw_pipeline.runtime.services import RawPlacement, RemainingDefects

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES: Final = 1 << 20
TEMP_PREFIX: Final = "raw-pipeline-fix-raw-"
PASSED: Final = "passed"
PASSED_RESIDUAL: Final = "passed_residual"
"""Verification passed, but a corrected line still scores weakly (next to a thin scene stripe)."""
SKIPPED: Final = "skipped"


@dataclass(frozen=True)
class FixRawRecord:
    """One row of the ``fix-raw`` CSV report."""

    input_file: str
    output_file: str = ""
    status: str = "failed"
    source: str = ""
    reference_raw: str = ""
    offset_y: int | None = None
    offset_x: int | None = None
    mapping_method: str = ""
    defect_method: str = ""
    defects_configured: int = 0
    corrected_pixels: int = 0
    passthrough: bool = False
    digest: str = ""
    preview: str = ""
    verification: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the corrected DNG was written and passed verification."""
        return self.status == "ok"

    def as_row(self) -> dict[str, object]:
        """The record as a CSV row (``None`` as empty)."""
        return {k: ("" if v is None else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class _Corrected:
    base: BaseDng
    raw: DngRawImage
    offset: CFAOffset
    method: str
    frame: CFAFrame
    corrected: CFAFrame
    count: int


def file_digest(path: Path) -> str:
    """SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class RawExportService:
    """Writes corrected DNGs with the configured defect corrector (ADR 0002 + 0003).

    Args:
        decoder: Opens the reference CR2 (LibRaw coordinates) and checks the output opens.
        source: Produces the base DNG.
        writer: Reads and patches DNG raw data.
        corrector: The configured ``defects.method``.
        defects: The configured ``defects.map``.
        placement: Coordinate mapping between reference and DNG.
        remaining_defects: Detector scoring of the corrected frame.
        detection: ``defects.detection`` settings for that scoring.
        preview: Preview handling.
        verify: Run the post-write checks.
    """

    def __init__(
        self,
        decoder: RawDecoder,
        source: DngSource,
        writer: DngRawWriter,
        corrector: DefectCorrector,
        defects: DefectMap,
        *,
        placement: RawPlacement,
        remaining_defects: RemainingDefects,
        detection: DefectDetectionConfig,
        preview: PreviewMode = PreviewMode.REGENERATE,
        verify: bool = True,
    ) -> None:
        self._decoder = decoder
        self._source = source
        self._writer = writer
        self._corrector = corrector
        self._defects = defects
        self._placement = placement
        self._remaining = remaining_defects
        self._detection = detection
        self._preview = preview
        self._verify = verify

    def export(self, path: Path, destination: Path, *, overwrite: bool) -> FixRawRecord:
        """Write the corrected DNG for ``path`` to ``destination``; never raises for image failures."""
        record = FixRawRecord(input_file=str(path), source=self._source.name, defect_method=self._corrector.name)
        try:
            return self._export(path, destination, record, overwrite=overwrite)
        except (PipelineError, OSError, ValueError) as exc:
            logger.error("%s: %s", path.name, exc)
            return _with(record, error=str(exc))

    def _export(self, path: Path, destination: Path, record: FixRawRecord, *, overwrite: bool) -> FixRawRecord:
        with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as work:
            base = self._source.prepare(path, Path(work))
            protected = {path, base.reference} | ({base.path} if base.path == path else set())
            if destination.exists() and any(destination.samefile(p) for p in protected):
                msg = f"{destination} is one of the inputs; choose another output folder"
                raise OutputExistsError(msg)
            hashes = {p: file_digest(p) for p in protected}
            fixed = self._correct(base)
            record = _with(
                record,
                reference_raw=str(base.reference),
                offset_y=fixed.offset.y,
                offset_x=fixed.offset.x,
                mapping_method=fixed.method,
                defects_configured=len(self._defects.defects),
                corrected_pixels=fixed.count,
                passthrough=fixed.count == 0,
            )
            changed = fixed.corrected.data != fixed.frame.data
            ys, xs = np.nonzero(changed)
            patch = RawPatch(
                (ys + fixed.offset.y).astype(np.intp),
                (xs + fixed.offset.x).astype(np.intp),
                fixed.corrected.data[changed].astype(np.uint16),
            )
            residuals: list[DefectResidual] = []
            result = self._writer.write(
                base.path,
                destination,
                patch,
                preview=self._preview,
                overwrite=overwrite,
                check=(lambda out: residuals.extend(self._check(fixed, out, hashes))) if self._verify else None,
            )
            record = _with(record, output_file=str(result.path), digest=str(result.digest), preview=str(result.preview))
            verification = (PASSED_RESIDUAL if residuals else PASSED) if self._verify else SKIPPED
        if record.passthrough:
            logger.info("%s: passthrough (no photosite changed)", path.name)
        return _with(record, status="ok", verification=verification)

    def _correct(self, base: BaseDng) -> _Corrected:
        raw = self._writer.read_raw(base.path)
        with self._decoder.open(base.reference) as handle:
            reference = handle.read_cfa()
        placement = self._placement.locate(reference, raw)
        frame = self._placement.visible_frame(reference, raw, placement.offset)
        result = self._corrector.correct(frame, self._defects)
        count = int(np.count_nonzero(result.frame.data != frame.data))
        return _Corrected(base, raw, placement.offset, placement.method, frame, result.frame, count)

    def _check(self, fixed: _Corrected, output: Path, hashes: dict[Path, str]) -> list[DefectResidual]:
        """Raise :class:`VerificationError` on any problem; return the weak residuals that were tolerated."""
        problems: list[str] = []
        written = self._writer.read_raw(output)
        if written.data.shape != fixed.raw.data.shape or written.cfa_pattern != fixed.raw.cfa_pattern:
            problems.append("raw IFD layout changed")
        else:
            expected = fixed.raw.data.copy()
            height, width = fixed.frame.data.shape
            window = (slice(fixed.offset.y, fixed.offset.y + height), slice(fixed.offset.x, fixed.offset.x + width))
            expected[window] = fixed.corrected.data
            mask = np.zeros(expected.shape, dtype=np.bool_)
            if fixed.count:  # passthrough: the whole raw image must equal the base
                mask[window] = self._defects.mask(height, width)
            problems.extend(_compare(written.data, fixed.raw.data, expected, mask))
        try:
            with self._decoder.open(output) as handle:
                handle.read_cfa()
        except PipelineError as exc:
            problems.append(f"LibRaw cannot open the output: {exc}")
        residuals = self._remaining(fixed.frame, fixed.corrected, self._defects, self._detection) if fixed.count else []
        problems.extend(
            f"defect still detected: {r.defect} (score {r.score_before:.1f} -> {r.score_after:.1f})"
            for r in residuals
            if not r.improved
        )
        problems.extend(f"input changed: {p}" for p, digest in hashes.items() if file_digest(p) != digest)
        if problems:
            msg = "verification failed (nothing written): " + "; ".join(problems)
            raise VerificationError(msg)
        for r in residuals:
            logger.warning(
                "%s: %s still scores %.1f after correction (%.1f before); accepted as a residual",
                fixed.base.reference.name,
                r.defect,
                r.score_after,
                r.score_before,
            )
        return residuals


def _compare(
    written: NDArray[np.uint16], base: NDArray[np.uint16], expected: NDArray[np.uint16], mask: NDArray[np.bool_]
) -> list[str]:
    problems = []
    if not np.array_equal(written[~mask], base[~mask]):
        problems.append(
            f"{int(np.count_nonzero(written[~mask] != base[~mask]))} photosites outside the defect mask differ"
        )
    if not np.array_equal(written[mask], expected[mask]):
        problems.append("photosites inside the defect mask differ from the corrector output")
    return problems


def _with(record: FixRawRecord, **changes: object) -> FixRawRecord:
    return replace(record, **changes)  # type: ignore[arg-type]
